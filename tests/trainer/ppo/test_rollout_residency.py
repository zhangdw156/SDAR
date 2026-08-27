import importlib
import sys
from types import ModuleType, SimpleNamespace

import pytest

import verl.workers.fsdp_workers as fsdp_workers
from verl.single_controller.base.decorator import MAGIC_ATTR, Dispatch
from verl.workers.fsdp_workers import (
    ActorRolloutRefWorker,
    AsyncActorRolloutRefWorker,
)


class FakeData:
    def __init__(self):
        self.meta_info = {}
        self.to_calls = []

    def to(self, device):
        self.to_calls.append(device)
        return self


class FakeShardingManager:
    supports_rollout_session = True

    def __init__(self):
        self.enter_count = 0
        self.exit_count = 0
        self.preprocess_count = 0
        self.postprocess_count = 0
        self.enter_error = None
        self.exit_error = None
        self._tainted = False

    def __enter__(self):
        self.enter_count += 1
        if self.enter_error is not None:
            raise self.enter_error
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.exit_count += 1
        if self.exit_error is not None:
            self._tainted = True
            raise self.exit_error

    def preprocess_data(self, data):
        self.preprocess_count += 1
        return data

    def postprocess_data(self, data):
        self.postprocess_count += 1
        return data


class FakeRollout:
    def __init__(self):
        self.generate_count = 0

    def generate_sequences(self, prompts):
        self.generate_count += 1
        return prompts


class FakeDevice:
    def __init__(self):
        self.empty_cache_count = 0
        self.empty_cache_error = None

    def current_device(self):
        return "fake-device"

    def empty_cache(self):
        self.empty_cache_count += 1
        if self.empty_cache_error is not None:
            raise self.empty_cache_error

    def memory_allocated(self):
        return 0

    def memory_reserved(self):
        return 0

    def mem_get_info(self):
        return 1, 1


@pytest.fixture
def worker_and_device(monkeypatch):
    device = FakeDevice()
    monkeypatch.setattr(fsdp_workers, "get_torch_device", lambda: device)
    monkeypatch.setattr(
        fsdp_workers,
        "log_gpu_memory_usage",
        lambda *args, **kwargs: None,
    )

    worker = object.__new__(ActorRolloutRefWorker)
    worker._is_rollout = True
    worker._is_actor = True
    worker._rollout_session_entering = False
    worker._rollout_session_active = False
    worker._rollout_session_tainted = False
    worker.rollout_sharding_manager = FakeShardingManager()
    worker.rollout = FakeRollout()
    worker.generation_config = SimpleNamespace(
        eos_token_id=2,
        pad_token_id=0,
    )
    worker.tokenizer = SimpleNamespace(eos_token_id=3, pad_token_id=1)
    return worker, device


def test_session_reuses_one_sharding_context_for_multiple_generations(
    worker_and_device,
):
    worker, device = worker_and_device

    worker.begin_rollout_session()
    first = worker.generate_sequences(FakeData())
    second = worker.generate_sequences(FakeData())
    worker.end_rollout_session()

    assert worker.rollout_sharding_manager.enter_count == 1
    assert worker.rollout_sharding_manager.exit_count == 1
    assert worker.rollout_sharding_manager.preprocess_count == 2
    assert worker.rollout_sharding_manager.postprocess_count == 2
    assert worker.rollout.generate_count == 2
    assert first.to_calls == ["fake-device", "cpu"]
    assert second.to_calls == ["fake-device", "cpu"]
    assert device.empty_cache_count == 1


def test_session_rpcs_are_one_to_all_and_async_worker_rejects_them():
    begin_attrs = getattr(ActorRolloutRefWorker.begin_rollout_session, MAGIC_ATTR)
    end_attrs = getattr(ActorRolloutRefWorker.end_rollout_session, MAGIC_ATTR)
    assert begin_attrs["dispatch_mode"] == Dispatch.ONE_TO_ALL
    assert end_attrs["dispatch_mode"] == Dispatch.ONE_TO_ALL

    async_worker = object.__new__(AsyncActorRolloutRefWorker)
    with pytest.raises(NotImplementedError, match="does not support"):
        async_worker.begin_rollout_session()
    with pytest.raises(NotImplementedError, match="does not support"):
        async_worker.end_rollout_session()


def test_unsupported_manager_keeps_per_generation_context(worker_and_device):
    worker, device = worker_and_device
    worker.rollout_sharding_manager.supports_rollout_session = False

    worker.begin_rollout_session()
    worker.generate_sequences(FakeData())
    worker.generate_sequences(FakeData())
    worker.end_rollout_session()

    assert not worker._rollout_session_active
    assert worker.rollout_sharding_manager.enter_count == 2
    assert worker.rollout_sharding_manager.exit_count == 2
    assert device.empty_cache_count == 2


def test_failed_begin_clears_state_and_preserves_primary_cleanup_error(
    worker_and_device,
):
    worker, device = worker_and_device
    enter_error = ValueError("enter failed")
    worker.rollout_sharding_manager.enter_error = enter_error
    device.empty_cache_error = RuntimeError("cache cleanup failed")

    with pytest.raises(ValueError, match="enter failed") as exc_info:
        worker.begin_rollout_session()

    assert exc_info.value is enter_error
    assert enter_error.rollout_session_cleanup_error is device.empty_cache_error
    assert not worker._rollout_session_entering
    assert not worker._rollout_session_active
    assert worker._rollout_session_tainted


@pytest.mark.parametrize(
    "state_name",
    [
        "_rollout_session_entering",
        "_rollout_session_active",
        "_rollout_session_tainted",
    ],
)
def test_update_actor_rejects_non_clean_rollout_session(
    worker_and_device,
    state_name,
):
    worker, _ = worker_and_device
    setattr(worker, state_name, True)

    with pytest.raises(RuntimeError, match="entering, active, or tainted"):
        worker.update_actor(FakeData())


def test_end_failure_clears_active_state_and_taints_worker(worker_and_device):
    worker, device = worker_and_device
    worker.begin_rollout_session()
    worker.rollout_sharding_manager.exit_error = ValueError("exit failed")

    with pytest.raises(ValueError, match="exit failed"):
        worker.end_rollout_session()

    assert not worker._rollout_session_active
    assert worker._rollout_session_tainted
    assert device.empty_cache_count == 1
    with pytest.raises(RuntimeError, match="tainted"):
        worker.update_actor(FakeData())


@pytest.fixture
def fsdp_vllm_module(monkeypatch):
    third_party_vllm = ModuleType("verl.third_party.vllm")
    third_party_vllm.LLM = object
    third_party_vllm.vllm_version = "0.8.0"
    third_party_vllm.parallel_state = SimpleNamespace()

    vllm_utils = ModuleType("verl.utils.vllm_utils")
    vllm_utils.TensorLoRARequest = object
    vllm_utils.VLLMHijack = SimpleNamespace(hijack=lambda: None)
    vllm_utils.is_version_ge = lambda **kwargs: False
    vllm_utils.patch_vllm_moe_model_weight_loader = lambda model: None

    monkeypatch.setitem(sys.modules, "verl.third_party.vllm", third_party_vllm)
    monkeypatch.setitem(sys.modules, "verl.utils.vllm_utils", vllm_utils)
    sys.modules.pop("verl.workers.sharding_manager.fsdp_vllm", None)
    module = importlib.import_module(
        "verl.workers.sharding_manager.fsdp_vllm"
    )
    monkeypatch.setattr(
        module,
        "log_gpu_memory_usage",
        lambda *args, **kwargs: None,
    )
    return module


class FakeRNGDevice:
    def __init__(self):
        self.state = "train"
        self.set_calls = []
        self.empty_cache_count = 0

    def get_rng_state(self):
        return self.state

    def set_rng_state(self, state):
        self.state = state
        self.set_calls.append(state)

    def empty_cache(self):
        self.empty_cache_count += 1

    def current_device(self):
        return "fake-device"

    def memory_allocated(self):
        return 0

    def memory_reserved(self):
        return 0

    def mem_get_info(self):
        return 1, 1


class FakeFSDPModule:
    def __init__(self):
        self._fsdp_wrapped_module = object()
        self.train_count = 0

    def state_dict(self):
        return {"weight": object()}

    def train(self):
        self.train_count += 1


class FakeVLLMEngine:
    def __init__(self):
        self.awake = False
        self.wake_count = 0
        self.sleep_count = 0
        self.sleep_error = None

    def wake_up(self, tags=None):
        if not self.awake:
            self.awake = True
            self.wake_count += 1

    def sleep(self, level):
        assert level == 1
        self.sleep_count += 1
        if self.sleep_error is not None:
            raise self.sleep_error
        self.awake = False


def _manager(module, device, monkeypatch):
    manager = object.__new__(module.FSDPVLLMShardingManager)
    manager.module = FakeFSDPModule()
    manager.inference_engine = FakeVLLMEngine()
    manager.device_mesh = object()
    manager.offload_param = False
    manager.full_params = False
    manager.layered_summon = False
    manager.base_sync_done = True
    manager.gen_random_states = "generation"
    manager.torch_random_states = "train"
    manager._enter_actor_params_loaded = False
    manager._enter_rollout_awake = False
    manager._enter_rng_switch_attempted = False
    manager._enter_rng_switched = False
    manager._enter_train_mode_restore_needed = False
    manager._enter_cache_cleanup_needed = False
    manager._enter_succeeded = False
    manager._tainted = False
    manager._taint_reason = None
    manager.update_params = lambda params, peft_config=None: None
    monkeypatch.setattr(module, "get_torch_device", lambda: device)
    debug_performance = importlib.import_module("verl.utils.debug.performance")
    monkeypatch.setattr(
        debug_performance,
        "get_torch_device",
        lambda: device,
    )
    return manager


def test_fsdp_vllm_session_keeps_weights_and_cache_resident_until_exit(
    fsdp_vllm_module,
    monkeypatch,
):
    device = FakeRNGDevice()
    manager = _manager(fsdp_vllm_module, device, monkeypatch)

    assert manager.supports_rollout_session
    manager.__enter__()

    assert manager.inference_engine.awake
    assert manager.inference_engine.wake_count == 1
    assert manager.inference_engine.sleep_count == 0
    assert device.state == "generation"

    device.state = "generation-next"
    manager.__exit__(None, None, None)

    assert not manager.inference_engine.awake
    assert manager.inference_engine.sleep_count == 1
    assert manager.gen_random_states == "generation-next"
    assert device.state == "train"
    assert not manager._enter_succeeded
    assert not manager._tainted


def test_fsdp_vllm_cleanup_failure_taints_without_hiding_primary_error(
    fsdp_vllm_module,
    monkeypatch,
):
    device = FakeRNGDevice()
    manager = _manager(fsdp_vllm_module, device, monkeypatch)
    primary = ValueError("weight update failed")
    manager.update_params = lambda params, peft_config=None: (
        _ for _ in ()
    ).throw(primary)
    manager.inference_engine.sleep_error = RuntimeError("sleep failed")

    with pytest.raises(ValueError, match="weight update failed") as exc_info:
        manager.__enter__()

    assert exc_info.value is primary
    assert primary.fsdp_vllm_cleanup_error is manager.inference_engine.sleep_error
    assert manager._tainted
    with pytest.raises(RuntimeError, match="tainted"):
        manager.__enter__()
