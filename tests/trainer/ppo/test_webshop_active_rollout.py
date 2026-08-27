# Copyright 2026 The verl-agent team.
#
# Licensed under the Apache License, Version 2.0 (the "License");

import sys
import types
from importlib.machinery import ModuleSpec

import numpy as np
import torch
from omegaconf import OmegaConf

if "datasets" not in sys.modules:
    datasets_stub = types.ModuleType("datasets")
    datasets_stub.__spec__ = ModuleSpec("datasets", loader=None)
    datasets_stub.Dataset = object
    sys.modules["datasets"] = datasets_stub

if "gym" not in sys.modules:
    gym_stub = types.ModuleType("gym")
    gym_stub.__spec__ = ModuleSpec("gym", loader=None)
    gym_stub.Env = object
    sys.modules["gym"] = gym_stub

if "gymnasium" not in sys.modules:
    gymnasium_stub = types.ModuleType("gymnasium")
    gymnasium_stub.__spec__ = ModuleSpec("gymnasium", loader=None)
    gymnasium_stub.Env = object
    gymnasium_stub.spaces = types.ModuleType("gymnasium.spaces")
    sys.modules["gymnasium"] = gymnasium_stub
    sys.modules["gymnasium.spaces"] = gymnasium_stub.spaces

if "torchvision" not in sys.modules:
    transforms_stub = types.ModuleType("torchvision.transforms")
    transforms_stub.Compose = lambda transforms: transforms
    transforms_stub.ToTensor = object
    torchvision_stub = types.ModuleType("torchvision")
    torchvision_stub.__spec__ = ModuleSpec("torchvision", loader=None)
    torchvision_stub.transforms = transforms_stub
    sys.modules["torchvision"] = torchvision_stub
    sys.modules["torchvision.transforms"] = transforms_stub

from agent_system.environments.env_manager import WebshopEnvironmentManager  # noqa: E402
from agent_system.environments.env_package.webshop.envs import WebshopMultiProcessEnv  # noqa: E402
from agent_system.multi_turn_rollout.rollout_loop import TrajectoryCollector  # noqa: E402
from verl import DataProto  # noqa: E402


def _config():
    return OmegaConf.create(
        {
            "env": {
                "max_steps": 3,
                "history_length": 10,
                "rollout": {"n": 1},
            },
            "algorithm": {
                "filter_groups": {"enable": False},
                "trajectory_grpo": {
                    "scheduler": "row",
                    "advantage": "step_row",
                    "filter": "off",
                },
            },
            "data": {"train_batch_size": 3},
        }
    )


def _info(index, done=False):
    return {
        "available_actions": {
            "has_search_bar": True,
            "clickables": [f"item-{index}"],
        },
        "won": bool(done),
        "task_score": float(done),
    }


def _observation(index, suffix):
    return (
        "WebShop [SEP] Instruction: [SEP] "
        f"task {index} [SEP] page-{index}-{suffix}"
    )


class FakeWebshopVectorEnv:
    def __init__(self, staggered=False):
        self.selected_calls = []
        self.staggered = staggered
        self.step_count = 0

    def reset(self):
        self.step_count = 0
        return (
            [_observation(index, "reset") for index in range(3)],
            [_info(index) for index in range(3)],
        )

    def step_selected(self, actions, indices):
        self.selected_calls.append((list(actions), list(indices)))
        self.step_count += 1
        dones = np.array(
            [
                self.staggered and index == self.step_count - 1
                for index in indices
            ],
            dtype=bool,
        )
        return (
            [
                _observation(index, action)
                for action, index in zip(actions, indices)
            ],
            np.ones(len(indices), dtype=np.float32),
            dones,
            [_info(index, done) for index, done in zip(indices, dones)],
        )


class FakeTokenizer:
    def batch_decode(self, responses, skip_special_tokens=True):
        return [f"act-{int(response[0])}" for response in responses]


class FakeActor:
    world_size = 1

    def __init__(self):
        self.generation_batch_sizes = []

    def generate_sequences(self, batch):
        self.generation_batch_sizes.append(len(batch))
        return DataProto.from_dict(
            tensors={
                **{key: value.clone() for key, value in batch.batch.items()},
                "responses": torch.zeros((len(batch), 1), dtype=torch.long),
            },
            non_tensors={
                key: value.copy()
                for key, value in batch.non_tensor_batch.items()
            },
        )


def _gen_batch():
    return DataProto.from_dict(
        tensors={"input_ids": torch.arange(3).reshape(-1, 1)},
        non_tensors={
            "raw_prompt": np.array(
                [[{"role": "user", "content": "prompt"}]] * 3,
                dtype=object,
            ),
            "data_source": np.array(["webshop"] * 3, dtype=object),
        },
    )


def _fake_preprocess(gen_batch, obs):
    size = len(gen_batch)
    return DataProto.from_dict(
        tensors={
            "input_ids": torch.arange(size).reshape(-1, 1),
            "attention_mask": torch.ones((size, 1), dtype=torch.long),
            "position_ids": torch.arange(size).reshape(-1, 1),
        },
        non_tensors={
            "raw_prompt_ids": np.array([[i] for i in range(size)], dtype=object),
            "observation": np.array(obs["text"], dtype=object),
            "index": np.arange(size, dtype=object),
        },
    )


def test_webshop_collector_compacts_finished_trajectories_without_sessions():
    envs = FakeWebshopVectorEnv(staggered=True)
    manager = WebshopEnvironmentManager(
        envs,
        lambda actions: (list(actions), [True] * len(actions)),
        _config(),
    )
    actor = FakeActor()
    collector = TrajectoryCollector(_config(), FakeTokenizer())
    collector.preprocess_batch = _fake_preprocess

    trajectories, rewards, lengths, _, _, _ = collector.vanilla_multi_turn_loop(
        _gen_batch(),
        actor,
        manager,
    )

    assert actor.generation_batch_sizes == [3, 2, 1]
    assert [indices for _, indices in envs.selected_calls] == [
        [0, 1, 2],
        [1, 2],
        [2],
    ]
    assert [len(rows) for rows in trajectories] == [1, 2, 3]
    assert [len(manager.memory[index]) for index in range(3)] == [1, 2, 3]
    np.testing.assert_array_equal(rewards, [1.0, 2.0, 3.0])
    np.testing.assert_array_equal(lengths, [1.0, 2.0, 3.0])


def test_webshop_manager_keeps_selected_state_on_original_indices():
    envs = FakeWebshopVectorEnv()
    manager = WebshopEnvironmentManager(
        envs,
        lambda actions: (list(actions), [True] * len(actions)),
        _config(),
    )
    manager.reset(kwargs=None)

    first, _, _, infos = manager.step_selected(["act-2", "act-0"], [2, 0])
    later, _, _, _ = manager.step_selected(["act-2b"], [2])

    assert [len(manager.memory[index]) for index in range(3)] == [1, 0, 2]
    assert "page-0-reset" in manager.memory[0][0]["text_obs"]
    assert "page-2-reset" in manager.memory[2][0]["text_obs"]
    assert "page-0-act-0" in manager.pre_text_obs[0]
    assert "page-1-reset" in manager.pre_text_obs[1]
    assert "page-2-act-2b" in manager.pre_text_obs[2]
    assert "task 2" in first["text"][0]
    assert "task 0" in first["text"][1]
    assert "act-2" in later["text"][0]
    assert all(bool(info["is_action_valid"]) for info in infos)


class FakeRemoteMethod:
    def __init__(self, result):
        self.result = result
        self.calls = []

    def remote(self, action):
        self.calls.append(action)
        return self.result


def test_webshop_env_steps_only_selected_workers(monkeypatch):
    env = object.__new__(WebshopMultiProcessEnv)
    env.num_processes = 3
    env._closed = True
    env._workers = []
    for index in range(3):
        env._workers.append(
            types.SimpleNamespace(
                step=FakeRemoteMethod(
                    (
                        f"obs-{index}",
                        float(index),
                        False,
                        _info(index),
                    )
                )
            )
        )

    monkeypatch.setattr(
        "agent_system.environments.env_package.webshop.envs.ray.get",
        lambda futures: futures,
    )

    obs, rewards, dones, _ = env.step_selected(
        ["action-2", "action-0"],
        [2, 0],
    )

    assert obs == ["obs-2", "obs-0"]
    assert rewards == [2.0, 0.0]
    assert dones == [False, False]
    assert env._workers[0].step.calls == ["action-0"]
    assert env._workers[1].step.calls == []
    assert env._workers[2].step.calls == ["action-2"]
