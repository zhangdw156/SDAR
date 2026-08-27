# Copyright 2026 The verl-agent team.
#
# Licensed under the Apache License, Version 2.0 (the "License");

import sys
import types
from importlib.machinery import ModuleSpec

import numpy as np
import pytest
import torch
from omegaconf import OmegaConf

if "datasets" not in sys.modules:
    datasets_stub = types.ModuleType("datasets")
    datasets_stub.__spec__ = ModuleSpec("datasets", loader=None)
    datasets_stub.Dataset = object
    sys.modules["datasets"] = datasets_stub

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

alfworld_vendor_stub = types.ModuleType(
    "agent_system.environments.env_package.alfworld.alfworld"
)
alfworld_agents_stub = types.ModuleType(
    "agent_system.environments.env_package.alfworld.alfworld.agents"
)
alfworld_environment_stub = types.ModuleType(
    "agent_system.environments.env_package.alfworld.alfworld.agents.environment"
)
alfworld_environment_stub.get_environment = lambda env_type: None
sys.modules.setdefault(alfworld_vendor_stub.__name__, alfworld_vendor_stub)
sys.modules.setdefault(alfworld_agents_stub.__name__, alfworld_agents_stub)
sys.modules.setdefault(alfworld_environment_stub.__name__, alfworld_environment_stub)

from agent_system.environments.env_manager import AlfWorldEnvironmentManager  # noqa: E402
from agent_system.environments.env_package.alfworld.envs import AlfworldEnvs  # noqa: E402
from agent_system.multi_turn_rollout.rollout_loop import TrajectoryCollector  # noqa: E402
from verl import DataProto  # noqa: E402


def _config(max_steps=3):
    return OmegaConf.create(
        {
            "env": {
                "max_steps": max_steps,
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


def _gen_batch():
    return DataProto.from_dict(
        tensors={"input_ids": torch.arange(3).reshape(-1, 1)},
        non_tensors={
            "raw_prompt": np.array(
                [[{"role": "user", "content": "prompt"}]] * 3,
                dtype=object,
            ),
            "data_source": np.array(["alfworld"] * 3, dtype=object),
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
            "gamefile": np.array(obs.get("gamefile", [None] * size), dtype=object),
            "index": np.arange(size, dtype=object),
        },
    )


class FakeTokenizer:
    def batch_decode(self, responses, skip_special_tokens=True):
        return [f"action-{int(response[0])}" for response in responses]


class FakeActor:
    world_size = 1

    def __init__(self, session=True):
        self.generation_batch_sizes = []
        self.begin_calls = 0
        self.end_calls = 0
        self.generation_error = None
        self.begin_error = None
        self.end_error = None
        if not session:
            self.begin_rollout_session = None
            self.end_rollout_session = None

    def begin_rollout_session(self):
        self.begin_calls += 1
        if self.begin_error is not None:
            raise self.begin_error

    def end_rollout_session(self):
        self.end_calls += 1
        if self.end_error is not None:
            raise self.end_error

    def generate_sequences(self, batch):
        self.generation_batch_sizes.append(len(batch))
        if self.generation_error is not None:
            raise self.generation_error
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


class StaggeredSelectedManager:
    def __init__(self):
        self.step_indices = []
        self.step_count = 0

    def reset(self, kwargs):
        self.step_count = 0
        return {
            "text": ["obs-0", "obs-1", "obs-2"],
            "image": None,
            "anchor": ["anchor-0", "anchor-1", "anchor-2"],
        }, [{"extra.gamefile": f"game-{i}"} for i in range(3)]

    def step_selected(self, actions, indices):
        self.step_indices.append(list(indices))
        self.step_count += 1
        dones = np.array(
            [index == self.step_count - 1 for index in indices],
            dtype=bool,
        )
        return (
            {
                "text": [
                    f"obs-{index}-step-{self.step_count}"
                    for index in indices
                ],
                "image": None,
                "anchor": [
                    f"anchor-{index}-step-{self.step_count}"
                    for index in indices
                ],
            },
            np.array([index + 1 for index in indices], dtype=np.float32),
            dones,
            [
                {
                    "won": float(done),
                    "extra.gamefile": f"game-{index}",
                    "transition": (index, self.step_count, action),
                }
                for action, index, done in zip(actions, indices, dones)
            ],
        )

    def success_evaluator(self, **kwargs):
        return {
            "success_rate": np.array(
                [infos[-1]["won"] for infos in kwargs["total_infos"]],
                dtype=np.float32,
            )
        }


class EquivalentLegacyManager:
    def __init__(self):
        self.step_count = 0

    def reset(self, kwargs):
        self.step_count = 0
        return {
            "text": ["obs-0", "obs-1", "obs-2"],
            "image": None,
            "anchor": ["anchor-0", "anchor-1", "anchor-2"],
        }, [{"extra.gamefile": f"game-{i}"} for i in range(3)]

    def step(self, actions):
        self.step_count += 1
        dones = np.array(
            [index <= self.step_count - 1 for index in range(3)],
            dtype=bool,
        )
        return (
            {
                "text": [
                    f"obs-{index}-step-{self.step_count}"
                    for index in range(3)
                ],
                "image": None,
                "anchor": [
                    f"anchor-{index}-step-{self.step_count}"
                    for index in range(3)
                ],
            },
            np.arange(1, 4, dtype=np.float32),
            dones,
            [
                {
                    "won": float(done),
                    "extra.gamefile": f"game-{index}",
                    "transition": (
                        index,
                        self.step_count,
                        actions[index],
                    ),
                }
                for index, done in enumerate(dones)
            ],
        )

    def success_evaluator(self, **kwargs):
        return {
            "success_rate": np.array(
                [
                    next(
                        info["won"]
                        for row, info in zip(rows, infos)
                        if row["active_masks"] and info["won"]
                    )
                    for rows, infos in zip(
                        kwargs["total_batch_list"],
                        kwargs["total_infos"],
                    )
                ],
                dtype=np.float32,
            )
        }


def _collector():
    collector = TrajectoryCollector(_config(), FakeTokenizer())
    collector.preprocess_batch = _fake_preprocess
    return collector


def _active_transitions(trajectories):
    return [
        [
            (
                int(row["index"]),
                int(row["turn_index"]),
                int(row["responses"][0]),
                float(row["rewards"]),
                str(row["gamefile"]),
            )
            for row in trajectory
            if row["active_masks"]
        ]
        for trajectory in trajectories
    ]


def test_compaction_shrinks_batches_and_preserves_slot_identity_and_step_rows():
    actor = FakeActor()
    envs = StaggeredSelectedManager()

    trajectories, rewards, lengths, _, _, _ = _collector().vanilla_multi_turn_loop(
        _gen_batch(),
        actor,
        envs,
    )

    assert actor.generation_batch_sizes == [3, 2, 1]
    assert envs.step_indices == [[0, 1, 2], [1, 2], [2]]
    assert [len(rows) for rows in trajectories] == [1, 2, 3]
    assert [[row["index"] for row in rows] for rows in trajectories] == [
        [0],
        [1, 1],
        [2, 2, 2],
    ]
    assert [[row["turn_index"] for row in rows] for rows in trajectories] == [
        [0],
        [0, 1],
        [0, 1, 2],
    ]
    assert all(row["gamefile"] == f"game-{slot}" for slot, rows in enumerate(trajectories) for row in rows)
    np.testing.assert_array_equal(rewards, [1.0, 4.0, 9.0])
    np.testing.assert_array_equal(lengths, [1.0, 2.0, 3.0])


def test_compacted_and_legacy_active_transitions_and_rewards_are_equivalent():
    compact = _collector().vanilla_multi_turn_loop(
        _gen_batch(),
        FakeActor(),
        StaggeredSelectedManager(),
    )
    legacy = _collector().vanilla_multi_turn_loop(
        _gen_batch(),
        FakeActor(),
        EquivalentLegacyManager(),
    )

    assert _active_transitions(compact[0]) == _active_transitions(legacy[0])
    np.testing.assert_array_equal(compact[1], legacy[1])
    np.testing.assert_array_equal(compact[2], legacy[2])
    np.testing.assert_array_equal(compact[3]["success_rate"], legacy[3]["success_rate"])


def test_compaction_stays_enabled_without_rollout_session_capability():
    actor = FakeActor(session=False)

    trajectories, *_ = _collector().vanilla_multi_turn_loop(
        _gen_batch(),
        actor,
        StaggeredSelectedManager(),
    )

    assert actor.generation_batch_sizes == [3, 2, 1]
    assert [len(rows) for rows in trajectories] == [1, 2, 3]


def test_rollout_error_wins_when_session_cleanup_also_fails(caplog):
    actor = FakeActor()
    actor.generation_error = RuntimeError("generation failed")
    actor.end_error = ValueError("cleanup failed")

    with pytest.raises(RuntimeError, match="generation failed") as exc_info:
        _collector().vanilla_multi_turn_loop(
            _gen_batch(),
            actor,
            StaggeredSelectedManager(),
        )

    assert actor.begin_calls == 1
    assert actor.end_calls == 1
    assert exc_info.value.rollout_session_cleanup_error is actor.end_error
    assert "cleanup failed" in caplog.text


def test_session_end_is_attempted_when_begin_raises():
    actor = FakeActor()
    actor.begin_error = RuntimeError("partial begin failed")

    with pytest.raises(RuntimeError, match="partial begin failed"):
        _collector().vanilla_multi_turn_loop(
            _gen_batch(),
            actor,
            StaggeredSelectedManager(),
        )

    assert actor.begin_calls == 1
    assert actor.end_calls == 1


class FakeAlfworldVectorEnv:
    def __init__(self):
        self.get_admissible_commands = [["a0"], ["a1"], ["a2"]]
        self.selected_calls = []

    def reset(self):
        return (
            [
                "room 0. Your task is to: task zero",
                "room 1. Your task is to: task one",
                "room 2. Your task is to: task two",
            ],
            None,
            [{"extra.gamefile": f"game-{index}"} for index in range(3)],
        )

    def step_selected(self, actions, indices):
        self.selected_calls.append((list(actions), list(indices)))
        text_obs = []
        infos = []
        for action, index in zip(actions, indices):
            self.get_admissible_commands[index] = [f"next-{index}"]
            text_obs.append(f"result-{index}-{action}")
            infos.append({"extra.gamefile": None, "won": 0})
        return (
            text_obs,
            None,
            np.zeros(len(indices), dtype=np.float32),
            np.zeros(len(indices), dtype=bool),
            infos,
        )


def test_alfworld_manager_keeps_selected_histories_on_original_indices():
    envs = FakeAlfworldVectorEnv()
    manager = AlfWorldEnvironmentManager(
        envs,
        lambda actions, pools: (list(actions), [True] * len(actions)),
        _config(),
    )
    manager.reset(kwargs=None)

    first, _, _, infos = manager.step_selected(["act-2", "act-0"], [2, 0])
    later, _, _, _ = manager.step_selected(["act-2b"], [2])

    assert [len(manager.memory[index]) for index in range(3)] == [1, 0, 2]
    assert manager.pre_text_obs == [
        "result-0-act-0",
        "room 1. Your task is to: task one",
        "result-2-act-2b",
    ]
    assert "task two" in first["text"][0]
    assert "task zero" in first["text"][1]
    assert "act-2" in later["text"][0]
    assert [info["extra.gamefile"] for info in infos] == ["game-2", "game-0"]


class FakeRemoteMethod:
    def __init__(self, result):
        self.result = result
        self.calls = []

    def remote(self, action=None):
        self.calls.append(action)
        return self.result


def test_alfworld_env_steps_only_selected_workers_and_updates_slots(monkeypatch):
    env = object.__new__(AlfworldEnvs)
    env.num_processes = 3
    env.multi_modal = False
    env.prev_admissible_commands = [["old-0"], ["old-1"], ["old-2"]]
    env.workers = []
    for index in range(3):
        env.workers.append(
            types.SimpleNamespace(
                step=FakeRemoteMethod(
                    (
                        [f"obs-{index}"],
                        [0],
                        [False],
                        {
                            "won": [0],
                            "goal_condition_success_rate": [0.0],
                            "admissible_commands": [[f"new-{index}"]],
                        },
                    )
                )
            )
        )

    monkeypatch.setattr(
        "agent_system.environments.env_package.alfworld.envs.ray.get",
        lambda futures: futures,
    )

    text_obs, _, _, _, _ = env.step_selected(
        ["action-2", "action-0"],
        [2, 0],
    )

    assert text_obs == ["obs-2", "obs-0"]
    assert env.workers[0].step.calls == ["action-0"]
    assert env.workers[1].step.calls == []
    assert env.workers[2].step.calls == ["action-2"]
    assert env.prev_admissible_commands == [
        ["new-0"],
        ["old-1"],
        ["new-2"],
    ]
