import json
import sys
import types

from omegaconf import OmegaConf

from agent_system.environments import env_manager
from agent_system.environments.env_manager import CanonicalValidationEnvironments


def _write_jsonl(path, rows):
    path.write_text(
        "".join(f"{json.dumps(row)}\n" for row in rows),
        encoding="utf-8",
    )


class _FakeManager:
    def __init__(self, raw_envs, _projection, _config):
        self.raw_envs = raw_envs
        self.closed = False

    def close(self):
        self.closed = True


def test_alfworld_canonical_validation_builds_full_seen_and_unseen_chunks(
    tmp_path,
    monkeypatch,
):
    fairness_dir = tmp_path / "alfworld"
    fairness_dir.mkdir()
    for split, count in (
        ("evaluation_seen", 140),
        ("evaluation_unseen", 134),
    ):
        _write_jsonl(
            fairness_dir / f"{split}.jsonl",
            [
                {
                    "metadata": {
                        "gamefile": f"${{ALFWORLD_DATA}}/{split}-{index}"
                    }
                }
                for index in range(count)
            ],
        )
    monkeypatch.setenv("ALFWORLD_FAIRNESS_DIR", str(fairness_dir))

    calls = []

    def builder(*args, **kwargs):
        calls.append((args, kwargs))
        return {"args": args, "kwargs": kwargs}

    package = types.ModuleType(
        "agent_system.environments.env_package.alfworld"
    )
    setattr(package, "alfworld_projection", lambda actions, *_args: actions)
    setattr(package, "build_alfworld_envs", builder)
    monkeypatch.setitem(
        sys.modules,
        "agent_system.environments.env_package.alfworld",
        package,
    )

    config = OmegaConf.create(
        {"env": {"env_name": "alfworld/AlfredTWEnv", "seed": 0}}
    )
    environments = CanonicalValidationEnvironments(
        config,
        resources_per_worker={"num_cpus": 0.1},
        alfworld_builder=builder,
        alfworld_manager_cls=_FakeManager,
        alfworld_config_path="config_tw.yaml",
    )

    chunks = list(environments.iter_chunks())

    assert [
        (chunk.split, chunk.metric_prefix, chunk.task_count)
        for chunk in chunks
    ] == [
        ("evaluation_seen", "seen/", 128),
        ("evaluation_seen", "seen/", 12),
        ("evaluation_unseen", "unseen/", 128),
        ("evaluation_unseen", "unseen/", 6),
    ]
    assert len(calls) == 4
    assert all(call[1]["env_kwargs"]["fairness"] is True for call in calls)
    assert all(getattr(chunk.manager, "closed") for chunk in chunks)


def test_webshop_canonical_validation_builds_all_500_goals(
    tmp_path,
    monkeypatch,
):
    fairness_dir = tmp_path / "webshop"
    fairness_dir.mkdir()
    _write_jsonl(
        fairness_dir / "evaluation.jsonl",
        [
            {"metadata": {"goal_idx": index}}
            for index in range(500)
        ],
    )
    monkeypatch.setenv("WEBSHOP_FAIRNESS_DIR", str(fairness_dir))

    calls = []

    def builder(*args, **kwargs):
        calls.append((args, kwargs))
        return {"args": args, "kwargs": kwargs}

    package = types.ModuleType(
        "agent_system.environments.env_package.webshop"
    )
    setattr(package, "webshop_projection", lambda actions, *_args: actions)
    setattr(package, "build_webshop_envs", builder)
    monkeypatch.setitem(
        sys.modules,
        "agent_system.environments.env_package.webshop",
        package,
    )

    config = OmegaConf.create(
        {
            "env": {
                "env_name": "Webshop",
                "seed": 0,
                "webshop": {"use_small": False, "human_goals": 0},
            }
        }
    )
    environments = CanonicalValidationEnvironments(
        config,
        resources_per_worker={"num_cpus": 0.1},
        webshop_builder=builder,
        webshop_manager_cls=_FakeManager,
    )

    chunks = list(environments.iter_chunks())

    assert [chunk.task_count for chunk in chunks] == [128, 128, 128, 116]
    assert [
        goal_idx
        for _, call in calls
        for goal_idx in call["env_kwargs"]["fairness_goal_indices"]
    ] == list(range(500))
    assert all(call[1]["env_kwargs"]["fairness"] is True for call in calls)
    assert all(getattr(chunk.manager, "closed") for chunk in chunks)


def test_webshop_lazy_reacquire_continues_training_rng(monkeypatch):
    rngs = []
    draws = []

    class RawPool:
        def __init__(self, draw):
            self.draw = draw

        def close(self):
            pass

    class Manager:
        def __init__(self, raw_pool, _projection, _config):
            self.raw_pool = raw_pool

        def identity(self):
            return self.raw_pool.draw

        def close(self):
            self.raw_pool.close()

    def builder(*, is_train, rng=None, **_kwargs):
        assert is_train
        rngs.append(rng)
        draw = int(rng.randint(0, 2**31))
        draws.append(draw)
        return RawPool(draw)

    package = types.ModuleType(
        "agent_system.environments.env_package.webshop"
    )
    setattr(package, "webshop_projection", lambda actions, *_args: actions)
    setattr(package, "build_webshop_envs", builder)
    monkeypatch.setitem(
        sys.modules,
        "agent_system.environments.env_package.webshop",
        package,
    )
    monkeypatch.setattr(
        env_manager,
        "WebshopEnvironmentManager",
        Manager,
    )

    config = OmegaConf.create(
        {
            "env": {
                "env_name": "Webshop",
                "seed": 0,
                "fairness": True,
                "rollout": {"n": 8},
                "resources_per_worker": {"num_cpus": 0.1},
                "webshop": {"use_small": True, "human_goals": 0},
            },
            "data": {"train_batch_size": 16, "val_batch_size": 128},
            "trainer": {"val_only": False},
        }
    )

    train_envs, _validation_envs = env_manager.make_envs(config)
    first = train_envs.identity()
    train_envs.release()
    second = train_envs.identity()
    train_envs.close()

    assert len(rngs) == 2
    assert rngs[0] is rngs[1]
    assert draws == [first, second]
    assert first != second
