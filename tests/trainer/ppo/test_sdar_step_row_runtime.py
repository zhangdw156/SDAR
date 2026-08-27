import sys
import types
from unittest.mock import patch

import numpy as np
import pytest
import torch
from omegaconf import OmegaConf

from verl import DataProto
from agent_system.multi_turn_rollout import TrajectoryCollector, adjust_batch
from verl.trainer.ppo import ray_trainer
from verl.trainer.ppo import skillsd_ray_trainer


class _StopAfterActorUpdate(Exception):
    pass


class _ActorRolloutStub:
    world_size = 1

    def compute_log_prob(self, batch):
        return DataProto.from_dict(
            tensors={
                "entropys": torch.zeros_like(
                    batch.batch["responses"],
                    dtype=torch.float32,
                ),
                "old_log_probs": torch.zeros_like(
                    batch.batch["responses"],
                    dtype=torch.float32,
                ),
            }
        )

    def update_actor(self, batch):
        assert "advantages" in batch.batch
        assert "returns" in batch.batch
        assert torch.count_nonzero(batch.batch["advantages"]).item() > 0
        torch.testing.assert_close(
            batch.batch["returns"],
            batch.batch["advantages"],
        )
        assert batch.meta_info["multi_turn"] is False
        if "teacher_log_probs" in batch.batch:
            assert batch.batch["teacher_log_probs"].shape == batch.batch[
                "old_log_probs"
            ].shape
        raise _StopAfterActorUpdate


def _config():
    return OmegaConf.create(
        {
            "trainer": {
                "project_name": "test",
                "experiment_name": "test",
                "logger": "console",
                "val_before_train": False,
                "total_epochs": 1,
                "balance_batch": False,
                "critic_warmup": 0,
                "test_freq": 0,
                "save_freq": 0,
                "rollout_data_dir": None,
            },
            "algorithm": {
                "adv_estimator": "grpo",
                "gamma": 1.0,
                "lam": 1.0,
                "norm_adv_by_std_in_grpo": True,
                "use_kl_in_reward": False,
                "use_pf_ppo": False,
                "pf_ppo": {
                    "reweight_method": "pow",
                    "weight_pow": 2.0,
                },
                "gigpo": {
                    "step_advantage_w": 1.0,
                    "mode": "mean_std_norm",
                    "enable_similarity": False,
                    "similarity_thresh": 0.95,
                },
                "trajectory_grpo": {
                    "scheduler": "row",
                    "reducer": "token_mean",
                    "advantage": "step_row",
                    "penalty": "step_local",
                    "filter": "off",
                },
            },
            "actor_rollout_ref": {
                "actor": {
                    "loss_agg_mode": "token-mean",
                    "use_invalid_action_penalty": False,
                },
                "rollout": {
                    "n": 2,
                    "multi_turn": {"enable": False},
                },
            },
            "reward_model": {"launch_reward_fn_async": False},
        }
    )


def _rollout_batch():
    responses = torch.tensor([[11, 12], [21, 22]])
    prompts = torch.tensor([[1, 2], [3, 4]])
    return DataProto.from_dict(
        tensors={
            "prompts": prompts,
            "responses": responses,
            "input_ids": torch.cat((prompts, responses), dim=-1),
            "attention_mask": torch.ones((2, 4), dtype=torch.long),
            "position_ids": torch.arange(4).repeat(2, 1),
        },
        non_tensors={
            "uid": np.asarray(["group", "group"], dtype=object),
            "traj_uid": np.asarray(["traj-0", "traj-1"], dtype=object),
        },
    )


def _trainer(trainer_cls):
    trainer = object.__new__(trainer_cls)
    trainer.config = _config()
    trainer.val_reward_fn = None
    trainer.total_training_steps = 1
    trainer.train_dataloader = [
        {
            "input_ids": torch.ones((1, 2), dtype=torch.long),
            "attention_mask": torch.ones((1, 2), dtype=torch.long),
            "position_ids": torch.arange(2).unsqueeze(0),
            "raw_prompt_ids": np.asarray([[1, 2]], dtype=object),
            "data_source": np.asarray(["text"], dtype=object),
        }
    ]
    rollout_batch = _rollout_batch()
    trainer.traj_collector = types.SimpleNamespace(
        multi_turn_loop=lambda **_kwargs: rollout_batch
    )
    trainer.actor_rollout_wg = _ActorRolloutStub()
    trainer.envs = object()
    trainer.reward_fn = object()
    trainer.use_rm = False
    trainer.use_reference_policy = False
    trainer.use_critic = False
    trainer.ref_in_actor = False
    trainer._load_checkpoint = types.MethodType(lambda _self: None, trainer)
    trainer.sdl_lambda = 0.01
    trainer.sdl_warmdown_steps = -1
    trainer._compute_teacher_log_probs = types.MethodType(
        lambda _self, batch: torch.zeros_like(
            batch.batch["responses"],
            dtype=torch.float32,
        ),
        trainer,
    )
    return trainer


@pytest.mark.parametrize(
    ("trainer_cls", "trainer_module"),
    [
        (ray_trainer.RayPPOTrainer, ray_trainer),
        (skillsd_ray_trainer.SkillSDRayTrainer, skillsd_ray_trainer),
    ],
)
def test_step_row_fit_populates_actor_advantages(
    monkeypatch,
    trainer_cls,
    trainer_module,
):
    trainer = _trainer(trainer_cls)
    tracking_stub = types.ModuleType("verl.utils.tracking")
    tracking_stub.Tracking = type(
        "Tracking",
        (),
        {"__init__": lambda self, **_kwargs: None},
    )
    monkeypatch.setattr(
        trainer_module,
        "adjust_batch",
        lambda _config, batch: batch,
    )
    monkeypatch.setattr(
        trainer_module,
        "compute_reward",
        lambda _batch, _reward_fn: (
            torch.tensor([[0.0, 0.0], [1.0, 0.0]]),
            {},
        ),
    )
    monkeypatch.setattr(
        trainer_module,
        "tqdm",
        lambda **_kwargs: types.SimpleNamespace(),
    )

    with patch.dict(sys.modules, {"verl.utils.tracking": tracking_stub}):
        with pytest.raises(_StopAfterActorUpdate):
            trainer.fit()


def test_step_row_runtime_rejects_unconsumed_trajectory_override():
    config = _config()
    config.algorithm.trajectory_grpo.advantage = "trajectory"

    with pytest.raises(NotImplementedError, match="canonical step_row"):
        ray_trainer.compute_step_row_advantage(_rollout_batch(), config)


def test_rollout_and_batch_adjustment_consume_trajectory_config():
    config = _config()
    config.algorithm.trajectory_grpo.scheduler = "invalid"

    with pytest.raises(ValueError, match="trajectory_grpo.scheduler"):
        TrajectoryCollector(config=config, tokenizer=None)
    with pytest.raises(ValueError, match="trajectory_grpo.scheduler"):
        adjust_batch(config, _rollout_batch())
