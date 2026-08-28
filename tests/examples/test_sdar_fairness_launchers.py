import os
import subprocess
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).parents[2]
SIZES = ("1.5b", "3b", "7b")
ENVIRONMENTS = ("alfworld", "webshop")
ROLLOUT_LOGPROB_MICRO_BATCHES = {
    ("1.5b", "alfworld"): "32",
    ("1.5b", "webshop"): "16",
    ("3b", "alfworld"): "16",
    ("3b", "webshop"): "8",
    ("7b", "alfworld"): "4",
    ("7b", "webshop"): "4",
}
ACTOR_AND_REF_MICRO_BATCHES = {
    ("1.5b", "alfworld"): ("64", "64"),
    ("1.5b", "webshop"): ("16", "32"),
    ("3b", "alfworld"): ("32", "32"),
    ("3b", "webshop"): ("8", "16"),
    ("7b", "alfworld"): ("8", "8"),
    ("7b", "webshop"): ("8", "8"),
}


def _launcher(size: str, environment: str) -> Path:
    return REPO_ROOT / "examples" / f"sdar_trainer_{size}" / f"run_{environment}.sh"


def _dry_run(launcher: Path, *overrides: str) -> list[str]:
    environment = os.environ.copy()
    environment["LAUNCHER_DRY_RUN"] = "true"
    completed = subprocess.run(
        [str(launcher), *overrides],
        cwd=REPO_ROOT,
        env=environment,
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.splitlines()


def test_examples_surface_is_paper_only():
    entries = {
        path.name
        for path in (REPO_ROOT / "examples").iterdir()
        if path.name != "__pycache__"
    }
    assert entries == {
        "README.md",
        "__init__.py",
        "data_preprocess",
        "sdar_trainer_1.5b",
        "sdar_trainer_3b",
        "sdar_trainer_7b",
    }
    assert {
        path.name
        for path in (REPO_ROOT / "examples" / "data_preprocess").iterdir()
        if path.name != "__pycache__"
    } == {"__init__.py", "prepare.py"}
    for size in SIZES:
        assert {
            path.name
            for path in (REPO_ROOT / "examples" / f"sdar_trainer_{size}").iterdir()
        } == {"run_alfworld.sh", "run_webshop.sh"}


@pytest.mark.parametrize(
    ("size", "tensor_parallel"),
    (("1.5b", "1"), ("3b", "2"), ("7b", "4")),
)
@pytest.mark.parametrize("environment", ENVIRONMENTS)
def test_launchers_match_fairness_training_contract(
    size: str,
    tensor_parallel: str,
    environment: str,
):
    lines = _dry_run(
        _launcher(size, environment),
        "trainer.experiment_name=cli_override",
    )
    expected = {
        "module=verl.trainer.main_sdar",
        "algorithm.adv_estimator=grpo",
        "algorithm.trajectory_grpo.scheduler=row",
        "algorithm.trajectory_grpo.reducer=token_mean",
        "algorithm.trajectory_grpo.advantage=step_row",
        "algorithm.trajectory_grpo.penalty=step_local",
        "algorithm.trajectory_grpo.filter=off",
        "+algorithm.sdar.sdar_coef=0.01",
        "+algorithm.sdar.gate_beta=5.0",
        f"+algorithm.sdar.skills_dir=skills/{environment}",
        "+algorithm.sdar.skill_all=false",
        "data.seed=0",
        "data.train_batch_size=16",
        "data.val_batch_size=128",
        "actor_rollout_ref.actor.optim.lr=1e-6",
        "++actor_rollout_ref.rollout.seed=0",
        f"actor_rollout_ref.rollout.tensor_model_parallel_size={tensor_parallel}",
        "env.fairness=true",
        "env.seed=0",
        "env.history_length=2",
        "env.rollout.n=8",
        "trainer.n_gpus_per_node=4",
        "trainer.save_freq=10",
        "trainer.test_freq=5",
        "trainer.total_training_steps=150",
        "trainer.max_actor_ckpt_to_keep=2",
        "trainer.max_critic_ckpt_to_keep=2",
    }
    assert expected.issubset(lines)
    assert (
        "actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu="
        f"{ROLLOUT_LOGPROB_MICRO_BATCHES[(size, environment)]}"
    ) in lines
    actor_micro_batch, ref_micro_batch = ACTOR_AND_REF_MICRO_BATCHES[
        (size, environment)
    ]
    assert (
        "actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu="
        f"{actor_micro_batch}"
    ) in lines
    assert (
        "actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu="
        f"{ref_micro_batch}"
    ) in lines
    assert lines[-1] == "trainer.experiment_name=cli_override"
    assert not any("sparse" in line or "random" in line for line in lines)

    if environment == "alfworld":
        assert "data.max_prompt_length=2048" in lines
        assert "env.max_steps=50" in lines
    else:
        assert "data.max_prompt_length=4096" in lines
        assert "env.max_steps=15" in lines
        assert "++env.validation_concurrency=128" in lines


@pytest.mark.parametrize("size", SIZES)
def test_alfworld_runtime_defaults(size: str):
    text = _launcher(size, "alfworld").read_text(encoding="utf-8")
    runtime_root = "/data/zhangdw12/work/verl-agent/.uv-venv/verl-agent"
    assert f"VERL_AGENT_RUNTIME_ROOT:-{runtime_root}" in text
    assert f'PYTHON_BIN="${{PYTHON_BIN:-{runtime_root}/bin/python3}}"' in text
    assert 'ALFWORLD_DATA="${ALFWORLD_DATA:-${HOME}/.cache/alfworld}"' in text
    assert (
        'GLIBC_SHIM="${GLIBC_SHIM:-${VERL_AGENT_RUNTIME_ROOT}/lib/'
        'libshim_glibc235.so}"'
        in text
    )
    assert 'if [[ -f "${GLIBC_SHIM}" ]]; then' in text


@pytest.mark.parametrize("size", SIZES)
def test_webshop_assumes_an_active_environment(size: str):
    text = _launcher(size, "webshop").read_text(encoding="utf-8")
    assert 'PYTHON_BIN="${PYTHON_BIN:-python3}"' in text
    assert "conda" not in text
    assert "mamba" not in text
    assert "WEBSHOP_RUNTIME_ROOT" not in text


def test_trainer_wiring_consumes_all_canonical_chunks():
    trainer = (
        REPO_ROOT / "verl" / "trainer" / "ppo" / "ray_trainer.py"
    ).read_text(encoding="utf-8")
    assert 'getattr(self.val_envs, "iter_chunks", None)' in trainer
    assert "validation_chunk.task_count" in trainer
    assert 'metric_dict[f"val/{prefix}completed_count"]' in trainer
