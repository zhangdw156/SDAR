import os
import shutil
import subprocess
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).parents[2]
SIZES = ("1.5b", "3b", "7b")
ENVIRONMENTS = ("alfworld", "webshop")
WEBSHOP_RESOURCES = (
    Path("data/items_shuffle_1000.json"),
    Path("data/items_ins_v2_1000.json"),
    Path("data/items_human_ins.json"),
    Path("search_engine/indexes"),
)


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


def _prepare_webshop_resources(shared_root: Path) -> None:
    for resource in WEBSHOP_RESOURCES:
        source = shared_root / resource
        if resource == Path("search_engine/indexes"):
            source.mkdir(parents=True)
        else:
            source.parent.mkdir(parents=True, exist_ok=True)
            source.write_text(resource.name, encoding="utf-8")


def _run_webshop_launcher(
    size: str,
    repo_root: Path,
    shared_root: Path,
    *,
    dry_run: bool = False,
) -> subprocess.CompletedProcess[str]:
    python_bin = shutil.which("true")
    if python_bin is None:
        raise RuntimeError("The launcher test requires the standard 'true' command")
    environment = os.environ.copy()
    environment.update(
        {
            "PYTHON_BIN": python_bin,
            "REPO_ROOT": str(repo_root),
            "WEBSHOP_SHARED_ROOT": str(shared_root),
        }
    )
    if dry_run:
        environment["LAUNCHER_DRY_RUN"] = "true"
    return subprocess.run(
        [str(_launcher(size, "webshop"))],
        cwd=REPO_ROOT,
        env=environment,
        check=False,
        capture_output=True,
        text=True,
    )


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
    assert (
        "WEBSHOP_SHARED_ROOT:-/data/zhangdw12/work/verl-agent/"
        "agent_system/environments/env_package/webshop/webshop"
    ) in text
    assert (
        'WEBSHOP_LOCAL_ROOT="${WEBSHOP_LOCAL_ROOT:-${REPO_ROOT}/'
        'agent_system/environments/env_package/webshop/webshop}"'
    ) in text


@pytest.mark.parametrize("size", SIZES)
def test_webshop_runtime_links_missing_resources(size: str, tmp_path: Path):
    repo_root = tmp_path / "clone"
    shared_root = tmp_path / "shared"
    repo_root.mkdir()
    _prepare_webshop_resources(shared_root)

    completed = _run_webshop_launcher(size, repo_root, shared_root)

    assert completed.returncode == 0, completed.stderr
    local_root = (
        repo_root
        / "agent_system/environments/env_package/webshop/webshop"
    )
    for resource in WEBSHOP_RESOURCES:
        target = local_root / resource
        assert target.is_symlink()
        assert target.resolve() == (shared_root / resource).resolve()


@pytest.mark.parametrize("size", SIZES)
def test_webshop_runtime_linking_is_idempotent(size: str, tmp_path: Path):
    repo_root = tmp_path / "clone"
    shared_root = tmp_path / "shared"
    repo_root.mkdir()
    _prepare_webshop_resources(shared_root)

    first = _run_webshop_launcher(size, repo_root, shared_root)
    assert first.returncode == 0, first.stderr
    local_root = (
        repo_root
        / "agent_system/environments/env_package/webshop/webshop"
    )
    first_inodes = {
        resource: (local_root / resource).lstat().st_ino
        for resource in WEBSHOP_RESOURCES
    }
    second = _run_webshop_launcher(size, repo_root, shared_root)

    assert second.returncode == 0, second.stderr
    assert first_inodes == {
        resource: (local_root / resource).lstat().st_ino
        for resource in WEBSHOP_RESOURCES
    }


@pytest.mark.parametrize("size", SIZES)
def test_webshop_runtime_fails_before_linking_when_shared_source_is_missing(
    size: str,
    tmp_path: Path,
):
    repo_root = tmp_path / "clone"
    shared_root = tmp_path / "shared"
    repo_root.mkdir()
    _prepare_webshop_resources(shared_root)
    missing_resource = WEBSHOP_RESOURCES[-1]
    (shared_root / missing_resource).rmdir()

    completed = _run_webshop_launcher(size, repo_root, shared_root)

    assert completed.returncode != 0
    assert str(shared_root / missing_resource) in completed.stderr
    assert "Run the WebShop setup first" in completed.stderr
    local_root = (
        repo_root
        / "agent_system/environments/env_package/webshop/webshop"
    )
    assert not any((local_root / resource).exists() for resource in WEBSHOP_RESOURCES)


@pytest.mark.parametrize("size", SIZES)
def test_webshop_dry_run_does_not_create_runtime_links(size: str, tmp_path: Path):
    repo_root = tmp_path / "clone"
    shared_root = tmp_path / "missing-shared"
    repo_root.mkdir()

    completed = _run_webshop_launcher(
        size,
        repo_root,
        shared_root,
        dry_run=True,
    )

    assert completed.returncode == 0, completed.stderr
    assert not (
        repo_root
        / "agent_system/environments/env_package/webshop/webshop"
    ).exists()


def test_trainer_wiring_consumes_all_canonical_chunks():
    trainer = (
        REPO_ROOT / "verl" / "trainer" / "ppo" / "ray_trainer.py"
    ).read_text(encoding="utf-8")
    assert 'getattr(self.val_envs, "iter_chunks", None)' in trainer
    assert "validation_chunk.task_count" in trainer
    assert 'metric_dict[f"val/{prefix}completed_count"]' in trainer
