import importlib.util
import sys
from pathlib import Path

import pytest

MODULE_PATH = Path(__file__).parents[3] / "verl/trainer/ppo/trajectory_grpo.py"
MODULE_SPEC = importlib.util.spec_from_file_location("trajectory_grpo", MODULE_PATH)
trajectory_grpo = importlib.util.module_from_spec(MODULE_SPEC)
sys.modules[MODULE_SPEC.name] = trajectory_grpo
MODULE_SPEC.loader.exec_module(trajectory_grpo)

from trajectory_grpo import (  # noqa: E402
    NATIVE_TRAJECTORY_GRPO_CONFIG,
    resolve_trajectory_grpo_config,
)


def test_native_trajectory_grpo_config_resolves_exactly():
    assert resolve_trajectory_grpo_config({}) == NATIVE_TRAJECTORY_GRPO_CONFIG
    assert (
        resolve_trajectory_grpo_config(
            {"algorithm": {"trajectory_grpo": NATIVE_TRAJECTORY_GRPO_CONFIG}}
        )
        == NATIVE_TRAJECTORY_GRPO_CONFIG
    )


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("scheduler", "trajectory"),
        ("reducer", "trajectory_mean"),
        ("advantage", "trajectory"),
        ("penalty", "trajectory"),
        ("filter", "penalty_aware"),
    ],
)
def test_non_native_trajectory_grpo_values_are_rejected(field, value):
    with pytest.raises(
        ValueError,
        match=rf"trajectory_grpo\.{field}.*{value!r}.*{NATIVE_TRAJECTORY_GRPO_CONFIG[field]!r}",
    ):
        resolve_trajectory_grpo_config(
            {"algorithm": {"trajectory_grpo": {field: value}}}
        )


def test_unknown_trajectory_grpo_fields_are_rejected():
    with pytest.raises(ValueError, match="unknown trajectory_grpo fields.*extra"):
        resolve_trajectory_grpo_config({"extra": True})
