# Copyright 2026 The verl-agent team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Fail-closed configuration contract for the native SDAR GRPO path."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

NATIVE_TRAJECTORY_GRPO_CONFIG = {
    "scheduler": "row",
    "reducer": "token_mean",
    "advantage": "step_row",
    "penalty": "step_local",
    "filter": "off",
}


def _trajectory_grpo_block(config: Mapping[str, Any]) -> Mapping[str, Any]:
    if not isinstance(config, Mapping):
        raise ValueError("trajectory_grpo config must be a mapping")
    if "algorithm" in config:
        algorithm = config["algorithm"]
        if not isinstance(algorithm, Mapping):
            raise ValueError("algorithm config must be a mapping")
        config = algorithm.get("trajectory_grpo", {})
    elif "trajectory_grpo" in config:
        config = config["trajectory_grpo"]

    if config is None:
        return {}
    if not isinstance(config, Mapping):
        raise ValueError("trajectory_grpo config must be a mapping")
    return config


def resolve_trajectory_grpo_config(config: Mapping[str, Any]) -> dict[str, str]:
    """Resolve defaults and reject every non-native trajectory-GRPO setting."""

    configured = dict(_trajectory_grpo_block(config))
    unknown_fields = sorted(set(configured) - set(NATIVE_TRAJECTORY_GRPO_CONFIG))
    if unknown_fields:
        raise ValueError(
            "unknown trajectory_grpo fields: " + ", ".join(unknown_fields)
        )

    resolved = {**NATIVE_TRAJECTORY_GRPO_CONFIG, **configured}
    for field, native_value in NATIVE_TRAJECTORY_GRPO_CONFIG.items():
        value = resolved[field]
        if value != native_value:
            raise ValueError(
                f"trajectory_grpo.{field}={value!r} is not supported; "
                f"expected native value {native_value!r}"
            )
    return resolved


__all__ = [
    "NATIVE_TRAJECTORY_GRPO_CONFIG",
    "resolve_trajectory_grpo_config",
]
