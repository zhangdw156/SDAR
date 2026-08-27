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

import importlib.util
import json
import sys
import types
from pathlib import Path

import numpy as np
import pytest

from agent_system.environments.fairness import (
    VALIDATION_CONCURRENCY,
    alfworld_fairness_split,
    alfworld_gamefiles,
    alfworld_worker_gamefiles,
    canonical_validation_chunks,
    canonical_validation_splits,
    load_fairness_rows,
    webshop_goal_fingerprint,
    webshop_goal_indices,
    webshop_reset_goal_indices,
)


def _write_jsonl(path, rows):
    path.write_text(
        "".join(f"{json.dumps(row)}\n" for row in rows),
        encoding="utf-8",
    )


def _alfworld_rows(count):
    return [
        {
            "index": index,
            "metadata": {
                "gamefile": f"${{ALFWORLD_DATA}}/game-{index}.tw-pddl",
                "split": "valid_seen",
            },
        }
        for index in range(count)
    ]


def _webshop_rows(indices):
    return [
        {
            "text": f"webshop goal {goal_idx}",
            "metadata": {
                "goal_idx": goal_idx,
                "split": "valid",
                "goal_seed": 0,
            },
        }
        for goal_idx in indices
    ]


def test_fairness_loaders_preserve_manifest_order(tmp_path, monkeypatch):
    alfworld_dir = tmp_path / "alfworld"
    webshop_dir = tmp_path / "webshop"
    alfworld_dir.mkdir()
    webshop_dir.mkdir()
    _write_jsonl(
        alfworld_dir / "evaluation_seen.jsonl",
        _alfworld_rows(140),
    )
    _write_jsonl(
        alfworld_dir / "evaluation_unseen.jsonl",
        _alfworld_rows(134),
    )
    webshop_indices = list(reversed(range(500)))
    _write_jsonl(
        webshop_dir / "evaluation.jsonl",
        _webshop_rows(webshop_indices),
    )
    monkeypatch.setenv("ALFWORLD_FAIRNESS_DIR", str(alfworld_dir))
    monkeypatch.setenv("WEBSHOP_FAIRNESS_DIR", str(webshop_dir))

    assert alfworld_gamefiles("evaluation_seen") == [
        f"${{ALFWORLD_DATA}}/game-{index}.tw-pddl"
        for index in range(140)
    ]
    assert webshop_goal_indices("evaluation") == webshop_indices


def test_fairness_loader_rejects_duplicate_task_identity(
    tmp_path,
    monkeypatch,
):
    alfworld_dir = tmp_path / "alfworld"
    alfworld_dir.mkdir()
    rows = _alfworld_rows(140)
    rows[-1]["metadata"]["gamefile"] = rows[0]["metadata"]["gamefile"]
    _write_jsonl(alfworld_dir / "evaluation_seen.jsonl", rows)
    monkeypatch.setenv("ALFWORLD_FAIRNESS_DIR", str(alfworld_dir))

    with pytest.raises(ValueError, match="Duplicate metadata.gamefile"):
        load_fairness_rows("alfworld", "evaluation_seen")


def test_alfworld_validation_assigns_one_fixed_game_per_chunk_worker():
    games = [f"game-{index}" for index in range(128)]

    assert alfworld_worker_gamefiles(
        fixed_assignment=True,
        canonical_gamefiles=games,
        num_processes=128,
    ) == games
    assert alfworld_worker_gamefiles(
        fixed_assignment=False,
        canonical_gamefiles=games,
        num_processes=16,
    ) == [None] * 16

    with pytest.raises(ValueError, match="chunk requires exactly 128 workers"):
        alfworld_worker_gamefiles(
            fixed_assignment=True,
            canonical_gamefiles=games,
            num_processes=127,
        )


def test_alfworld_fairness_split_keeps_seen_and_unseen_distinct():
    assert alfworld_fairness_split(
        is_train=True,
        eval_dataset="eval_in_distribution",
    ) == "train"
    assert alfworld_fairness_split(
        is_train=False,
        eval_dataset="eval_in_distribution",
        requested_split="evaluation_seen",
    ) == "evaluation_seen"
    assert alfworld_fairness_split(
        is_train=False,
        eval_dataset="eval_out_of_distribution",
    ) == "evaluation_unseen"


def test_webshop_evaluation_order_is_fixed_and_training_is_seeded():
    validation = list(range(500))
    assert webshop_reset_goal_indices(
        is_train=False,
        goal_indices=validation,
        env_num=500,
        group_n=1,
        rng=np.random.RandomState(7),
    ) == validation

    train_pool = list(range(500, 700))
    first = webshop_reset_goal_indices(
        is_train=True,
        goal_indices=train_pool,
        env_num=8,
        group_n=2,
        rng=np.random.RandomState(11),
    )
    second = webshop_reset_goal_indices(
        is_train=True,
        goal_indices=train_pool,
        env_num=8,
        group_n=2,
        rng=np.random.RandomState(11),
    )
    assert first == second
    assert all(first[index] == first[index + 1] for index in range(0, 16, 2))
    assert len(set(first)) == 8

    legacy_first = webshop_reset_goal_indices(
        is_train=False,
        fairness_enabled=False,
        goal_indices=validation,
        env_num=8,
        group_n=1,
        rng=np.random.RandomState(13),
    )
    legacy_second = webshop_reset_goal_indices(
        is_train=False,
        fairness_enabled=False,
        goal_indices=validation,
        env_num=8,
        group_n=1,
        rng=np.random.RandomState(13),
    )
    assert legacy_first == legacy_second
    assert len(legacy_first) == 8
    assert legacy_first != validation[:8]


def test_webshop_goal_fingerprint_covers_resolved_content():
    goals = [
        {
            "goal_idx": 7,
            "instruction_text": "find a blue shirt",
            "goal_options": {"color": "blue"},
        }
    ]
    assert webshop_goal_fingerprint(goals) == webshop_goal_fingerprint(
        [dict(goals[0])]
    )
    changed = [dict(goals[0], instruction_text="find a red shirt")]
    assert webshop_goal_fingerprint(goals) != webshop_goal_fingerprint(
        changed
    )


def test_canonical_validation_splits_and_chunks(
    tmp_path,
    monkeypatch,
):
    alfworld_dir = tmp_path / "alfworld"
    webshop_dir = tmp_path / "webshop"
    alfworld_dir.mkdir()
    webshop_dir.mkdir()
    _write_jsonl(
        alfworld_dir / "evaluation_seen.jsonl",
        _alfworld_rows(140),
    )
    _write_jsonl(
        alfworld_dir / "evaluation_unseen.jsonl",
        _alfworld_rows(134),
    )
    _write_jsonl(
        webshop_dir / "evaluation.jsonl",
        _webshop_rows(range(500)),
    )
    monkeypatch.setenv("ALFWORLD_FAIRNESS_DIR", str(alfworld_dir))
    monkeypatch.setenv("WEBSHOP_FAIRNESS_DIR", str(webshop_dir))

    assert canonical_validation_splits("alfworld") == (
        "evaluation_seen",
        "evaluation_unseen",
    )
    assert canonical_validation_splits("webshop") == ("evaluation",)
    assert [
        len(chunk)
        for chunk in canonical_validation_chunks(
            "alfworld",
            "evaluation_seen",
        )
    ] == [128, 12]
    assert [
        len(chunk)
        for chunk in canonical_validation_chunks(
            "alfworld",
            "evaluation_unseen",
        )
    ] == [128, 6]
    webshop_chunks = canonical_validation_chunks(
        "webshop",
        "evaluation",
    )
    assert [len(chunk) for chunk in webshop_chunks] == [128, 128, 128, 116]
    assert [value for chunk in webshop_chunks for value in chunk] == list(
        range(500)
    )

    for invalid in (0, -1, VALIDATION_CONCURRENCY + 1):
        with pytest.raises(ValueError, match="validation concurrency"):
            canonical_validation_chunks(
                "webshop",
                "evaluation",
                concurrency=invalid,
            )


def _load_webshop_text_env_with_stubs(monkeypatch):
    gym_module = types.ModuleType("gym")
    gym_module.Env = object
    monkeypatch.setitem(sys.modules, "gym", gym_module)

    bs4_module = types.ModuleType("bs4")
    bs4_module.BeautifulSoup = object
    bs4_element_module = types.ModuleType("bs4.element")
    bs4_element_module.Comment = object
    monkeypatch.setitem(sys.modules, "bs4", bs4_module)
    monkeypatch.setitem(sys.modules, "bs4.element", bs4_element_module)

    class FakeFlask:
        def __init__(self, _name):
            pass

        def route(self, *_args, **_kwargs):
            return lambda function: function

    flask_module = types.ModuleType("flask")
    flask_module.Flask = FakeFlask
    monkeypatch.setitem(sys.modules, "flask", flask_module)

    engine_module = types.ModuleType("web_agent_site.engine.engine")

    def load_products(**kwargs):
        marker = kwargs["rng"].random()
        return [{"marker": marker}], {}, {}, {}

    engine_module.load_products = load_products
    engine_module.init_search_engine = lambda **_kwargs: object()
    engine_module.get_top_n_product_from_keywords = lambda *_a, **_k: []
    engine_module.map_action_to_html = lambda *_a, **_k: ("", "")
    engine_module.parse_action = lambda action: (action, "")
    engine_module.get_product_per_page = lambda *_a, **_k: []
    engine_module.ACTION_TO_TEMPLATE = {}
    engine_module.END_BUTTON = "Buy Now"
    engine_module.NEXT_PAGE = "Next >"
    engine_module.PREV_PAGE = "< Prev"
    engine_module.BACK_TO_SEARCH = "Back to Search"

    goal_module = types.ModuleType("web_agent_site.engine.goal")
    goal_module.get_reward = lambda *_a, **_k: 0
    goal_module.get_goals = lambda products, *_a, **kwargs: [
        {
            "weight": 1,
            "marker": products[0]["marker"],
            "sample": kwargs["rng"].random(),
        }
        for _ in range(5)
    ]

    utils_module = types.ModuleType("web_agent_site.utils")
    utils_module.DEFAULT_FILE_PATH = "items.json"
    utils_module.DEFAULT_ATTR_PATH = "attrs.json"
    utils_module.FEAT_CONV = "features.pt"
    utils_module.FEAT_IDS = "ids.pt"
    utils_module.random_idx = lambda _weights: 0

    web_agent_site = types.ModuleType("web_agent_site")
    web_agent_site.__path__ = []
    engine_package = types.ModuleType("web_agent_site.engine")
    engine_package.__path__ = []
    monkeypatch.setitem(sys.modules, "web_agent_site", web_agent_site)
    monkeypatch.setitem(sys.modules, "web_agent_site.engine", engine_package)
    monkeypatch.setitem(
        sys.modules,
        "web_agent_site.engine.engine",
        engine_module,
    )
    monkeypatch.setitem(
        sys.modules,
        "web_agent_site.engine.goal",
        goal_module,
    )
    monkeypatch.setitem(sys.modules, "web_agent_site.utils", utils_module)

    module_path = (
        Path(__file__).parents[2]
        / "agent_system/environments/env_package/webshop/webshop/"
        "web_agent_site/envs/web_agent_text_env.py"
    )
    spec = importlib.util.spec_from_file_location(
        "_fairness_web_agent_text_env",
        module_path,
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_webshop_goal_construction_ignores_worker_seed(monkeypatch):
    module = _load_webshop_text_env_with_stubs(monkeypatch)

    first = module.SimServer(
        seed=11,
        goal_seed=0,
        base_url="http://example",
        file_path="items.json",
        attr_path="attrs.json",
    )
    second = module.SimServer(
        seed=99,
        goal_seed=0,
        base_url="http://example",
        file_path="items.json",
        attr_path="attrs.json",
    )

    assert first.goals == second.goals


@pytest.mark.parametrize(
    "relative_path",
    [
        "examples/sdar_trainer_1.5b/run_webshop.sh",
        "examples/sdar_trainer_3b/run_webshop.sh",
        "examples/sdar_trainer_7b/run_webshop.sh",
    ],
)
def test_webshop_launchers_use_one_validation_batch(relative_path):
    launcher = Path(__file__).parents[2] / relative_path
    text = launcher.read_text(encoding="utf-8")

    assert (
        "--val_data_size $val_data_size" in text
        or "--val_data_size 128" in text
        or (
            '"--val_data_size"' in text
            and '"128"' in text
        )
    )
    assert "val_data_size * 2" not in text
