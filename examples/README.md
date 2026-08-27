# ICLR SDAR fairness launchers

This branch intentionally exposes only the six paper experiment launchers:

- `sdar_trainer_1.5b/run_{alfworld,webshop}.sh`
- `sdar_trainer_3b/run_{alfworld,webshop}.sh`
- `sdar_trainer_7b/run_{alfworld,webshop}.sh`

All launchers use the official SDAR method (`sdar_coef=0.01`, `gate_beta=5.0`,
benchmark skillbank, `skill_all=false`) through `verl.trainer.main_sdar`. The
official implementation intrinsically distills every generated turn. No
sparse or random turn-selection variant is provided.

The common model, optimizer, rollout, reference, batch/group, seed, GPU,
checkpoint, logging, horizon/history, and validation settings match the
corresponding `verl-agent` GRPO 1.5B/3B/7B fairness launchers. Fairness mode is
on by default. Training-time validation exhaustively evaluates ALFWorld's 140
seen plus 134 unseen tasks and WebShop's 500 evaluation goals in chunks of at
most 128 environments.

## Rollout performance behavior

The ALFWorld and WebShop launchers automatically compact active trajectories:
after an environment slot terminates, later generation and environment steps
exclude that slot while preserving its original task, observation history,
trajectory ID, and SDAR `turn_index`/step-row identity. The legacy full-batch
path remains available for environment managers without `step_selected`.

For synchronous FSDP-vLLM rollout workers, one bounded multi-turn collection
attempt also keeps the rollout weights and KV cache resident across turns.
Unsupported rollout backends retain their per-generation enter/exit behavior;
trajectory compaction remains enabled independently. Session cleanup is
failure-safe, and actor updates reject entering, active, or tainted rollout
sessions rather than training against uncertain residency state.

## Runtime paths

ALFWorld launchers default to:

- `PYTHON_BIN=/data/zhangdw12/work/verl-agent/.uv-venv/verl-agent/bin/python3`
- `ALFWORLD_DATA=${HOME}/.cache/alfworld`
- `GLIBC_SHIM=/data/zhangdw12/work/verl-agent/.uv-venv/verl-agent/lib/libshim_glibc235.so`

The shim is added to `LD_PRELOAD` only when it exists. WebShop launchers default
to `PYTHON_BIN=python3`, do not activate Conda/Mamba, and assume the caller has
already activated a runnable environment. On a clean H20 clone, each WebShop
launcher reuses the prepared resources under
`WEBSHOP_SHARED_ROOT=/data/zhangdw12/work/verl-agent/agent_system/environments/env_package/webshop/webshop`
by creating missing repository-local symbolic links for the three JSON data
files and `search_engine/indexes`. Existing local resources are left unchanged.
Override `WEBSHOP_SHARED_ROOT` or `WEBSHOP_LOCAL_ROOT` when either location
differs. Bootstrap is serialized across concurrent launchers, rolls back links
created by a failed attempt, and verifies non-empty data files plus a usable
Lucene index before data preparation or training.

Canonical fairness manifests are downloaded on first use to
`${VERL_AGENT_FAIRNESS_CACHE:-${HOME}/.cache/verl-agent/fairness}`. Per-benchmark
overrides remain available through `ALFWORLD_FAIRNESS_DIR` and
`WEBSHOP_FAIRNESS_DIR`.

Set `LAUNCHER_DRY_RUN=true` to print the resolved trainer module and Hydra
overrides without creating WebShop links, preparing data, or starting training.
Additional CLI Hydra overrides are appended last and therefore take precedence.
