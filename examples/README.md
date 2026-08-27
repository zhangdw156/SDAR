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

## Runtime paths

ALFWorld launchers default to:

- `PYTHON_BIN=/data/zhangdw12/work/verl-agent/.uv-venv/verl-agent/bin/python3`
- `ALFWORLD_DATA=${HOME}/.cache/alfworld`
- `GLIBC_SHIM=/data/zhangdw12/work/verl-agent/.uv-venv/verl-agent/lib/libshim_glibc235.so`

The shim is added to `LD_PRELOAD` only when it exists. WebShop launchers default
to `PYTHON_BIN=python3`, do not activate Conda/Mamba, and assume the caller has
already activated a runnable environment. WebShop data and indexes remain at
the repository-relative paths used by the bundled environment package.

Canonical fairness manifests are downloaded on first use to
`${VERL_AGENT_FAIRNESS_CACHE:-${HOME}/.cache/verl-agent/fairness}`. Per-benchmark
overrides remain available through `ALFWORLD_FAIRNESS_DIR` and
`WEBSHOP_FAIRNESS_DIR`.

Set `LAUNCHER_DRY_RUN=true` to print the resolved trainer module and Hydra
overrides without preparing data or starting training. Additional CLI Hydra
overrides are appended last and therefore take precedence.
