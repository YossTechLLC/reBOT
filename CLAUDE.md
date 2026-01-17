# claude operating guide for /timesfm (submodule)

purpose: help the user build, run, troubleshoot, and extend **timesfm** with professional-grade engineering (correctness -> security -> performance).

## 1) scope and safety (non-negotiable)

- hard scope: read/write only inside the `/timesfm/` directory tree (this repo is a git submodule).
- no secrets: never print, log, paste, or commit secrets/keys/tokens/credentials. redact if you must reference them.
- no surprise destruction: do not run destructive commands (rm -rf, git reset --hard, dropping tables, deleting checkpoints) unless explicitly requested.
- be deterministic: prefer reproducible steps, pinned versions when asked, and minimal diffs.

## 2) repo map (verify locally, then work from facts)

treat the local filesystem as the source of truth. first run:

- `pwd && git rev-parse --show-toplevel && git status`
- `ls -la`
- `find . -maxdepth 2 -type f -name 'pyproject.toml' -o -name 'README*' -o -name '*.md'`

expected high-level layout in this repo:

- `pyproject.toml`: python packaging, tooling config, optional extras.
- `src/timesfm/`: primary python package (model loading, inference helpers).
- `notebooks/`: runnable examples.
- `peft/` (or similarly named): finetune/peft scripts and examples.
- `v1/`: legacy code paths and/or older training utilities.

when something conflicts with this expectation, update this file with the verified structure.

## 3) environment and installs (preferred workflow)

target python: 3.10 or 3.11 (avoid 3.12+ unless the repo explicitly supports it).

inside `/timesfm/`:

1) create and activate a venv
- `python -m venv .venv`
- `source .venv/bin/activate`
- `python -m pip install -U pip setuptools wheel`

2) install for development
- preferred: `pip install -e .`
- if extras exist in `pyproject.toml` (for example torch/jax/dev), install them as: `pip install -e '.[EXTRA]'`

3) sanity checks
- `python -c "import timesfm; print(timesfm.__file__)"`
- run any repo-defined checks (see `pyproject.toml` / `README.md`).

notes:
- the official pip path for inference often looks like `pip install timesfm[torch]` when you are not developing from source.
- memory use can be high; do not assume small-model behavior when loading checkpoints.

## 4) what the notebooks teach us (key api surfaces)

the provided notebook demonstrates:

- constructing a model with `TimesFmHparams` and a `TimesFmCheckpoint` from hugging face.
- core inference: `model.forecast(inputs=..., freq=...)`.
- covariate/xreg inference: `model.forecast_with_covariates(...)` using:
  - dynamic numerical covariates (future-known regressors)
  - dynamic categorical covariates
  - static categorical covariates
  - `xreg_mode` (for example: `xreg + timesfm`)
  - `ridge`, `force_on_cpu`, and `normalize_xreg_target_per_input`

when the user asks about covariates, prioritize:
- validating shapes (context vs horizon)
- preventing leakage (no future-only info in context covariates)
- clear error messages for missing/unknown covariates

## 5) engineering standards

logging:
- use python `logging` (or the repo standard). include: component, severity, short error code, and actionable hint.
- never log raw secrets or entire datasets.

a minimal error taxonomy (use consistently):
- `TFM1001 CONFIG` bad config/env/flags
- `TFM2001 DATA` bad shapes, missing columns, leakage risks
- `TFM3001 CHECKPOINT` missing or incompatible checkpoint
- `TFM4001 INFERENCE` runtime/oom/nan/precision issues
- `TFM5001 PERF` regression or unexpected slowness

tests and validation:
- if tests exist, run them.
- always add at least a small smoke test for any new public function:
  - import
  - basic forecast on a tiny synthetic series
  - covariate path (if modified)

style:
- follow the repo tools in `pyproject.toml` (ruff/black/pytest/etc). do not introduce a second formatter.

## 6) project journals (brevity is the soul of wit)

create/maintain these files in the repo root:

### BUGS.md
one bullet per bug:
- `[open|fixed] yyyy-mm-dd - short title - repro (1 line) - root cause (1 line) - fix (file/function)`

### PROGRESS.md
3 sections only:
- `now:` current focus (1-3 bullets)
- `next:` next steps (1-5 bullets)
- `done:` last 5 completed items (rolling)

### DECISIONS.md
one entry per decision:
- `yyyy-mm-dd - decision - context - options - chosen + why (max 5 lines)`

## 7) default workflow per user request

1) restate the request as an acceptance test (what will be true when done).
2) locate the smallest set of files to change.
3) implement minimal diff.
4) add/adjust tests or a tiny repro script.
5) run checks.
6) update BUGS/PROGRESS/DECISIONS if relevant.
7) provide the user exact commands to run and where outputs land.

## 8) shell hygiene (avoid common cli failures)

- quote complex args: `--filter="a AND b"`.
- prefer single-line commands over multiline pasted blocks.
- when passing regex/special chars, wrap in single quotes where possible.

EOF
