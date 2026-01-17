# timesfm codebase map

**what**: decoder-only foundation model for time-series forecasting (google research, icml 2024)
**current**: v2.5 (200m params, 16k context, quantile forecasting, no freq indicator)
**legacy**: v1.0/2.0 archived in `v1/` (use `pip install timesfm==1.3.0`)

---

## directory tree

```
timesfm/
├── src/timesfm/              # main package (v2.5)
│   ├── __init__.py           # exports: ForecastConfig, TimesFM_2p5_200M_{torch,flax}
│   ├── configs.py            # ForecastConfig, TransformerConfig (frozen dataclasses)
│   ├── torch/                # pytorch backend (transformer, dense, normalization, util)
│   ├── flax/                 # jax/flax backend (parallel impl, same api)
│   ├── timesfm_2p5/          # 200m model implementations
│   │   ├── timesfm_2p5_base.py       # abstract base (load_checkpoint, forecast, forecast_with_covariates)
│   │   ├── timesfm_2p5_torch.py      # pytorch concrete (HF ModelHubMixin, JIT compile)
│   │   └── timesfm_2p5_flax.py       # flax concrete (quantile flip invariance, continuous head)
│   └── utils/
│       └── xreg_lib.py       # covariates/external regressors (BatchedInContextXRegLinear)
│
├── v1/                       # legacy (1.0: 200m/512ctx, 2.0: 500m/2048ctx)
│   ├── src/timesfm/          # runtime backend selection (JAX/PyTorch)
│   ├── src/adapter/          # PEFT layers (LoRA, DoRA)
│   ├── src/finetuning/       # torch finetuning pipeline
│   ├── peft/                 # full finetuning framework
│   ├── notebooks/            # finetuning.ipynb, covariates.ipynb
│   └── experiments/          # baselines, benchmarks
│
├── .github/workflows/
│   ├── main.yml              # build validation (ubuntu, py3.11, uv)
│   └── manual_publish.yml    # pypi release (manual dispatch)
│
├── pyproject.toml            # v2.0.0, py>=3.11, extras: [torch|flax|xreg]
├── requirements.txt          # pinned deps (uv-compiled)
├── README.md                 # install, quickstart, huggingface collection
└── CLAUDE.md                 # this submodule's operating guide
```

---

## core api (v2.5)

### basic forecast
```python
import timesfm
model = timesfm.TimesFM_2p5_200M_torch.from_pretrained("google/timesfm-2.5-200m-pytorch")
model.compile(timesfm.ForecastConfig(max_context=1024, max_horizon=256, normalize_inputs=True))
point, quantile = model.forecast(horizon=12, inputs=[series1, series2])  # (N, H), (N, H, 10)
```

### with covariates (xreg)
```python
point, quantile = model.forecast_with_covariates(
    inputs=time_series_list,
    dynamic_numerical_covariates={"feature": covariate_data},  # shape validation critical
    xreg_mode="xreg + timesfm",  # or "timesfm + xreg"
    ridge=0.01,
    force_on_cpu=False,
    normalize_xreg_target_per_input=True
)
```

**key**: src/timesfm/timesfm_2p5/timesfm_2p5_base.py:422 (abstract base)
**impl**: src/timesfm/timesfm_2p5/timesfm_2p5_{torch,flax}.py
**xreg**: src/timesfm/utils/xreg_lib.py:520 (BatchedInContextXRegLinear)

---

## install & setup

```bash
# clone & env
git clone https://github.com/google-research/timesfm.git && cd timesfm
python -m venv .venv && source .venv/bin/activate
python -m pip install -U pip setuptools wheel

# install (pick backend)
pip install -e .              # base (safetensors, huggingface_hub)
pip install -e '.[torch]'     # pytorch inference
pip install -e '.[flax]'      # jax/flax inference (faster)
pip install -e '.[xreg]'      # covariates (requires jax)

# sanity check
python -c "import timesfm; print(timesfm.__file__)"
```

**python**: >=3.11 strict (pyproject.toml:14)
**deps**: numpy (2.2.6), huggingface_hub (1.0.1), safetensors (0.6.2)

---

## v1 vs v2

| aspect          | v1 (legacy)                     | v2 (current)                       |
|-----------------|---------------------------------|------------------------------------|
| version         | 1.3.0 (poetry)                  | 2.0.0 (pep 518)                    |
| models          | 1.0 (200m/512ctx), 2.0 (500m/2k)| 2.5 (200m/16k ctx)                 |
| python          | 3.10-3.11                       | 3.11+ only                         |
| backend select  | runtime (try/except)            | explicit import (torch/flax)       |
| api             | TimesFm (auto-backend)          | TimesFM_2p5_200M_{torch,flax}      |
| freq indicator  | required                        | removed                            |
| quantile head   | n/a                             | continuous (30m params, 1k horizon)|
| peft            | LoRA/DoRA in v1/src/adapter/    | tbd (not in main src)              |
| notebooks       | v1/notebooks/ (finetuning, cov) | none yet (under construction)      |

**migration**: v1 code in `v1/` for reference; `pip install timesfm==1.3.0` for old checkpoints

---

## key files (line counts)

```
src/timesfm/timesfm_2p5/timesfm_2p5_flax.py       602L  # flax impl (quantile fixing, flip invariance)
src/timesfm/utils/xreg_lib.py                     520L  # covariate handling
src/timesfm/timesfm_2p5/timesfm_2p5_torch.py      472L  # pytorch impl (HF mixin, JIT)
src/timesfm/timesfm_2p5/timesfm_2p5_base.py       422L  # abstract base (load, forecast)
src/timesfm/torch/transformer.py                  370L  # pytorch transformer layers
src/timesfm/flax/transformer.py                   356L  # flax transformer layers
src/timesfm/configs.py                            105L  # config dataclasses
```

---

## ci/cd

- **build**: .github/workflows/main.yml (push/pr → ubuntu, py3.11, uv, build only)
- **release**: .github/workflows/manual_publish.yml (manual dispatch → pypi via twine)
- **tests**: none automated (manual via pytest in v1/tests/)
- **tooling**: ruff (88 chars, 2-space indent), setuptools build backend

---

## development workflow

1. **read first**: `cat README.md pyproject.toml src/timesfm/__init__.py`
2. **env**: `python -m venv .venv && source .venv/bin/activate && pip install -e '.[torch,flax,xreg]'`
3. **edit**: minimal diff, follow ruff style (line-length=88, indent-width=2)
4. **test**: `python -c "from timesfm import TimesFM_2p5_200M_torch; print('ok')"` or pytest
5. **build**: `python -m build` (creates dist/)
6. **docs**: update BUGS.md, PROGRESS.md, DECISIONS.md per CLAUDE.md

---

## error taxonomy (from CLAUDE.md)

```
TFM1001 CONFIG      # bad config/env/flags
TFM2001 DATA        # bad shapes, missing columns, leakage risks
TFM3001 CHECKPOINT  # missing or incompatible checkpoint
TFM4001 INFERENCE   # runtime/oom/nan/precision issues
TFM5001 PERF        # regression or unexpected slowness
```

---

## quick refs

- **paper**: https://arxiv.org/abs/2310.10688 (icml 2024)
- **checkpoints**: https://huggingface.co/collections/google/timesfm-release-66e4be5fdb56e960c1e482a6
- **blog**: https://research.google/blog/a-decoder-only-foundation-model-for-time-series-forecasting/
- **bigquery**: https://cloud.google.com/bigquery/docs/timesfm-model

---

**last updated**: 2026-01-16 (via claude code scan)
**operating guide**: CLAUDE.md
