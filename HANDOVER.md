# Handover

> **Session handover** — read the Active Topic first, then reference docs as needed.

---

## Active Topic: Grid Search

### Context

Market env redesign is **implemented and pushed** (2 commits on main). All design docs are up to date:
- `MARKET_ENV.md` — fill mechanics (aggressive + passive)
- `RL_DESIGN.md` — reward function, observation space, action space, λ calibration

### What was just done

- Implemented new `market_env.py` with aggressive/passive fills, 4D obs, width/skew actions, time-varying penalty
- Updated `policy.py`, `train_rl.py`, `simulate.py`, `run_experiment.py`, experiment configs
- Fixed context_len mismatch bug in `train_rl.py` (was hardcoded, now reads from checkpoint)
- Added auto-loading of R² from transformer metrics when not set in config
- Set `half_spread` default to 5bps (0.0005)
- Smoke-tested full pipeline with `experiments/smoke.json`
- Created `grid_search.py` — grid search over `n_sigma` and `lambda2` using existing experiment data + transformer

### Grid search script (`grid_search.py`)

Reuses an existing experiment's data and trained transformer. Creates `output/<experiment>_grid/ns{n_sigma}_l2{lambda2}/` for each combination. Symlinks data/checkpoints, runs RL + simulation only.

Grid values:
- `n_sigma`: 0.5, 1.0, 2.0, 3.0
- `lambda2`: 1.0, 3.0, 5.0

**Status: script written, not yet tested on experiment2.** The user interrupted before the test run. Next step is to run it:

```bash
python grid_search.py output/experiment2 2>&1 | tee log.txt
```

`output/experiment2` exists with trained transformer (R²=0.558, SNR=1.0, 95 stocks, 2000 steps).

### Known issue

`simulate.py` generates a fresh stock for visualization. With `n_stocks=1` and certain DGP params, `generate_data.py` can produce NaN (divide-by-zero in vol scaling). This is a pre-existing bug in single-stock generation, not related to the env redesign. Only affects simulation, not training.

---

## Design Docs

| Doc | Contents |
|---|---|
| `MARKET_ENV.md` | Fill mechanics: aggressive/passive fills, examples, multiplicity rules |
| `RL_DESIGN.md` | Observation space (4D), reward function (time-varying penalty + terminal), action space (width/skew), λ calibration from R² |
| `FUTURE_DEVELOPMENTS.md` | Multi-step predictions, Kelly criterion penalty |
| `DATA_MODEL.md` | DGP specification, parameter interpretation |

---

## Key Parameters

| Parameter | Value | Source |
|---|---|---|
| `half_spread` | 0.0005 (5bps) | Fixed, not searched |
| `R²` | Auto-loaded from transformer metrics | Measured |
| `target_vol` | From data config (typically 0.02) | Known DGP param |
| `n_sigma` | Grid searched: 0.5, 1, 2, 3 | Risk tolerance tuning knob |
| `lambda2` | Grid searched: 1, 3, 5 | EOD liquidation impact |
| `tau` | 20 | Penalty floor, not sensitive |

---

## Quick Reference

```bash
# Full pipeline
python run_experiment.py experiments/example.json

# Grid search (RL only, reuses existing data + transformer)
python grid_search.py output/experiment2 2>&1 | tee log.txt

# Smoke test (fast, tiny config)
python run_experiment.py experiments/smoke.json
```
