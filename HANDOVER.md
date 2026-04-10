# Handover

> **Session handover** — read the Active Topic first, then reference docs as needed.

---

## Active Topic: RL Training Fixes + Grid Search

### Context

First grid search (12 runs on `experiment2`) showed PPO completely failed to train. Diagnosis:

1. **Episode length was wrong** — `seq_len = min(500, T//2)` truncated episodes to 500 steps, but the penalty, terminal cost, and time-remaining obs are calibrated for a full trading day (T=2000). The agent learned wrong flattening behavior at artificial boundaries.

2. **Reward scale broke PPO** — the quadratic inventory penalty produced per-step rewards at O(1-100) while PnL was O(0.01). The critic couldn't fit value targets of O(10^4-10^7), so advantages were meaningless, policy loss stayed ~0, entropy drifted upward, and the agent stayed random. Random agents accumulate positions via random-walk fills, making the penalty worse.

### What was just done

- **Fixed episode length** — removed `seq_len = min(500, T//2)`, episodes now use full trading day. `make_env_data()` gives each env the complete stock series.
- **Added `RunningMeanStd`** to `policy.py` — Welford's online algorithm for running mean/variance.
- **Reward normalization** in `train_rl.py` — rewards divided by running std (no mean subtraction). Critic now sees O(1) targets.
- **Observation normalization** in `train_rl.py` — obs centered and scaled by running mean/std. All 4 features now O(1) for the tanh trunk.
- **Normalizer persistence** — saved as `best_normalizer.npz` / `final_normalizer.npz` alongside policy checkpoints. Loaded in `simulate.py` for inference (backward-compatible: falls back to identity if missing).
- **Updated `grid_search.py`** — `-i`/`-o` flags for input experiment and output name prefix. Runs 10 sims per grid point with seed-stamped files (`sim_results_{seed}.json`). Saves `sim_summary.json` with aggregated stats.
- **Sim output path** — `simulate.py` now writes to `checkpoints_rl/sim_results_{seed}.json` (was overwriting a single `sim_results.json`).
- Also committed prior uncommitted fixes: HANDOVER rewrite, `generate_data.py` single-stock vol scaling, `model.py` nested tensor, `half_spread` default 5bps, `market_env.py` EOD cost tracking in cumulative PnL.

### Next step

**Run the grid search** with the fixes and evaluate results:

```bash
python grid_search.py -i output/experiment2 -o grid3 2>&1 | tee log_grid3.txt
```

`output/experiment2` has trained transformer (R²=0.558, SNR=1.0, 95 stocks, 2000 steps).

### What to look for

- Value loss should drop to O(1-10) instead of O(10^4-10^7)
- Policy loss should be non-zero and move
- Entropy should decrease (policy becoming less random)
- `reward_std` in logs should adapt over iterations
- Avg |position| in sims should be much smaller (was 13-159 in broken runs)
- PnL should show differentiation across `n_sigma` × `lambda2` grid

### Known issue (pre-existing)

`simulate.py` generates a fresh stock via `generate_data.py` with `n_stocks=1`. With certain DGP params, this can produce NaN (divide-by-zero in vol scaling). Only affects simulation, not training. The `generate_data.py` fix handles the single-stock case by skipping the empirical daily vol correction.

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
python grid_search.py -i output/experiment2 -o grid3 2>&1 | tee log_grid3.txt

# Smoke test (fast, tiny config)
python run_experiment.py experiments/smoke.json
```
