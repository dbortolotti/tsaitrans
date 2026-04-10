# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

A learning experiment (not production) with two modules:

1. **Prediction** — Encoder-only transformer for univariate synthetic time series forecasting. Each stock is an independent realization; the model processes one stock at a time.
2. **Placing** — RL market-making agent (PPO) that uses transformer predictions as features to learn bid/ask placement.

PyTorch only — no HuggingFace, no Lightning. Targets Apple Silicon (M3/M4) via MPS.

## Running Experiments

An experiment is defined by a JSON config file (see `experiments/example.json`). Only parameters you want to vary need to be specified — everything else uses defaults from `run_experiment.py:DEFAULTS`.

```bash
# Install
pip install torch numpy matplotlib gymnasium

# Run full pipeline (generate data → train transformer → inference → train RL)
python run_experiment.py experiments/example.json

# All outputs go to output/<experiment_name>/
```

Individual scripts can still be run standalone — see their `--help` for args.

### Experiment JSON structure

```json
{
  "data": {
    "target_vol": 0.02,
    "snr": 0.3,
    "factor_half_life": 0.1,
    "noise_half_life_range": [0.005, 0.025],
    "n_factors": 3,
    "steps_per_day": 2000,
    "n_steps": 2000,
    "stocks_transformer_train": 20,
    "stocks_transformer_val": 5,
    "stocks_rl_train": 10,
    "stocks_rl_val": 3,
    "stocks_test": 5
  },
  "transformer": {
    "d_model": 64, "n_heads": 4, "n_layers": 3,
    "lr": 3e-4, "n_epochs": 50
  },
  "rl": {
    "predictor": "transformer",
    "half_spread": 0.001, "n_sigma": 1.0, "lambda2": 1.5,
    "n_iterations": 200
  }
}
```

See `DATA_MODEL.md` for the full model spec, parameter interpretation, and calibration discussion.

## Architecture

### Prediction (`prediction/`)

**Pipeline: generate_data.py → train.py → inference.py**, with `model.py` as shared module.

- **`model.py`** — `TimeSeriesDataset` (per-stock sliding windows, univariate), `FactorTransformer` (linear projection → sinusoidal PE → TransformerEncoder → linear head, n_stocks=1), `get_device()`, `make_splits()` (stock-based train/val/test with train-only normalization)
- **`generate_data.py`** — Latent factor model: observations = loadings × VAR(1) factors + AR(1) noise. Signal and noise are rescaled to hit `target_vol` and `snr`. Loadings are ground truth only, never model input.
- **`train.py`** — AdamW + cosine LR with warmup + early stopping + grad clipping.
- **`inference.py`** — Loads checkpoint + normalization stats, computes metrics on test stocks, saves plots.

### Placing (`placing/`)

**Pipeline: train_rl.py (loads frozen transformer + data → trains PPO agent)**

- **`market_env.py`** — Gymnasium env (`MarketMakingEnv`) + `VectorizedMarketEnv`. Agent quotes (width, skew) in half_spread multiples. Two fill types: aggressive (crosses market, pays spread) and passive (resting quote crossed by market, earns spread). 4D observation: predicted move, position risk, time remaining, cumulative PnL. See `MARKET_ENV.md` for fill mechanics, `RL_DESIGN.md` for reward/obs/action design.
- **`policy.py`** — `ActorCritic` shared-trunk MLP (~10k params) with PPO update logic and GAE-Lambda advantage estimation via `RolloutBuffer`.
- **`train_rl.py`** — Loads data, generates transformer predictions (or falls back to momentum), runs PPO over vectorized envs. Samples across RL train stocks for env creation.
- **`generate_demo_data.py`** — Creates fake `sim_results.json` for testing the visualizer without training.

### Key coupling

The transformer is **frozen** during RL training — its predictions are just input features to the policy. `train_rl.py` imports from `prediction/model.py` via sys.path.

## Key Design Decisions

- **Stocks as realizations** — each stock is an i.i.d. sample from the same DGP. Splits are across stocks, not time.
- **Stock split** — 5 disjoint groups: transformer train, transformer val, RL train, RL val, test. RL trains on stocks that are out-of-sample for the transformer, so the policy learns from realistic prediction quality.
- **Univariate model** — the transformer processes one stock at a time (n_stocks=1). All stocks in a split are pooled into one training dataset.
- **Returns not prices** — avoids non-stationarity
- **Normalization** — scalar mean/std computed from transformer train stocks only; saved in checkpoint dir, reloaded at inference
- **MPS device** — `num_workers=0` in DataLoaders (MPS + multiprocessing breaks on macOS)
- **R² vs naive** — baseline is predicting zero. R² near 0 is expected; R² < 0 is a bug signal
- **Reward function** — Time-varying quadratic inventory penalty calibrated from R² and remaining time. Terminal EOD liquidation cost. See `RL_DESIGN.md` for derivation.
- **RL tuning levers** — `n_sigma` (risk tolerance), `half_spread` (market spread), `lambda2` (EOD impact), `R²` (measured from transformer)

## Output Directories

All outputs go to `output/<experiment_name>/` (gitignored):

- `data/` — generated returns, ground truth, metadata
- `checkpoints/` — transformer checkpoint (model, config, normalization stats, training log)
- `results/` — transformer inference metrics and prediction plots
- `checkpoints_rl/` — RL policy checkpoint (best/final policy, config, training log)
- `<name>.json` — copy of the experiment config
- `resolved_config.json` — config with all defaults filled in

## Style

- Code readable and commented for learning, not production-clever
- Direct tone — challenge assumptions, flag uncertainty, no filler
