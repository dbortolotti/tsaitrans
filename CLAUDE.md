# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

A learning experiment exploring how transformer models predict structured synthetic time series data. The goal is to understand the relationship between model complexity (d_model, n_layers), noise level (SNR), and prediction quality. PyTorch only — no HuggingFace, no Lightning.

## Commands

```bash
# Install dependencies
pip install torch numpy matplotlib

# Generate synthetic data (controls SNR via --sigma_eps; lower = higher SNR)
python generate_data.py --sigma_eps 0.5 --tag high_snr
python generate_data.py --sigma_eps 2.0 --tag low_snr

# Train
python train.py --data data/returns_high_snr.npy --tag run_high_snr

# Evaluate
python inference.py --checkpoint checkpoints/run_high_snr --data data/returns_high_snr.npy
```

There are no tests or linting configured.

## Architecture

**Pipeline: generate_data.py → train.py → inference.py**, with `model.py` as shared module.

- **`model.py`** — `TimeSeriesDataset` (sliding window over returns), `FactorTransformer` (encoder-only transformer: linear projection → sinusoidal positional encoding → TransformerEncoder → linear output head), `get_device()` (mps → cuda → cpu), `make_splits()` (time-based train/val/test split with train-only normalization)
- **`generate_data.py`** — Latent factor model: observations = factor loadings × VAR(1) factors + AR(1) idiosyncratic noise. Factor loadings are ground truth only, never fed to the model. Outputs `data/returns_{tag}.npy`, `data/ground_truth_{tag}.npz`, `data/metadata_{tag}.json`
- **`train.py`** — AdamW + cosine LR with linear warmup + early stopping + gradient clipping. Saves checkpoint, config, normalization stats, and training log to `checkpoints/{tag}/`
- **`inference.py`** — Loads checkpoint + saved normalization stats, runs test set, computes metrics (MSE, RMSE, MAE, directional accuracy, R² vs naive-zero baseline), saves metrics JSON and prediction plots to `results/`

## Key Design Decisions

- **Returns not prices** — avoids non-stationarity misleading the model
- **Normalization** — per-stock, train-set stats only; saved as `mean.npy`/`std.npy` in checkpoint dir and reloaded at inference
- **MPS device** — `num_workers=0` in DataLoaders (MPS + multiprocessing breaks on macOS)
- **R² vs naive** — naive baseline is predicting zero (mean of normalized returns). R² near 0 is expected for noisy returns; R² < 0 is a bug signal

## Not Yet Built

- `experiment.py` — sweep runner across data files + model configs, logging to `results/experiment_log.csv`
- `analysis.ipynb` — visualization notebook for cross-run comparison

## Style Preferences

- Code should be readable and commented for learning, not production-clever
- Direct tone — challenge assumptions, flag uncertainty, no filler
