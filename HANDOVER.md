# Handover: Transformer Time Series Experiment

## What This Is

A learning experiment (not production) to understand how transformer models behave when predicting structured synthetic time series data. The user wants to understand the relationship between:
- Model complexity (d_model, n_layers)
- Noise level / SNR
- Transformer prediction quality

---

## What Was Built

Four Python scripts, ready to run on an M3/M4 Mac Mini:

| File | Purpose |
|---|---|
| `generate_data.py` | Generates synthetic returns from a latent factor model |
| `model.py` | Shared module: `TimeSeriesDataset`, `FactorTransformer`, `get_device` |
| `train.py` | Training loop with AdamW, cosine LR + warmup, early stopping |
| `inference.py` | Loads checkpoint, runs test set, computes metrics, saves plots |

---

## Data Generation Model (Equations)

**Observation:**
$$x_{i,t} = \mathbf{\lambda}_i^\top \mathbf{f}_t + \epsilon_{i,t}$$

**Factor dynamics (VAR(1)):**
$$\mathbf{f}_t = \mathbf{A} \mathbf{f}_{t-1} + \boldsymbol{\eta}_t, \quad \boldsymbol{\eta}_t \sim \mathcal{N}(\mathbf{0}, \sigma_f^2 \mathbf{I})$$

**Idiosyncratic noise (AR(1) per stock):**
$$\epsilon_{i,t} = \rho_i \epsilon_{i,t-1} + \sigma_i \xi_{i,t}, \quad \xi_{i,t} \sim \mathcal{N}(0,1)$$

**Key independent variable (SNR per stock):**
$$\text{SNR}_i = \frac{\text{Var}(\mathbf{\lambda}_i^\top \mathbf{f}_t)}{\text{Var}(\epsilon_{i,t})}$$

The user controls SNR via `--sigma_eps` (lower = higher SNR). `sigma_f=1.0` is fixed as signal scale.

---

## Key Design Decisions (and Why)

- **Returns, not prices** — avoids non-stationarity misleading the model
- **Factor loadings NOT used as training features** — the point is to watch the transformer infer structure from prices alone. Loadings are saved in `ground_truth_*.npz` for post-hoc analysis only
- **Noise level tracked via metadata JSON** (not filename) — `data/metadata_{tag}.json` contains all generation params including `sigma_eps`, `mean_snr`, `median_snr`
- **MPS device** — `get_device()` auto-selects mps → cuda → cpu. `num_workers=0` in DataLoaders (MPS + multiprocessing = trouble on macOS)
- **Normalisation** — per-stock, train set stats only. Stats saved as `checkpoints/<tag>/mean.npy` and `std.npy` and reloaded at inference to prevent leakage

---

## Recommended Problem Size (M3/M4, ~1hr budget)

| Parameter | Value |
|---|---|
| n_stocks | 50 |
| n_timesteps | 2000 |
| n_factors | 3 |
| context_len | 60 |
| horizon | 1 |
| d_model | 64 |
| n_heads | 4 |
| n_layers | 3 |
| ffn_dim | 256 |
| batch_size | 128 |
| ~parameters | 1-2M |

---

## How to Run

```bash
# Install
pip install torch numpy matplotlib

# Generate data at different noise levels
python generate_data.py --sigma_eps 0.5 --tag high_snr
python generate_data.py --sigma_eps 2.0 --tag low_snr

# Train
python train.py --data data/returns_high_snr.npy --tag run_high_snr
python train.py --data data/returns_low_snr.npy  --tag run_low_snr

# Evaluate
python inference.py --checkpoint checkpoints/run_high_snr --data data/returns_high_snr.npy
python inference.py --checkpoint checkpoints/run_low_snr  --data data/returns_low_snr.npy
```

---

## What Still Needs Building

- [ ] `experiment.py` — sweep runner that iterates over data files + model configs, logs to `results/experiment_log.csv`
- [ ] `analysis.ipynb` — plots: val loss vs model size, test MSE vs noise level, prediction vs ground truth for best model
- [ ] These were described in the Claude Code prompt (`transformer_experiment_prompt.md`) but not yet coded

---

## Outputs Per Run

```
checkpoints/<tag>/
    best_model.pt       # best val loss checkpoint
    config.json         # full config dict
    mean.npy            # train set normalisation mean (1, n_stocks)
    std.npy             # train set normalisation std  (1, n_stocks)
    train_log.csv       # per-epoch train/val loss

results/
    <tag>_metrics.json  # MSE, MAE, RMSE, directional accuracy, R² vs naive
    <tag>_predictions.png
```

---

## Metrics to Watch

- **R² vs naive** — naive baseline is predicting zero (mean of normalised returns). R² near 0 is normal for noisy returns. R² < 0 is a bug signal.
- **Directional accuracy** — did the model get the sign right? Useful intuition check.
- **Don't fixate on absolute MSE** — compare across runs, not to an external benchmark.

---

## User Preferences

- Direct and honest advisor tone — challenge assumptions, flag uncertainty explicitly
- No agreeable filler
- Cite confidence level on facts and opinions
- PyTorch only — no HuggingFace, no Lightning
- Code should be readable and commented for learning, not production-clever
- Cannot push to GitHub from Claude (no git tool connected)
