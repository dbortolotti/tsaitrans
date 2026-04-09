# Handover: Transformer + RL Market Making Experiment

## What This Is

A learning experiment (not production) with two modules:

1. **Prediction** — Univariate encoder-only transformer for synthetic time series forecasting. Explores how model complexity and noise level (SNR) affect prediction quality.
2. **Placing** — RL market-making agent (PPO) that uses transformer predictions as features to learn bid/ask placement.

The transformer and RL agent are trained separately. The transformer is frozen during RL training — its predictions are just input features to the policy.

Runs on M3/M4 Mac Mini via MPS. PyTorch only — no HuggingFace, no Lightning.

---

## How To Run

Everything is driven by a single experiment JSON config:

```bash
pip install torch numpy matplotlib gymnasium

# Run the full pipeline (generate data → train transformer → inference → train RL)
python run_experiment.py experiments/example.json

# All outputs go to output/<experiment_name>/
```

The JSON only needs to specify parameters you want to vary — everything else uses defaults from `run_experiment.py:DEFAULTS`.

### Example config (`experiments/example.json`)

```json
{
  "data": {
    "sigma_eps": 0.5,
    "stocks_transformer_train": 3,
    "stocks_transformer_val": 2,
    "stocks_rl_train": 2,
    "stocks_rl_val": 2,
    "stocks_test": 1
  },
  "transformer": { "d_model": 64, "n_layers": 3, "n_epochs": 50 },
  "rl": { "lambda_inv": 0.01, "kappa_spread": 0.0005, "n_iterations": 200 }
}
```

Individual scripts can still be run standalone — see their `--help` for args.

---

## Architecture

```
┌────────────────────┐     ┌──────────────┐     ┌─────────────┐
│ generate_data.py   │────▶│  train.py     │────▶│ Transformer │
│ (factor model)     │     │  (prediction) │     │ (frozen)    │
└────────────────────┘     └──────────────┘     └──────┬──────┘
                                                       │ predictions
                                                       ▼
┌────────────────────┐     ┌──────────────┐     ┌─────────────┐
│ market_env.py      │◀───│ train_rl.py   │────▶│ policy.py   │
│ (gym environment)  │     │ (PPO loop)    │     │ (actor-     │
└────────────────────┘     └──────────────┘     │  critic MLP)│
                                                └─────────────┘
```

### Prediction (`prediction/`)

| File | Purpose |
|---|---|
| `generate_data.py` | Latent factor model: observations = loadings × VAR(1) factors + AR(1) noise |
| `model.py` | `TimeSeriesDataset` (per-stock sliding windows), `FactorTransformer` (univariate, n_stocks=1) |
| `train.py` | AdamW + cosine LR with warmup + early stopping + grad clipping |
| `inference.py` | Loads checkpoint, runs on test stocks, computes metrics, saves plots |

### Placing (`placing/`)

| File | Purpose |
|---|---|
| `market_env.py` | Gymnasium env: order placement, fills, inventory tracking |
| `policy.py` | ActorCritic MLP (~5k params), PPO rollout buffer, update logic |
| `train_rl.py` | Loads data + transformer, runs PPO over vectorized envs |
| `generate_demo_data.py` | Creates fake `sim_results.json` for testing the visualizer |
| `trading_visualizer.jsx` | Interactive React component for price/bid/ask/position/PnL visualization |

---

## Data Generation Model

$$x_{i,t} = \mathbf{\lambda}_i^\top \mathbf{f}_t + \epsilon_{i,t}$$

- **Factors:** $\mathbf{f}_t = \mathbf{A} \mathbf{f}_{t-1} + \boldsymbol{\eta}_t$ (VAR(1), spectral radius controls persistence)
- **Noise:** $\epsilon_{i,t} = \rho_i \epsilon_{i,t-1} + \sigma_i \xi_{i,t}$ (AR(1) per stock)
- **SNR:** $\text{SNR}_i = \text{Var}(\mathbf{\lambda}_i^\top \mathbf{f}_t) / \text{Var}(\epsilon_{i,t})$

Control SNR via `sigma_eps` (lower = higher SNR). `sigma_f=1.0` is the signal scale.

---

## Key Design Decisions

- **Stocks as realizations** — each stock is an i.i.d. sample from the same DGP. Splits are across stocks, not time. The 5 stock groups (transformer train/val, RL train/val, test) are disjoint.
- **RL trains on transformer-OOS stocks** — the RL agent sees predictions on stocks the transformer never trained on, so it learns from realistic (not overfit) prediction quality.
- **Univariate model** — the transformer processes one stock at a time (n_stocks=1). All stocks in a split are pooled into one training dataset.
- **Normalized RL inputs** — returns and predictions are standardized (using transformer's normalization stats) before feeding to the market env, preventing price explosion from large raw returns.
- **Returns not prices** — avoids non-stationarity.
- **Factor loadings not used as features** — the transformer must infer structure from returns alone. Loadings saved in `ground_truth_*.npz` for analysis only.
- **Normalization** — scalar mean/std from transformer train stocks. Saved in checkpoint dir, reloaded at inference.
- **MPS device** — `num_workers=0` in DataLoaders (MPS + multiprocessing = trouble on macOS).

---

## Stock Split

Controlled by 5 parameters in the experiment config:

| Stocks | Group | Purpose |
|---|---|---|
| 0–2 | `stocks_transformer_train` | Train the transformer |
| 3–4 | `stocks_transformer_val` | Validate / early-stop the transformer |
| 5–6 | `stocks_rl_train` | Train the RL agent (transformer OOS) |
| 7–8 | `stocks_rl_val` | Validate the RL agent |
| 9 | `stocks_test` | Final evaluation (both models) |

Total `n_stocks` is the sum of all 5 group sizes.

---

## Reward Function

```
reward = realized_pnl - λ * position² - κ * spread_cost
```

| Term | Default | Tuning |
|---|---|---|
| `λ (lambda_inv)` | 0.01 | Increase if agent holds too long; decrease if it barely trades |
| `κ (kappa_spread)` | 0.0005 | Increase to force tighter quotes; decrease if agent never gets filled |

Based on Avellaneda-Stoikov. The quadratic inventory penalty is critical — without it the agent takes directional bets instead of market-making.

---

## Environment Details

**Observation (5D):** predicted return, position (normalized), realized vol, inventory age (normalized), last return.

**Action (2D continuous):** bid/ask offsets in [-1, 1], rescaled to [min_offset, max_offset] in vol units.

**Fill logic:** if next_price ≤ bid → buy; if next_price ≥ ask → sell. Hard position limit ±10.

---

## Output Structure

```
output/<experiment_name>/
    <name>.json              # copy of experiment config
    resolved_config.json     # config with all defaults filled in
    data/
        returns_<name>.npy   # (T, n_stocks) returns
        ground_truth_<name>.npz
        metadata_<name>.json
    checkpoints/
        best_model.pt        # best val loss transformer
        config.json
        mean.npy, std.npy    # normalization stats (scalars)
        train_log.csv
    results/
        metrics.json         # MSE, R², directional accuracy
        predictions.png
    checkpoints_rl/
        best_policy.pt
        final_policy.pt
        config.json
        train_log.json
```

---

## Metrics to Watch

- **R² vs naive** — baseline is predicting zero. R² near 0 is normal for noisy returns. R² < 0 is a bug signal.
- **Directional accuracy** — did the model get the sign right?
- **RL mean reward** — should start negative and improve. Entropy should decline as policy converges.

---

## What To Experiment With

- **Vary `sigma_eps`** — higher noise = harder prediction = agent relies more on spread capture than alpha.
- **Compare transformer vs momentum** — set `"predictor": "momentum"` in the RL config.
- **Vary `lambda_inv`** — watch how position behavior changes.
- **Model size** — vary `d_model`, `n_layers` to see complexity vs. prediction quality tradeoff.
- **Stock split sizes** — more transformer train stocks = better predictions but fewer RL train stocks.

---

## Known Limitations

1. **No adverse selection.** Real market makers get picked off by informed traders.
2. **Single stock per env.** A real market maker would manage a portfolio.
3. **No latency / queue priority.** Fills are instantaneous and guaranteed if price crosses.
4. **Reward scale sensitivity.** Changing `sigma_eps` significantly may require re-tuning `lambda_inv` and `kappa_spread`.

---

## User Preferences

- Direct tone — challenge assumptions, flag uncertainty, no filler
- PyTorch only — no HuggingFace, no Lightning
- Code readable and commented for learning, not production-clever
