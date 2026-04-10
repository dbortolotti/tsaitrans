# Data Generation Model

## Latent Factor Model

Synthetic stock returns are drawn from a standard factor model:

```
x_{i,t} = lambda_i' @ f_t + epsilon_{i,t}        (observation equation)
f_t     = A @ f_{t-1} + eta_t,  eta_t ~ N(0, I)  (shared factors, VAR(1))
eps_{i,t} = rho_i * eps_{i,t-1} + xi_{i,t}       (idiosyncratic noise, AR(1))
```

- `lambda_i ~ N(0, I)` are per-stock factor loadings — ground truth only, never seen by the model
- `A` is constructed to have a given spectral radius (derived from `factor_half_life`)
- `rho_i` is drawn uniformly from a derived range (from `noise_half_life_range`)

The transformer's task is to predict `x_{i,t+1}` from a window of past returns. It never sees factors, loadings, or noise separately.

## Free Parameters

| Parameter | What it controls | Default |
|---|---|---|
| `target_vol` | Daily volatility (std of cumulative return over one day) | 0.02 |
| `snr` | Signal variance / noise variance | 0.3 |
| `factor_half_life` | Factor mean-reversion half-life, as fraction of a trading day | 0.1 |
| `noise_half_life_range` | (min, max) idiosyncratic half-life, as fraction of a trading day | [0.005, 0.025] |
| `n_factors` | Dimensionality of the shared signal space | 3 |
| `steps_per_day` | Number of timesteps in one trading day | 2000 |

### Volatility scaling

`target_vol` is the **daily** volatility — the std of the cumulative return over one trading day.

Calibration is a two-step process:

**Step 1 — set the SNR ratio** using a naive i.i.d. approximation:

```
sigma_signal = (target_vol / sqrt(steps_per_day)) * sqrt(snr / (1 + snr))
sigma_noise  = (target_vol / sqrt(steps_per_day)) / sqrt(1 + snr)
```

This correctly partitions variance between signal and noise.

**Step 2 — correct for autocorrelation inflation.** The i.i.d. formula `daily_vol = per_step_vol * sqrt(steps_per_day)` only holds for independent returns. The VAR(1) factor structure and AR(1) noise create strong autocorrelation, which inflates the actual daily vol (std of the cumulative sum) well beyond the i.i.d. prediction. The inflation factor can be large:

| factor_half_life | noise_half_life | snr | inflation factor |
|---|---|---|---|
| 0.10 (200 steps) | 0.005–0.025 | 0.3 | ~6× |
| 0.01 (20 steps)  | 0.005–0.025 | 0.3 | ~5× |
| 0.01 (20 steps)  | 0.001–0.005 | 0.3 | ~3× |
| 0.005 (10 steps) | 0.001–0.003 | 1.0 | ~1.5× |

To correct, after step 1 the entire returns array is rescaled uniformly so that `std(sum(returns, axis=0)) == target_vol`. This preserves the SNR ratio (both components scaled equally) and the autocorrelation structure.

### Half-life parametrisation

`factor_half_life` and `noise_half_life_range` are expressed as **fractions of a trading day**,
making them independent of `steps_per_day`. The per-step AR parameters are derived as:

```
spectral_radius = 0.5 ** (1 / (factor_half_life * steps_per_day))
rho_i           = 0.5 ** (1 / (noise_half_life_i * steps_per_day))
```

Example half-lives at `steps_per_day=2000`:

| factor_half_life | Steps | spectral_radius |
|---|---|---|
| 0.05 | 100 | 0.9931 |
| 0.10 | 200 | 0.9965 |
| 0.25 | 500 | 0.9986 |
| 0.50 | 1000 | 0.9993 |

## Predictability Ceiling

The oracle R² — the maximum achievable by a predictor with access to the true latent signal — is:

```
oracle R² ≈ corr(signal[t], returns[t+1])²   (averaged across stocks)
```

This differs from the naive `SNR/(1+SNR)` formula because the signal at time `t` predicts `returns[t+1]` only to the extent that the signal persists (i.e., depends on `spectral_radius`). Empirically measured values:

| factor_half_life | noise_half_life | snr | lag-1 AC | oracle R² |
|---|---|---|---|---|
| 0.10 | 0.005–0.025 | 0.3 | 0.60 | 0.18 |
| 0.01 | 0.005–0.025 | 0.3 | 0.61 | 0.16 |
| 0.01 | 0.001–0.005 | 0.3 | 0.51 | 0.16 |
| 0.005 | 0.001–0.003 | 1.0 | 0.21 | 0.26 |

**Lag-1 autocorrelation** measures how easy the task is for any predictor, including naive momentum. High lag-1 AC means even a simple rolling-mean baseline does well. The fourth config has lower AC (harder to detect) but higher oracle R² (more genuine signal), making it the most interesting for evaluating a real model vs a momentum shortcut.

### Appropriate context length

`context_len` should cover several factor half-lives. Beyond ~4–5 half-lives the remaining signal correlation is <5%, so additional context adds noise without information:

```
context_len ≈ 5 × (factor_half_life × steps_per_day)
```

| factor_half_life | steps_per_day | recommended context_len |
|---|---|---|
| 0.10 | 2000 | ~200 (default) |
| 0.01 | 2000 | ~40–50 |
| 0.005 | 2000 | ~30–40 |

## Calibration vs SPX Intraday

| Parameter | Interpretation | Default | experiment2 |
|---|---|---|---|
| `target_vol` | 2% daily vol | 0.02 | 0.02 |
| `snr` | Signal fraction of per-step variance | 0.3 (~23%) | 1.0 (50%) |
| `factor_half_life` | Factors halve in 10% of day (~25 min at 1-min bars) | 0.10 | 0.005 (~1 min at 1-min bars) |
| `noise_half_life_range` | Idiosyncratic noise persistence | [0.005, 0.025] | [0.001, 0.003] |

The default config produces visually smooth, trend-dominated price paths due to the long factor half-life. `experiment2` uses much shorter half-lives that produce jaggier, more realistic intraday-looking series, while having a higher oracle R² — making it a better test of whether the transformer learns genuine patterns vs. momentum.
