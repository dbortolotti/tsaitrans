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
Per-step vol = `target_vol / sqrt(steps_per_day)`. Signal and noise are rescaled to:

```
sigma_signal = (target_vol / sqrt(steps_per_day)) * sqrt(snr / (1 + snr))
sigma_noise  = (target_vol / sqrt(steps_per_day)) / sqrt(1 + snr)
```

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

Even an oracle with perfect model knowledge (Kalman filter) achieves roughly:

```
R²_max ≈ SNR/(1+SNR) × spectral_radius²
```

Because `spectral_radius` is close to 1 for all reasonable intraday half-lives,
`R²_max` is driven primarily by `snr`. At `snr=0.3`: `R²_max ≈ 0.23`.

The transformer's context window also matters: with `context_len=60` steps, the
signal at lag 60 has correlation `spectral_radius^60`. At `factor_half_life=0.1`
(spectral_radius ≈ 0.9965), that's `0.9965^60 ≈ 0.81` — substantial signal remains
across the full context window.

## Calibration vs SPX Intraday

| Parameter | Interpretation | Default |
|---|---|---|
| `target_vol` | 2% daily vol | Reasonable for large-cap SPX |
| `snr` | ~23% of variance is factor-driven | Plausible for a well-specified factor model |
| `factor_half_life` | Factors halve in 10% of the day (~25 min at minute frequency) | Stylised intraday momentum |
| `noise_half_life_range` | Idiosyncratic effects last 0.5–2.5% of day (1–5 min) | Microstructure persistence |
