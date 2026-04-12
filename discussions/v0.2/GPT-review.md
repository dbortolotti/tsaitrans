# RL Trading System — Diagnosis, Root Cause, and Refactor Plan

## 1. Executive Summary

The system is **technically correct but economically broken**.

- PPO implementation is stable and functioning
- Transformer produces **high R² (~0.56) predictions**
- Despite this, **PnL is consistently negative**

This is not a training issue.  
This is a **signal design and objective misalignment problem**.

> The system predicts well but predicts the *wrong thing* for trading.

---

# 2. Observed Behaviour (Empirical Reality)

## RL Outcomes

- Negative mean PnL across runs
- Increasing risk (lower `n_sigma`) → **worse PnL**
- Removing penalties → **very large positions + larger losses**
- High trading frequency → **cost accumulation**

## Interpretation

- PPO is learning correctly
- The learned policy is **economically rational given inputs**
- The inputs (signal) are **not monetizable under costs**

---

# 3. Root Cause Analysis

## 3.1 Misaligned Objective (Critical)

Current transformer objective:

```python
minimize (x_{t+1} - prediction)^2
```

Actual trading objective:

```python
maximize E[PnL after costs]
```

These are **not equivalent**.

### Consequence

- Model learns small, precise predictions
- Predictions are **centered near zero**
- Trading decisions based on them → **cost > edge**

---

## 3.2 Signal-to-Noise Problem

From data model:

- SNR ≈ 0.30
- ~77% of variance is noise
- Strong autocorrelation → inflated R²

### Implication

> High R² ≠ tradable signal

The model is largely predicting:
- **noise persistence**
- not **actionable price movement**

---

## 3.3 Horizon Mismatch

Current:
- Predict 1-step ahead

Reality:
- Trading requires **multi-step cumulative edge**

### Consequence

- Edge per trade too small
- Spread dominates

---

## 3.4 No Uncertainty Awareness

Current model outputs:

```python
μ = E[x_{t+1}]
```

Missing:

```python
σ = uncertainty
```

### Consequence

- Small noisy predictions treated as strong signals
- Leads to **overtrading**
- No filtering of low-quality signals

---

## 3.5 RL Is Solving the Wrong Problem

RL is trying to:
- compensate for weak signal
- infer confidence implicitly

This is inefficient and unstable.

> RL cannot manufacture edge — only exploit it.

---

# 4. Key Insight

> The system is in the wrong phase.

You are currently:
- optimizing policy (RL)

But you should be:
- validating and redesigning signal

---

# 5. Refactor Plan

---

# Phase 0 — Economic Validation (MANDATORY)

## Objective

Determine if the signal has **positive expected value after costs**

---

## 0.1 Linear Strategy

```python
position = k * signal
```

Sweep:
- k values

Measure:
- PnL
- turnover

---

## 0.2 Threshold Strategy

```python
if abs(signal) > θ:
    position = sign(signal)
else:
    position = 0
```

---

## 0.3 Signal Bucketing

```python
E[return | signal_bin]
```

Compare to:
- spread / cost

---

## Decision Rule

| Result | Action |
|------|--------|
| No profitability | Redesign signal |
| Weak profitability | Improve signal |
| Strong profitability | Proceed |

---

# Phase 1 — Signal Redesign

## 1.1 Multi-Horizon Prediction

Replace:

```python
predict x_{t+1}
```

With:

```python
predict [x_{t+1}, ..., x_{t+H}]
```

Where:
- H = 10–50

---

## 1.2 Add Uncertainty (Heteroscedastic Head)

Output:

```python
μ_h, log σ_h
```

Loss:

```python
Σ ((y - μ)^2 / σ^2 + log σ^2)
```

---

## 1.3 Construct Trading Signal

Aggregate:

```python
μ_total = Σ μ_h
σ_total = sqrt(Σ σ_h²)
z = μ_total / σ_total
```

---

## Interpretation

- z = signal strength (risk-adjusted)
- replaces raw prediction

---

## 1.4 Re-test Economic Viability

Repeat Phase 0 with `z`.

---

# Phase 2 — Structural Improvements

## 2.1 Cross-Sectional Learning

Current:
- per-stock independent model ❌

Fix:
- joint modelling across assets

---

## 2.2 Reduce Trading Noise

Apply:

```python
if |z| > threshold:
    trade
else:
    hold
```

---

## 2.3 Align with Trading Horizon

- decisions based on cumulative expected return
- not single-step noise

---

# Phase 3 — Policy Layer (RL Optional)

## Role of RL

ONLY:
- execution refinement
- inventory management

NOT:
- signal discovery

---

## 3.1 State

```python
[μ_1..μ_H, σ_1..σ_H, z, position]
```

---

## 3.2 Baseline Policy

```python
position = k * z
```

Use as:
- benchmark
- RL initialization

---

## 3.3 Reward

```python
reward = ΔPnL - inventory_penalty - turnover_penalty
```

---

# Phase 4 — Diagnostics

## 4.1 Edge per Trade

```python
edge = total_pnl / trades
```

Must be > 0

---

## 4.2 PnL vs Turnover

- more trades → worse PnL ⇒ cost dominated

---

## 4.3 Calibration Curve

```python
E[return | z-bin]
```

Should be:
- monotonic
- exceed cost at extremes

---

# 6. Minimal Viable Fix (Fast Path)

If you want speed:

1. Add uncertainty head
2. Compute:

```python
z = μ / σ
```

3. Trade only when:

```python
|z| > threshold
```

4. Position:

```python
position ∝ z
```

Skip RL initially.

---

# 7. Expected Outcomes

## Best Case (~40–50%)
- Signal becomes profitable
- RL adds incremental improvement

## Neutral (~30–40%)
- Only low-frequency trading works

## Worst (~20–30%)
- No edge → redesign data model

---

# 8. Final Conclusion

The system fails because:

- It optimizes **prediction accuracy**, not **economic value**
- It ignores **uncertainty and horizon**
- It relies on RL to compensate for signal weaknesses

---

## Final Principle

> A good trading system requires:
>
> **Expected return + uncertainty + time horizon**
>
> Not just prediction accuracy.

---

## Bottom Line

You do not need a better PPO.

You need a **different signal definition**.

Fix that, and the rest becomes straightforward.
