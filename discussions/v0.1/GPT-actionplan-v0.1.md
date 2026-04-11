# RL System — Action Plan

## Objective

Stabilize PPO training and ensure meaningful policy learning before refining reward design.

---

# Tier 1 — Mandatory Fixes (Do First)

## 1. Bound Policy Standard Deviation

```python
log_std = torch.clamp(log_std, -5, 2)
```

### Why
- Prevent entropy explosion
- Stabilize training

---

## 2. Keep Entropy Annealing

Already implemented — retain.

### Target behaviour
- High entropy early
- Decreasing entropy over time

---

## 3. Re-run Training (Baseline Reset)

After fixes above:

Check:
- entropy curve
- reward trend
- policy loss

---

# Tier 2 — High Value Improvements

## 4. Replace Clamp with Tanh-Squashed Distribution

```python
z = Normal(mean, std).rsample()
action = torch.tanh(z)

log_prob = base_log_prob - torch.log(1 - action.pow(2) + 1e-6)
```

### Why
- Correct likelihood
- Improve gradient quality

---

## 5. Add KL Monitoring

```python
approx_kl = (old_log_prob - new_log_prob).mean()
```

Stop update if:

```
KL > 0.01–0.02
```

---

## 6. Optional: State-Dependent Exploration

```python
log_std = Linear(state_features)
```

---

# Tier 3 — Experimental Changes

## 7. Remove Cumulative PnL from State

Test:
- with PnL
- without PnL

Compare:
- stability
- reward
- policy behaviour

---

## 8. Evaluate Reward Function Variants

### Current
Uses global R² scaling

### Alternatives

- Fixed scaling (remove R²)
- Local confidence proxy:
  - normalized prediction
  - rolling error estimate

---

# Tier 4 — Longer-Term Improvements

## 9. Introduce Signal Confidence

Replace:

```
global R²
```

with:

```
local confidence(t)
```

---

## 10. Improve Value Function Inputs

Add:
- volatility proxy
- signal uncertainty

---

# Diagnostics Checklist

After each change, verify:

- [ ] Entropy does not diverge
- [ ] Entropy peaks then decreases
- [ ] Policy loss not near zero
- [ ] Reward improves over time
- [ ] Std remains bounded

---

# Expected Outcomes

After Tier 1 + 2:

- Stable entropy
- Non-zero policy updates
- Improved convergence
- Reduced variance across runs

---

# Implementation Order (Strict)

1. Clamp std
2. Re-run training
3. Validate logs
4. Fix action distribution
5. Re-run training
6. Experiment with state/reward

---

# Key Principle

> Do not change reward design until PPO is stable

Otherwise:
- you change signal
- while optimizer is broken
- and cannot interpret results

---
