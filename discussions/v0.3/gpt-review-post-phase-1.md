# RL Trading System — Failure Diagnosis (Phase 1)

## Executive Summary

The reinforcement learning system is **not malfunctioning**. It is behaving optimally under the current specification, which exposes fundamental issues in:

- reward design
- signal strength
- scaling
- objective alignment

The current setup leads to either:
- **no trading (optimal under penalties)**, or
- **correct signal tracking without monetization**

---

# 1. PPO Status

## Observation

- KL divergence stable (~0.002–0.006)
- Entropy decays smoothly
- Value loss decreases
- No instability or collapse

## Conclusion

> PPO is functioning correctly and is not the bottleneck.

**Certainty: 95%**

---

# 2. Experiment A — Market Making Reward

## Outcome

- Mean reward: **-85 → ~0**
- Position magnitude: **→ 0**
- Signal correlation: low (~0.1–0.2)

## Interpretation

The agent converges to:

> "Do nothing"

This is rational because:

- inventory penalties dominate
- spread costs dominate
- signal edge is weak (R² ≈ 0.165)
- credit assignment is noisy

## Conclusion

> The learned policy is optimal under the current reward.

**Certainty: 98%**

---

# 3. Experiment B — Signal Exposure

## Outcome

- Signal correlation: **~0.6+**
- Stable positions (~0.4)
- Mean reward: ~0

## Baseline Comparison

- Baseline reward: **+1.31**
- RL reward: ~0

## Interpretation

- RL successfully learns the signal mapping
- RL fails to monetize it

## Conclusion

> The signal is being extracted, but not converted into profit.

**Certainty: 95%**

---

# 4. Core Problems

## 4.1 Weak Signal (Low SNR)

- R² ≈ 0.165
- Return std ≈ 0.0002

This implies:

- extremely low signal-to-noise ratio
- noisy gradients
- weak policy improvement signal

> RL is optimizing marginal alpha under high stochasticity.

**Certainty: 90%**

---

## 4.2 Broken Reward Scaling

### Evidence

- Experiment A reward std ≈ **400**
- Experiment B reward scale ≈ **0.001**

### Effects

- unstable value function learning
- poor advantage estimation
- ineffective gradients

## Conclusion

> Reward scaling is inconsistent and not controlled.

**Certainty: 95%**

---

## 4.3 Horizon Mismatch

- Transformer predicts: `[1, 2, 4, 8, 16]`
- RL uses: `target_horizon = 16`
- PPO uses: `gamma = 0.99` (~100-step horizon)

### Problem

- prediction horizon ≠ optimization horizon
- inconsistent credit assignment

## Conclusion

> The objective is temporally misaligned.

**Certainty: 90%**

---

## 4.4 No Use of Uncertainty

The transformer provides:

- probabilistic outputs

But RL uses:

- point estimates only

### Consequence

- no confidence weighting
- no noise suppression
- overtrading weak signals

## Conclusion

> Critical information is discarded.

**Certainty: 85%**

---

## 4.5 RL is Premature

Baseline strategy:

- directly uses signal
- achieves positive reward

RL:

- adds complexity
- reduces performance

## Conclusion

> RL is being applied before validating the signal monetization layer.

**Certainty: 95%**

---

# 5. Structural Misunderstanding

## Incorrect Assumption

> RL will unlock signal value

## Reality

> RL already extracts the signal — and there is limited value to extract

---

# 6. Required Actions

## 6.1 Validate Signal Monetization

Before RL:

- Define deterministic strategy:
  position = k × signal

- Run grid search over k
- Evaluate:
  - Sharpe ratio
  - turnover
  - PnL stability
  - horizon sensitivity

### Rule

> If this fails, RL must be abandoned.

---

## 6.2 Fix Reward Scaling

Normalize reward:

- mean ≈ 0
- std ≈ 1

Approaches:

- per-episode normalization
- rolling normalization

### Requirement

> Stable reward scale is mandatory for PPO.

---

## 6.3 Align Horizons

Choose one:

### Option A — Single-step alignment
- reward = PnL at target horizon

### Option B — Multi-step consistency
- aggregate signal into discounted returns

### Rule

> Do not mix prediction and optimization horizons.

---

## 6.4 Use Uncertainty

Incorporate:

position ∝ signal / variance

or:

- penalize high-uncertainty trades

### Objective

> Reduce exposure to noisy predictions.

---

## 6.5 Reduce System Complexity

Current system includes:

- multi-horizon prediction
- probabilistic outputs
- inventory penalties
- execution modeling
- PPO

### Recommendation

Simplify to:

signal → position → PnL

Then reintroduce complexity incrementally.

---

# 7. Final Assessment

| Component | Status |
|----------|--------|
| PPO implementation | ✅ Correct |
| Training stability | ✅ Good |
| Signal extraction | ⚠️ Partial |
| Reward design | ❌ Flawed |
| Scaling | ❌ Broken |
| System design | ❌ Mis-specified |

---

# 8. Bottom Line

> The system is functioning correctly but solving the wrong problem.

The current configuration demonstrates:

- weak signal
- misaligned objectives
- inconsistent scaling

## Final Conclusion

> RL is not underperforming — it is exposing that the current setup does not produce exploitable PnL.

---

# Next Steps (Optional)

- Derive reward gradients vs signal structure
- Design minimal RL formulation with provable behavior
- Build signal-only optimal policy benchmark