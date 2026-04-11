# Reinforcement Learning System Audit (PPO + Environment)

## Scope
This audit focuses on:
- PPO implementation
- Training dynamics (logs)
- RL environment and reward design

Transformer component intentionally excluded.

---

# Executive Summary

The system is conceptually well-structured but currently fails to learn meaningful policies due to:

- A mathematically incorrect action distribution
- Unbounded exploration (entropy collapse into noise)
- Reward mis-specification
- Mismatch between intended design and implementation

The most critical issue is:

> The PPO policy is optimizing an incorrect likelihood function, causing unstable and ineffective learning.

---

# 1. Critical Issues

## 1.1 Incorrect Action Distribution (PPO is mathematically broken)

### Current implementation

```python
action = Normal(mean, std).sample()
action = clamp(action)
log_prob = Normal(mean, std).log_prob(action)
```

### Problem

- Sampling and likelihood are inconsistent
- Clipping is not part of the distribution
- PPO ratio is therefore incorrect

### Impact

- Biased gradients
- Incorrect policy updates
- Learning instability

### Severity
**Critical (blocks correct learning)**

---

## 1.2 Entropy Explosion (Policy collapses into noise)

### Observed in logs

- Entropy: ~3 → ~49
- Policy loss → ~0
- Reward flat or degrading

### Interpretation

- Policy variance (std) is exploding
- Agent maximizes entropy instead of reward

### Root cause

- `log_std` is unbounded
- Entropy bonus incentivizes variance increase
- Reward scale too small relative to entropy

### Impact

- Policy becomes nearly random
- Learning signal disappears

### Severity
**Critical**

---

## 1.3 Reward Function Mis-specification

### Problems

- Use of R² in denominator is conceptually invalid
- Dimensional inconsistency
- Time scaling differs from design

### Impact

- Incorrect optimal policy
- Over-penalization when model is weak

### Severity
**High**

---

## 1.4 State Includes Cumulative PnL

### Problem

- Introduces path dependence
- Breaks stationarity

### Impact

- Unstable learning
- Implicit unintended utility shaping

### Severity
**High**

---

## 1.5 Exploration is State-Independent

### Problem

- Same exploration across all states

### Impact

- Inefficient exploration
- Slower convergence

### Severity
**Medium-High**

---

# 2. Observed Training Behaviour

| Metric        | Behaviour |
|--------------|----------|
| Reward        | Flat |
| Policy loss   | ~0 |
| Entropy       | Explodes |
| Value loss    | No clear improvement |

### Interpretation

The agent is effectively optimizing:

> maximize entropy subject to weak reward constraints

---

# 3. Action Plan

## Tier 1 — Mandatory Fixes

### 1. Fix Action Distribution

```python
z = Normal(mean, std).rsample()
action = torch.tanh(z)

log_prob = base_log_prob - torch.log(1 - action.pow(2) + 1e-6)
```

Remove:
- clamp
- tanh on mean

---

### 2. Constrain Standard Deviation

```python
log_std = torch.clamp(log_std, -5, 2)
```

---

### 3. Reduce Entropy Coefficient

```
entropy_coef ≈ 1e-3 or lower
```

---

### 4. Fix Reward Function

```python
position_risk = position * target_vol * price

remaining_steps = max(T - t, 1)
risk_budget = n_sigma * target_vol * sqrt(remaining_steps)

penalty = (position_risk / risk_budget)**2

reward = MTM_PnL - penalty
```

---

### 5. Remove Cumulative PnL from State

---

## Tier 2 — Improvements

### 6. State-Dependent Exploration

```python
log_std = Linear(state_features)
```

---

### 7. Normalize Value Targets

```python
returns = (returns - mean) / (std + 1e-8)
```

---

### 8. Add KL Monitoring

```python
approx_kl = (old_log_prob - new_log_prob).mean()
```

Stop if:
```
KL > 0.01–0.02
```

---

# 4. Expected Behaviour After Fixes

- Entropy increases early, then decreases
- Policy loss non-zero and fluctuating
- Reward improves over time
- Std stabilizes

---

# 5. Diagnostic Checklist

- [ ] Entropy does not diverge
- [ ] Policy loss not ~0
- [ ] Reward trending upward
- [ ] KL small but non-zero
- [ ] Std bounded

---

# Final Assessment

The system is structurally promising, but currently:

> It is optimizing the wrong objective with incorrect gradients and unconstrained exploration.

Fixing the action distribution and entropy dynamics is the highest leverage step.
