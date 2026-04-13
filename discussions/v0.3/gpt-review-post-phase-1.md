# RL Diagnosis — Why Learning is Failing

## 1. Core Observation (from logs)

### Transformer
- Converges normally
- Probabilistic (heteroscedastic NLL)
- Multi-horizon targets: `[1, 2, 4, 8, 16]`
- Loss stabilises ~2.4–2.45 → not noise, but not high SNR either

### RL Behaviour
- Stable PPO (no divergence)
- Low entropy
- Near-zero actions
- Near-zero reward

**Conclusion:**
> The policy has found that any action is worse than no action.

This is not an optimisation failure.  
It is a correct solution to the objective.

---

## 2. Root Cause Breakdown

### 2.1 Signal-to-noise vs penalty imbalance

Transformer:
- Weak but real signal

RL sees:
- Noisy immediate reward
- Strong penalties (inventory / variance / costs)

Expected value:
```
E[action reward] ≈ small positive signal − penalties − noise
                ≈ negative or ~0
```

→ Optimal policy = do nothing

---

### 2.2 Horizon mismatch (critical)

| Component | Time scale |
|----------|------------|
| Signal   | Multi-horizon (forward-looking) |
| Reward   | Immediate / step-wise |

Problem:
- Agent cannot connect prediction at `t` with PnL at `t+H`
- Signal appears as noise

---

### 2.3 Uncertainty is ignored

Transformer outputs:
- Mean (μ)
- Variance (σ)

RL uses:
- Mean only

Implication:
- High and low confidence predictions treated equally
- Optimal behaviour = ignore signal

---

### 2.4 Action space too flexible

Current:
```
action ∈ [-1, 1]
```

With weak signal:
- Continuous optimisation collapses to zero

---

### 2.5 Reward mixes multiple objectives

Reward includes:
- PnL
- Inventory penalty
- Possibly variance / costs

Effect:
- Signal diluted across competing gradients
- RL cannot isolate alpha

---

## 3. Incorrect Assumption

Assumption:
> If the signal has edge, PPO should discover it

False because:
- PPO is myopic with noisy gradients
- Signal is low SNR
- Reward is misaligned
- Exploration is penalised immediately

---

## 4. Structural Diagnosis

Current system:
```
state_t = {features, signal_t}
action_t ~ policy(state_t)
reward_t = f(PnL_t, penalties_t)
```

Required structure:
```
reward_t ≈ alignment(signal_t, future_return)
```

Current issue:
> Reward is not functionally aligned with the signal

---

## 5. Fix Plan

### Phase 1 — Force signal learning

Replace reward with:
```
R_t = sign(action_t) * future_return_{t,H}
```

Constraints:
- Use a single horizon (e.g. H=4 or H=8)
- Remove:
  - Inventory penalty
  - Costs
  - Execution complexity

---

### Phase 2 — Use uncertainty

From transformer:
```
μ_t, σ_t
```

Define:
```
signal_strength = μ_t / σ_t
```

Options:
- Threshold:
  ```
  trade only if |μ/σ| > k
  ```
- Reward scaling:
  ```
  R_t = action_t * μ_t / σ_t
  ```

---

### Phase 3 — Discretise actions

Replace:
```
action ∈ [-1,1]
```

With:
```
action ∈ { -1, 0, +1 }
```

Reason:
- Reduces optimisation ambiguity
- Forces directional learning

---

### Phase 4 — Align reward horizon

If H = 8:
```
R_t = action_t * (price_{t+8} - price_t)
```

Not step-wise PnL

---

### Phase 5 — Remove penalties

Temporarily remove:
- Inventory penalties
- Transaction costs

Reintroduce later

---

## 6. High-Confidence Conclusions

Confidence ≈ 0.9

- Reward misalignment is dominant failure
- Horizon mismatch blocks learning
- Ignoring uncertainty wastes model output
- Continuous actions accelerate collapse

---

## 7. Uncertain Areas

Confidence ≈ 0.6–0.7

- True magnitude of signal edge
- Out-of-sample stability
- Possible feature leakage or regime mismatch

---

## 8. Direct Assessment

Current issues:
- Solving execution + alpha simultaneously
- Multi-horizon signal vs single-step reward
- Penalising exploration prematurely
- Ignoring uncertainty
- Over-relying on PPO to discover structure

---

## Bottom Line

Your RL is not broken.

> The objective function is incoherent relative to the signal.

Until this is fixed, no tuning or architectural changes will produce learning.