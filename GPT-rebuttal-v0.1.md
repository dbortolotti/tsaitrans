# RL Audit — Final Technical Position (Post Discussion)

## Scope

This document captures the refined technical assessment after:
- Initial PPO + RL audit
- Review of training logs
- Review of `CLAUDE-response-v0.1.md`
- Review of `RL_DESIGN.md`

Focus: PPO + reward design + system-level behaviour

---

# Executive Summary

The system is **conceptually sound and thoughtfully designed**, particularly:

- Clear problem framing (execution / quoting control)
- Coherent reward derivation based on risk vs expected alpha
- Clean action parameterization (width + skew)

However, current training behaviour shows:

> The system fails primarily due to **PPO instability and entropy collapse**,  
> not due to reward design.

The most important correction from earlier review:

> The reward function is **not conceptually invalid**, but is based on a **strong simplifying assumption** that limits robustness.

---

# 1. PPO Assessment (Updated)

## 1.1 Action Distribution

Original claim: "mathematically broken"  
Revised position:

- The current implementation is **formally incorrect** (log-prob mismatch due to clamp)
- But impact depends on regime:
  - Low variance → minor
  - High variance (observed) → significant

### Final position

> This is a **moderate-to-high impact issue in your current setup**, but not a theoretical showstopper.

---

## 1.2 Entropy Explosion

Confirmed from logs:

- Entropy diverges (~3 → ~49)
- Policy loss collapses (~0)
- Reward stagnates

### Root cause

- Unbounded `log_std`
- Entropy incentive dominates reward scale

### Final position

> This is the **primary failure mode of the system**

---

## 1.3 Exploration Design

- State-independent std is acceptable baseline
- But suboptimal for this environment

### Final position

> Improvement opportunity, not critical flaw

---

# 2. Reward Function — Final Assessment

## 2.1 What the design does

From `RL_DESIGN.md`:

```text
Expected alpha ≈ R² · σ · sqrt(T - t)
```

→ leads to:

```text
p_crit(t) = R² · sqrt(T - t)
```

→ penalty:

```text
penalty ∝ (p / (R² · sqrt(T - t)))²
```

---

## 2.2 Validity

### Strengths

- Time-consistent scaling (`sqrt(T - t)`)
- Clear economic intuition:
  - larger signal → larger justified inventory
- Clean derivation based on "risk vs alpha"

### Verdict

> The formulation is **internally consistent and well motivated**

---

## 2.3 Core limitation

The key assumption:

```text
R² ≈ expected tradable alpha
```

This is not generally true.

### Issues

- R² is:
  - global
  - variance-based
- but decision requires:
  - local
  - directional
  - uncertainty-aware information

---

## 2.4 Structural implication

The system uses:

```text
global model quality → local risk constraint
```

This creates:

- no adaptation to local signal quality
- no regime awareness

---

## 2.5 Final position

> The reward function is:
- **Not wrong**
- **Not optimal**
- **Appropriate for a controlled toy model**
- **Too coarse for general RL trading**

---

# 3. Cumulative PnL in State

## Observation

Included as state variable:

```text
cumulative_pnl
```

---

## Effect

- Introduces path dependence
- Creates implicit utility shaping
- Not explicitly modeled

---

## Final position

> Acceptable as a design choice, but:
- high leverage
- should be explicitly validated

---

# 4. System-Level Interaction Effects

The key insight not captured in individual reviews:

## Failure is caused by interaction, not individual components

```text
unbounded std
+ entropy bonus
+ imperfect log_prob
→ entropy explosion
→ policy collapse
```

---

## Implication

- Individual issues may appear minor
- Combined effect is dominant

---

# 5. What Was Overstated in Initial Audit

### Revised positions

| Original claim | Revised |
|------|--------|
"PPO broken" | PPO degraded under current regime |
"Reward invalid" | Reward valid but assumption-heavy |
"Critical across board" | Issues are tiered |

---

# 6. What Was Understated in Claude Response

### Key gap

Claude evaluates issues independently.

Missing:

> **compound system dynamics**

---

# 7. Final System Characterization

The system is:

> A controlled RL environment optimizing a **risk-aware quoting policy conditioned on a predictive signal**

It is:

- well-structured
- interpretable
- but currently limited by:
  - exploration dynamics
  - coarse uncertainty modelling

---

# 8. Bottom Line

## What matters most

1. PPO stability (primary bottleneck)
2. Exploration control
3. Only then: reward refinement

---

## What does NOT currently matter

- fine reward tuning
- architectural redesign

---

# Certainty Summary

| Topic | Certainty |
|------|----------|
Entropy collapse is root issue | 100% |
Reward is internally consistent | 95% |
R² assumption is limiting | 90% |
Action distribution issue moderate impact | 90% |
PnL state effect context-dependent | 80% |

---