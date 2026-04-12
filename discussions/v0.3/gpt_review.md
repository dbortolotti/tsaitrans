# Quantitative Summary of Proposed Phase 1 Changes

## Context from Current Results

### Transformer quality
- horizons used: `[1, 2, 4, 8, 16]`
- `r2_h1 = 0.5581`
- `r2_h2 = 0.4272`
- `r2_h4 = 0.3479`
- `r2_h8 = 0.2212`
- `r2_h16 = 0.1037`

### Uncertainty / signal strength
- `mean_abs_z_h1 = 0.8183`
- `mean_abs_z_h2 = 0.7190`
- `mean_abs_z_h4 = 0.6127`
- `mean_abs_z_h8 = 0.4620`
- `mean_abs_z_h16 = 0.2804`
- `|z_h16| > 1` ≈ `1.4%`
- `|z_h16| > 2` ≈ `0%`

### Analytical backtest
- linear strategy Sharpe ≈ `23.35`
- thresholded strategies also clearly positive
- monotonic bucket structure

### RL outcome
- mean reward: strongly negative → ~0
- best mean reward ≈ `0`
- entropy collapses
- policy variance shrinks steadily
- reward variance drops significantly

---

# Interpretation

- The **predictive signal is real and economically meaningful**
- The **RL system is stable but converging to inactivity**
- The failure is not numerical instability, but **objective misalignment**

---

# Weaknesses of the Previous RL Approach (Detailed)

## 1. Reward mixes multiple objectives simultaneously

Previous reward implicitly combined:
- directional PnL (via MTM)
- spread capture
- inventory penalties
- execution/friction effects

This creates:

$begin:math:display$
r\_t \= f\(\\text\{signal\}\, \\text\{inventory\}\, \\text\{execution\}\, \\text\{noise\}\)
$end:math:display$

### Problem

PPO receives a **high-variance, multi-factor reward signal** with unclear attribution:
- Was the trade bad because direction was wrong?
- Or because inventory penalty dominated?
- Or because spread/friction wiped out gains?

This leads to **poor credit assignment**.

---

## 2. Weak signal-to-noise ratio in reward

Given:
- `mean_abs_z_h16 ≈ 0.28`
- strong signals are rare

Most timesteps:
- expected edge is small
- realized returns are noisy

So:

$begin:math:display$
\\text\{signal term\} \\ll \\text\{noise \+ penalties\}
$end:math:display$

### Consequence

The policy experiences:
- noisy rewards
- inconsistent gradients
- low confidence learning signal

Optimal PPO response:
> reduce exposure → minimize variance → converge to near-zero action

---

## 3. Misalignment between prediction horizon and reward horizon

The model predicts:
$begin:math:display$
\\mu\_h \= E\[\\text\{mid\}\_\{t\+h\} \- \\text\{mid\}\_t\]
$end:math:display$

But reward was dominated by:
$begin:math:display$
\\Delta \\text\{mid\}\_\{t\+1\}
$end:math:display$

### Problem

The policy is asked to:
- act on multi-step predictions
- but rewarded on mostly **one-step outcomes**

This creates:
- inconsistent learning target
- noisy feedback loop
- poor exploitation of longer-horizon signal

---

## 4. Overlapping reward windows and credit diffusion

Using short-horizon MTM reward repeatedly:

$begin:math:display$
r\_t \\sim p\_t \\cdot \(\\text\{mid\}\_\{t\+1\} \- \\text\{mid\}\_t\)
$end:math:display$

while the signal predicts multi-step returns means:
- the same future movement contributes to multiple rewards
- reward attribution is spread across many actions

### Consequence

- PPO struggles to associate actions with outcomes
- gradients become noisy and diluted

---

## 5. Inventory penalty dominates learning early

If inventory penalty is significant:

$begin:math:display$
\-\\lambda \\cdot p\_t\^2
$end:math:display$

and signal is weak/noisy:

$begin:math:display$
\|p\_t \\cdot R\_t\| \\approx 0
$end:math:display$

then expected reward becomes:

$begin:math:display$
E\[r\_t\] \\approx \-\\lambda p\_t\^2
$end:math:display$

### Optimal solution

$begin:math:display$
p\_t \= 0
$end:math:display$

This matches observed behaviour:
- shrinking policy variance
- collapsing entropy
- near-zero actions

---

## 6. Exploration is punished asymmetrically

In the current setup:
- taking action exposes the agent to:
  - noisy returns
  - penalties
- doing nothing yields:
  - stable ~0 reward

### PPO implication

Exploration becomes **negatively rewarded**, so PPO:
- reduces entropy
- shrinks action variance
- converges to inactivity

---

## 7. Signal structure was previously underutilized

Before refactor:
- multi-horizon predictions were collapsed to a scalar
- term structure information was lost

This prevented distinguishing:
- transient vs persistent signals
- high vs low confidence regimes

Even after fixing this structurally, the reward still did not exploit it.

---

# Summary of Failure Mode

The previous RL setup effectively optimized:

$begin:math:display$
\\text\{minimize risk and variance\} \\Rightarrow p\_t \\approx 0
$end:math:display$

because:
- reward signal was weak relative to penalties
- objectives were mixed
- horizon alignment was poor
- credit assignment was noisy

---

# Phase 1 Reward Design

## New reward

$begin:math:display$
r\_t \= c \\cdot p\_t \\cdot R\_t \- \\alpha p\_t\^2 \- \\beta \(p\_t \- p\_\{t\-1\}\)\^2
$end:math:display$

Where:
- $begin:math:text$ p\_t $end:math:text$: position at time $begin:math:text$ t $end:math:text$
- $begin:math:text$ p\_\{t\-1\} $end:math:text$: previous position
- $begin:math:text$ R\_t $end:math:text$: realized future return over chosen horizon
- $begin:math:text$ c $end:math:text$: reward scale
- $begin:math:text$ \\alpha $end:math:text$: position penalty
- $begin:math:text$ \\beta $end:math:text$: turnover penalty

---

## Target return definition

$begin:math:display$
R\_t \= \\text\{mid\}\_\{t\+16\} \- \\text\{mid\}\_t
$end:math:display$

This aligns reward with the **longest modeled horizon**.

---

# Initial Parameter Values

| Parameter | Value | Role |
|---|---:|---|
| `reward_mode` | `"signal_exposure"` | simplified objective |
| `target_horizon` | `16` | aligns reward with signal |
| `reward_scale` | `10.0` | strengthens learning signal |
| `alpha_pos` | `0.01` | controls position size |
| `beta_trade` | `0.005` | controls turnover |
| `gamma` | `0.99` | supports long-horizon reward |
| `max_position` | `1.0` | caps exposure |

---

# Expected Behaviour After Fix

## Reward
- mean reward > 0 (no longer flat at zero)
- reward responds to signal quality

## Policy
- non-zero positions
- reduced collapse to inactivity
- action correlates with signal

## Diagnostics
- entropy decreases but does not collapse
- action variance stabilizes at non-zero level

---

# Grid Search (Later)

| Parameter | Grid |
|---|---|
| `reward_scale` | `[3.0, 10.0, 30.0]` |
| `alpha_pos` | `[0.003, 0.01, 0.03]` |
| `beta_trade` | `[0.001, 0.005, 0.02]` |
| `target_horizon` | `[8, 16]` |

Run only after verifying Phase 1 works.

---

# Quantitative Goal

The next run should demonstrate:

1. **No collapse to zero action**
2. **Positive or clearly improved reward vs baseline**
3. **Visible correlation between action and signal**
4. **PPO approaching or matching linear baseline behaviour**

---

# Bottom Line

The previous RL formulation failed because:
- reward was too noisy and multi-objective
- signal was weak relative to penalties
- horizon alignment was poor
- exploration was discouraged

Phase 1 isolates the core skill:

> learn to take directional exposure based on a predictive signal

Only after that should more complex objectives be reintroduced.
