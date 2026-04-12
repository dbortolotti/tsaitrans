# Phase 2 — Hybrid Reward (Add Execution + Realism)

## Objective

Extend Phase 1 by introducing:
- execution effects
- spread capture
- more realistic inventory management

while preserving:
- the ability to exploit the predictive signal

---

# Core Reward

$begin:math:display$
r\_t \=
\\underbrace\{\\text\{spread\\\_pnl\}\_t\}\_\{\\text\{execution\}\}
\+
\\underbrace\{c \\cdot p\_t \\cdot R\_t\}\_\{\\text\{signal\}\}
\-
\\underbrace\{\\alpha p\_t\^2\}\_\{\\text\{inventory\}\}
\-
\\underbrace\{\\beta \(p\_t \- p\_\{t\-1\}\)\^2\}\_\{\\text\{turnover\}\}
$end:math:display$

---

# Components Explained

## 1. Signal term (unchanged from Phase 1)

$begin:math:display$
c \\cdot p\_t \\cdot R\_t
$end:math:display$

- keeps alignment with predictive model
- ensures policy does not “forget” signal-following

---

## 2. Spread / execution PnL

$begin:math:display$
\\text\{spread\\\_pnl\}\_t
$end:math:display$

Examples:
- earned spread when passively filled
- execution edge when crossing vs quoting

This term introduces:
- microstructure realism
- incentive to provide liquidity when appropriate

---

## 3. Inventory penalty (same form, slightly stronger)

$begin:math:display$
\\alpha p\_t\^2
$end:math:display$

Now more important because:
- inventory interacts with execution risk
- policy must manage exposure actively

---

## 4. Turnover penalty

$begin:math:display$
\\beta \(p\_t \- p\_\{t\-1\}\)\^2
$end:math:display$

Now represents:
- transaction costs
- overtrading penalty

---

# Key Difference vs Phase 1

Phase 1:
- clean directional learning
- no execution realism

Phase 2:
- introduces **conflicting objectives**
  - follow signal
  - manage inventory
  - capture spread
  - avoid overtrading

---

# New Free Parameters

| Parameter | Meaning | Role |
|---|---|---|
| `spread_weight` | scaling of spread pnl | balances MM vs directional |
| `alpha_pos` | inventory penalty | stronger than Phase 1 |
| `beta_trade` | turnover penalty | controls execution cost |
| `reward_scale` | signal weight | must remain significant |

---

# Critical Constraint

You must ensure:

$begin:math:display$
\\text\{signal term\} \\not\\ll \\text\{penalties \+ spread\}
$end:math:display$

Otherwise:
- policy reverts to inactivity again

---

# Suggested Initial Phase 2 Values

| Parameter | Value |
|---|---:|
| `reward_scale` | 10.0 |
| `alpha_pos` | 0.02 |
| `beta_trade` | 0.01 |
| `spread_weight` | 1.0 |

Key idea:
- increase realism gradually
- do not overwhelm signal term

---

# Expected Behaviour in Phase 2

Compared to Phase 1:

## Policy
- smaller positions
- more selective trading
- sensitivity to execution conditions

## Strategy
- mix of:
  - directional trades (signal-driven)
  - passive liquidity provision (spread-driven)

## Risk
- inventory controlled more actively
- fewer large exposures

---

# Diagnostics to Monitor

Track:

- mean reward
- signal term contribution
- spread term contribution
- average position
- turnover
- inventory distribution

Check:
- signal term is still meaningful
- spread term is not dominating

---

# Failure Modes

## 1. Collapse back to inactivity
Cause:
- penalties too strong
- spread too weak

## 2. Pure market making (ignoring signal)
Cause:
- spread term dominates signal term

## 3. Overtrading
Cause:
- beta too low

## 4. Inventory blow-up
Cause:
- alpha too low

---

# Success Criteria

Phase 2 is successful if:

- policy still correlates with signal
- reward remains positive
- spread term adds incremental value
- inventory is controlled without collapsing trading

---

# Transition Rule

Only move to Phase 2 if Phase 1 shows:

- non-zero action
- positive reward
- clear signal usage

Otherwise Phase 2 will fail.

---

# Summary

Phase 2 introduces:

- execution realism
- competing objectives
- risk management

while preserving:

- signal exploitation learned in Phase 1

The key is **balance**, not realism at all costs.
