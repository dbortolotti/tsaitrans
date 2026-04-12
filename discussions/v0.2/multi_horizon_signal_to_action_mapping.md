# Multi-Horizon Signal Interpretation for RL Trading

## Objective

Understand how the **term structure of predicted returns** (multi-horizon outputs from the transformer) should map to **optimal trading behaviour**.

The goal is to ensure the RL policy learns:
- position sizing
- timing
- holding duration
- risk management

from the **shape of the horizon curve**, not a single scalar signal.

---

# Model Output Structure

At each timestep $begin:math:text$ t $end:math:text$, the model outputs:

$begin:math:display$
\\\{ \(\\mu\_\{h\_1\}\, \\sigma\_\{h\_1\}\)\, \\ldots\, \(\\mu\_\{h\_H\}\, \\sigma\_\{h\_H\}\) \\\}
$end:math:display$

Where:
- $begin:math:text$ \\mu\_h $end:math:text$: expected return from $begin:math:text$ t \\to t\+h $end:math:text$
- $begin:math:text$ \\sigma\_h $end:math:text$: uncertainty of that prediction

This forms a **term structure of expected returns and uncertainty**.

---

# Derived Quantities (Key Features)

These are the most informative transformations:

## Short-term signal
$begin:math:display$
\\mu\_\{\\text\{short\}\} \= \\mu\_\{h\_1\}
$end:math:display$

## Long-term signal
$begin:math:display$
\\mu\_\{\\text\{long\}\} \= \\mu\_\{h\_H\}
$end:math:display$

## Slope (term structure)
$begin:math:display$
\\text\{slope\} \= \\mu\_\{\\text\{long\}\} \- \\mu\_\{\\text\{short\}\}
$end:math:display$

## Signal-to-noise ratio
$begin:math:display$
z\_h \= \\frac\{\\mu\_h\}\{\\sigma\_h \+ \\epsilon\}
$end:math:display$

---

# Behavioural Cases

## Case 1 — Strong, consistent signal

$begin:math:display$
\\mu\_\{\\text\{short\}\} \> 0\,\\quad \\mu\_\{\\text\{long\}\} \> 0\,\\quad \|\\mu\| \\gg \\sigma
$end:math:display$

**Interpretation:**
- clear directional edge
- high confidence
- persistent signal

**Optimal behaviour:**
- large position
- hold position
- low turnover

---

## Case 2 — Short-term signal only

$begin:math:display$
\\mu\_\{\\text\{short\}\} \> 0\,\\quad \\mu\_\{\\text\{long\}\} \\approx 0
$end:math:display$

**Interpretation:**
- transient move
- no persistence

**Optimal behaviour:**
- small position
- fast entry/exit
- avoid holding inventory

---

## Case 3 — Long-term drift

$begin:math:display$
\\mu\_\{\\text\{short\}\} \\approx 0\,\\quad \\mu\_\{\\text\{long\}\} \> 0
$end:math:display$

**Interpretation:**
- slow-moving trend
- noisy entry

**Optimal behaviour:**
- gradual position build
- tolerate noise
- moderate holding

---

## Case 4 — Reversal structure

$begin:math:display$
\\mu\_\{\\text\{short\}\} \> 0\,\\quad \\mu\_\{\\text\{long\}\} \< 0
$end:math:display$

**Interpretation:**
- short-term momentum
- long-term mean reversion

**Optimal behaviour:**
- avoid large positions
- consider fading
- high sensitivity to uncertainty

---

## Case 5 — High uncertainty

$begin:math:display$
\\sigma\_h \\gg \|\\mu\_h\|
$end:math:display$

**Interpretation:**
- weak or noisy signal

**Optimal behaviour:**
- no trading
- reduce inventory
- low activity

---

## Case 6 — Signal grows with horizon

$begin:math:display$
\|\\mu\_\{h\_1\}\| \< \|\\mu\_\{h\_2\}\| \< \\ldots \< \|\\mu\_\{h\_H\}\|
$end:math:display$

**Interpretation:**
- increasing conviction over time

**Optimal behaviour:**
- scale into position
- longer holding
- avoid reacting too early

---

## Case 7 — Signal decays with horizon

$begin:math:display$
\|\\mu\_\{h\_1\}\| \> \|\\mu\_\{h\_2\}\| \> \\ldots \> \|\\mu\_\{h\_H\}\|
$end:math:display$

**Interpretation:**
- short-lived effect

**Optimal behaviour:**
- short-duration trades
- high turnover
- minimal inventory

---

# Implications for RL Design

## 1. No single “optimal horizon”

The correct mapping is:

$begin:math:display$
f\(\\mu\_\{h\_1\}\, \.\.\.\, \\mu\_\{h\_H\}\, \\sigma\_\{h\_1\}\, \.\.\.\, \\sigma\_\{h\_H\}\) \\rightarrow \\text\{action\}
$end:math:display$

Not:

$begin:math:display$
h\^\* \\rightarrow \\text\{action\}
$end:math:display$

---

## 2. Horizon structure encodes behaviour

The multi-horizon output provides:
- timing (short vs long)
- persistence (slope)
- confidence (sigma)

This replaces the need to explicitly choose a horizon.

---

## 3. Why scalar collapse fails

Collapsing to a single value removes:
- slope information
- persistence
- timing differences

This prevents RL from distinguishing:
- transient vs persistent signals
- high vs low confidence
- entry vs hold conditions

---

# Recommended Features for RL Input

In addition to raw outputs, include:

- $begin:math:text$ \\mu\_\{\\text\{short\}\} $end:math:text$
- $begin:math:text$ \\mu\_\{\\text\{long\}\} $end:math:text$
- slope
- $begin:math:text$ z\_\{\\text\{long\}\} $end:math:text$
- optionally:
  - max $begin:math:text$ \|z\_h\| $end:math:text$
  - average $begin:math:text$ z\_h $end:math:text$

These reduce learning burden and improve stability.

---

# Key Insight

The model does not just output “direction”.

It outputs:
- when to act
- how long to hold
- how confident the signal is

The RL policy must convert this into:
- position size
- holding duration
- turnover

---

# Failure Mode

If RL fails after:
- multi-horizon inputs
- aligned reward

then the issue is likely:
- reward scaling
- signal strength
- or training instability

not architecture.

---

# Summary

- Multi-horizon outputs form a **term structure of expected returns**
- The shape of this structure determines optimal trading behaviour
- RL should learn a **mapping from structure to action**, not a fixed horizon
- Proper feature exposure is critical for learning efficiency
