# RL Design: Reward Function and Observation Space

## Observation Space (4D)

All prices scaled by initial price (start at 1.0). "Dollars" means scaled dollars throughout.

| # | Feature | Formula | Units |
|---|---|---|---|
| 1 | Predicted move | `prediction * price` | dollars |
| 2 | Position risk | `position * target_vol * price` | dollars |
| 3 | Time remaining | `(T - t) / T` | [0, 1] |
| 4 | Cumulative PnL | running total | dollars |

### Design rationale

- **No rolling vol** — DGP has constant conditional variance, no vol clustering to detect. Rolling vol is noise around `target_vol`.
- **No last return** — redundant with transformer prediction, which is trained on return history.
- **No inventory age** — redundant given position risk + time remaining.
- **No forward-looking normalization** — `return_scale = std(full series)` was a look-ahead bug. Use `target_vol` (known DGP parameter, analogous to historical daily vol) as the normalization constant.
- **Position as dollar risk** not raw units — `position * target_vol * price` is directly comparable with cumulative PnL and spread costs.

---

## Reward Function

### Per-step reward

```
reward(t) = mark_to_market_pnl(t) − (n · l(t) · position(t))²
```

- **Mark-to-market PnL:** `position_before · Δmid + fill_pnl` (unrealized change on existing position + realized PnL from any fills this step).
- **Inventory penalty:** Time-varying quadratic, calibrated from first principles (see below).

### Terminal penalty (last step only)

```
terminal_cost = λ2 · |position| · half_spread
```

- Models forced EOD liquidation: cross the spread + extra impact for urgency.
- `λ2 >= 1` — the 1x component is real spread cost, anything above captures market impact.

---

## Calibrating the inventory penalty

### Critical position

At time t with position p, ask: when does a 1σ adverse move on my inventory wipe out all expected remaining PnL?

- **1σ move on inventory:** `|p| · σ_step` (one-step dollar risk)
- **Expected remaining alpha:** `R² · σ_step · sqrt(T - t)` (assuming one trade per step, each capturing R² of per-step vol)

Setting them equal, `σ_step` cancels:

```
p_crit(t) = R² · sqrt(T - t)
```

This is the position at which a single 1σ move wipes out all expected future gains. Above `p_crit`, holding inventory is uncompensated risk.

### Defining l(t)

Set `l(t)` so that `l(t) · p = 1` at the critical position:

```
l(t) = 1 / (R² · sqrt(max(T - t, τ)))
```

`l(t)` increases as the day progresses — the critical position shrinks, so the penalty tightens.

**Floor `τ`:** Without it, `l(t) → ∞` as `t → T`, causing exploding penalty values that dwarf PnL signal and destabilize PPO training (exploding value function gradients, wildly varying reward scale across the episode). Clamping `T - t` to `τ` (e.g. 10–50 steps) keeps `l(t)` bounded in the final steps. The terminal penalty `λ2` handles the actual EOD flattening incentive. `τ` doesn't need careful tuning — it just prevents numerical blow-up.

### The penalty

```
penalty(t) = (n · l(t) · p)² = (n · p / (R² · sqrt(T - t)))²
```

Compare the linear and quadratic forms of `n · l · p`:
- They meet at `l(t) · p = 1/n`, i.e. at `p = p_crit / n`
- Below: quadratic is forgiving (small positions tolerated)
- Above: quadratic punishes harder than linear

### Parameters

| Parameter | Meaning | How to set |
|---|---|---|
| `R²` | Transformer prediction quality | Measured, not tuned |
| `n` | Risk tolerance in sigmas | Single tuning knob. Higher n = more conservative, penalty bites at smaller positions (`p_crit / n`) |
| `τ` | Penalty floor (steps) | 10–50. Prevents `l(t)` blow-up near EOD. Not sensitive. |
| `λ2` | EOD liquidation impact multiplier | `>= 1`. 1 = real spread cost, higher = extra impact |

`R²` is known from the trained transformer. `n` is the only free parameter controlling inventory penalty — interpretable as "how many sigmas of safety margin."

### Behavior

- **Early in day** (`T - t` large): `l(t)` small, penalty lenient, agent can hold larger positions — more time to capture remaining alpha.
- **Late in day** (`T - t` small): `l(t)` large, penalty steep, agent pressured to flatten — little alpha left to justify risk.
- **At `T - t ≤ τ`**: `l(t)` held constant at its maximum. Terminal penalty `λ2` handles the final flattening incentive.
- **Low R²**: `p_crit` small everywhere — penalty tight all day, agent stays nearly flat (correct: no alpha to justify holding).
- **High R²**: `p_crit` larger — agent has room to hold meaningful positions when alpha justifies it.

---

## Action Space (2D)

### Parameterization: width + skew

The network outputs two values in [-1, 1], mapped to:

- **Width** `w ∈ [0, max_width]` — symmetric distance from the skewed midpoint, in multiples of `half_spread`. Controls how tight/wide the quotes are.
- **Skew** `s ∈ [-max_skew, max_skew]` — directional shift of both quotes, in multiples of `half_spread`. Controls which side is aggressive.

Quote levels:

```
b(t) = mid(t) - w · half_spread + s · half_spread
o(t) = mid(t) + w · half_spread + s · half_spread
```

### Width interpretation

| Width | Meaning |
|---|---|
| > 1 | Wider than market — passive, may not fill |
| = 1 | At market bid/offer |
| (0, 1) | Inside the spread — tighter than market, still passive |
| = 0 | Both quotes at mid + skew |

Width is always >= 0. The agent cannot aggress both sides simultaneously.

### Skew interpretation

Skew shifts both quotes in the same direction. When `|skew|` is large enough, one side crosses the market (aggressive fill) while the other moves further away (unlikely to fill).

Example (`half_spread = 0.02`, `mid = 100`, `w = 0.5`, `s = 1.5`):

```
b(t) = 100 - 0.5·0.02 + 1.5·0.02 = 100.02 = om(t)  → aggressive buy
o(t) = 100 + 0.5·0.02 + 1.5·0.02 = 100.04            → passive, above market offer
```

Skew pushes both quotes up: bid crosses market offer (buys aggressively), offer moves further away (won't sell).

### Design rationale

- **Width >= 0** ensures the agent can't aggress both sides in the same step. Directional aggression comes from skew only.
- **Width and skew are independent decisions**: "how aggressive am I" vs "which direction do I lean." Easier for the network to learn than independent bid/ask offsets.
- **Units in `half_spread` multiples** — natural scale. Width = 1 means quoting at the market, directly interpretable.
- **`max_width` and `max_skew`** are env parameters that cap how passive or directional the agent can be.
