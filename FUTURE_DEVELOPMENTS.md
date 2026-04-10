# Future Developments

## Multi-step transformer predictions

Currently the transformer predicts one-step-ahead returns. After a fill, the agent holds inventory with zero expected return — the alpha is consumed in one step. This makes it unclear why the agent should hold rather than immediately flatten.

If the transformer predicted returns over a longer horizon (e.g. cumulative return over the next k steps), the agent would have a reason to hold: unrealized predicted alpha. This would:

- Make the inventory vs. alpha tradeoff meaningful — the agent holds because it expects further gains, not by accident.
- Create richer behavior — the agent could learn to build a position as alpha develops and unwind as it decays.

Requires changes to the transformer target (predict cumulative k-step return instead of 1-step), and potentially the observation space (include predicted horizon or remaining alpha).

## Kelly criterion inventory penalty

Replace the quadratic-from-flat penalty with a penalty that targets the Kelly-optimal position:

```
penalty = (p - p*(t))²
p*(t) = W(t) · prediction(t) / (σ_step² · price)
W(t) = initial_capital + cumulative_pnl
```

Key properties:
- **Directional** — penalizes deviation from optimal position, not from flat. Tells the agent *where to be*.
- **Wealth-dependent** — when winning, agent can take larger positions. When losing, must reduce. Natural risk management.
- **Signal-adaptive** — large prediction → large p* → agent encouraged to hold. No prediction → p* ≈ 0 → equivalent to penalizing from flat.

The main design question is `initial_capital`. It sets the scale of optimal positions, so it can't be arbitrary. A natural anchor is the spread: `initial_capital = k · half_spread`, where k = "how many round trips can I afford." This ties capital to market microstructure rather than an ad hoc number.

Tradeoff vs current design: Kelly adds two parameters (k, the framework itself) versus one (n) in the current time-varying quadratic. It also couples the penalty to the prediction signal, which the current design keeps separate. Consider once multi-step predictions are in place and the inventory-holding rationale is clearer.
