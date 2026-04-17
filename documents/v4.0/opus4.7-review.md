# Opus 4.7 review — why the RL isn't converging

Verified against the code — the dominant issues, ranked:

## 1. Reward magnitude too small
`market_env.py:357-361`, `train_rl.py` default `reward_scale=10.0`

Per-step reward is `reward_scale · position · target_ret`. With `target_ret` std ≈ 2e-4 and positions ≤ 1, typical reward is ~1e-3 — below the PPO gradient noise floor. This is the root cause driving everything else.

## 2. Double normalization destabilizes small rewards
`train_rl.py:491-495`

After tiny raw rewards, `rewards / reward_rms.std` re-normalizes on a near-zero denominator. `RewStd` logs ~0.01 across hundreds of iterations — that second division amplifies noise.

## 3. Entropy collapse
`train_rl.py:314-315`, `policy.py:23`

`ent_coef_start=5e-4` annealing to `1e-5`, plus `log_std_min=-3.0` (std≈0.05), means the Gaussian collapses before the agent discovers the signal. LOW-ENT warnings confirm this.

## 4. Horizon mismatch
`market_env.py:307-312, 354`

Observation contains predictions at horizons `[1,2,4,8,16]`, but reward only uses `target_horizon` (default 16). Agent can't credit-assign between the signal channel it sees and the one that pays.

## 5. Quadratic penalties dominate
`market_env.py:359-360`, `alpha_pos=0.01`

With reward O(1e-3) and `alpha_pos · position²` up to 0.01, the penalty exceeds the alpha signal. Optimal policy becomes `position=0`.

## 6. Value function has nothing to fit
Normalized returns in `ppo_update` (`policy.py:241`) are essentially noise while (1) is unfixed; VLoss stays flat ~random baseline.

## Suggested fix order for Phase 1

1. Set `reward_scale` to ~1000–5000 (so per-step reward ≈ O(1)).
2. Disable reward RMS normalization (`rewards_norm = rewards`).
3. Zero `alpha_pos` and `beta_trade` first — verify signal learning in isolation.
4. Bump `ent_coef` to ~0.01–0.02 (no anneal), raise `log_std_min` to -1.0.
5. Reduce to a single horizon reward matching the observation, or aggregate horizons.

Fix (1) first; (2)–(6) likely resolve themselves once gradients have signal.
