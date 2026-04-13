# RL Diagnosis: Why PPO Fails to Match the Linear Baseline

## The Core Problem: PPO Can't See the Reward

**RewStd stays at 0.01 throughout the entire training run across all three experiments.** This is the smoking gun. The `RunningMeanStd` for reward normalization has initialized and then the actual reward magnitudes are so tiny that the normalizer's running std never budges from its initial state.

## The Arithmetic of the Reward Signal

In `SignalExposureEnv`, the reward is:

    reward = reward_scale * position * target_return - alpha_pos * position² - beta_trade * Δposition²

With `reward_scale=10`, `max_position=1.0`, and typical returns at std ≈ 0.0002 (normalization std), a single-step return is ~O(0.0002). Even over a 16-step horizon, target_return ≈ O(0.0002 × 4) ≈ O(0.0008). So the signal term per step is:

    10 * 1.0 * 0.0008 = 0.008

The penalty terms are:

    alpha_pos * 1² = 0.01
    beta_trade * 1² = 0.005

**The penalty terms are the same order as or larger than the maximum possible signal reward.** The optimal policy from PPO's perspective is to minimize the quadratic penalties, which means: do nothing. And that's exactly what it learns.

## What the Logs Confirm

### Experiment A (full_market_making, 300 iters)

- Position collapses from 0.32 → 0.005 by iter 80
- Entropy stays high (~1.0) — agent never commits to a strategy
- Reward goes from -85 → ~-0.03
- "Success" = doing nothing

### Experiment B (signal_exposure, 300 iters) — Most Revealing

- Entropy drops steadily from 0.89 → -1.58 (saturates at log_std_min = -3.0)
- Correlation rises nicely: 0.00 → 0.80
- But |Pos| is only ~0.22 and |Act| ~0.22
- **Mean reward plateaus at 0.001 vs baseline of 1.31**

The policy *learns the direction* (correlation 0.8 is excellent!) but the reward normalization crushes the actual magnitude. The reward_rms.std stays at 0.01 because the normalized rewards create a feedback loop: tiny raw rewards → tiny updates to RMS → perpetually tiny std.

### Experiment C (signal_exposure, 50 iters)

- Same pattern, shorter run
- Baseline is 4.0, PPO gets 0.0014

## Why the Baseline Crushes PPO

The linear baseline does `position = clip(k * z_h, -1, 1)` and gets reward ~1.3–4.0. It uses the raw z-score directly, taking large positions when the signal is strong. PPO can't do this because:

1. **The reward is too small relative to the penalties** — at reward_scale=10 and returns O(0.0002), the directional edge per step barely exceeds alpha_pos.
2. **The reward normalization (rewards / reward_rms.std) doesn't help** — it would help if reward scale were the issue, but the fundamental problem is that the signal-to-penalty ratio is <1. Normalizing doesn't change ratios.
3. **Entropy collapse confirms premature convergence** — LogStd hits -3.0 (the floor) while the policy is still suboptimal. The agent has become deterministic too early, locking in a mediocre strategy.

## Root Causes (Ranked by Severity)

### 1. Unit Mismatch Between Observations and Rewards (PRIMARY)

`SignalExposureEnv._target_return()` sums **raw returns** (O(0.0002)), but `mu_predictions` comes from the transformer trained on `(returns - mean) / std` with `std=0.0002`. The mu values are O(1) in normalized space.

**The observation and reward operate in completely different scales.**

The agent sees signals of O(0.3–0.8) in the obs but gets rewarded on realized returns of O(0.0008). This is why:
- Correlation is high (it learns direction from large-scale obs)
- Reward is negligible (the actual payoff is in a different unit)

`reward_scale=10` doesn't come close to bridging this gap. You'd need `reward_scale ≈ 10 / std ≈ 50,000` to make the signal term comparable to what the baseline sees.

### 2. Penalty Parameters Calibrated for Wrong Scale

`alpha_pos=0.01` and `beta_trade=0.005` are set as if the signal term produces O(1) rewards, but it produces O(0.001). These need to be proportional to `reward_scale * return_scale`.

### 3. gamma=0.99 with T=2000 Steps Is Borderline

The effective horizon is 1/(1-gamma) = 100 steps, so rewards beyond ~100 steps are heavily discounted. target_horizon is 16 so this isn't catastrophic, but the cumulative_reward term in the obs could become very large and noisy.

### 4. Entropy Annealing Too Fast

Annealing from 5e-4 to 1e-5 combined with log_std_min=-3.0 causes LogStd to hit the floor by iter ~70. Once hit, the policy is nearly deterministic and can't explore better strategies.

## Recommended Fixes

### Fix 1: Normalize target_return (Critical)

Make `_target_return` use normalized returns so the reward lives in the same space as the predictions:

    def _target_return(self, t):
        end = min(t + self.target_horizon, self.T)
        if end <= t:
            return 0.0
        return float(((self.returns[t:end] - self.norm_mean) / self.norm_std).sum())

Pass `norm_mean` and `norm_std` into the env constructor.

### Fix 2: Recalibrate Penalties

With normalized returns at O(1), the signal term becomes `reward_scale * position * O(4)` ≈ 40. Set penalties proportionally:

    alpha_pos  = 1.0   (was 0.01)
    beta_trade = 0.5   (was 0.005)

### Fix 3: Slower Entropy Schedule

    log_std_min = -2.0   (was -3.0)
    ent_coef    = 1e-3   (was 5e-4)
    ent_coef_end = 1e-4  (was 1e-5)

### Fix 4: Lower gamma (Optional)

    gamma = 0.97  (was 0.99, matches experiment4 base config anyway)

## Confidence Level

~90% confident the unit mismatch between obs and reward (issue #1) is the primary failure mode. The other issues (entropy schedule, gamma) are secondary but real. The log patterns — high correlation, near-zero reward, RewStd frozen at 0.01 — are exactly what you'd see when the policy learns the direction from correctly-scaled features but gets near-zero payoff from incorrectly-scaled returns.
