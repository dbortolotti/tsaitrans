# Phase 1 RL Diagnosis

## Summary

PPO learns the right direction (action-signal correlation reaches 0.75-0.80) but fails to convert this into reward because three interlocking problems suppress the gradient signal and kill exploration before the policy can learn sizing.

Baseline reward: 4.0 per episode. PPO best: 0.0014 per step (~2.8 total). PPO never matches the trivial linear baseline.

---

## Problem 1: Reward Scale Is Too Small (THE KILLER)

### Evidence

- RewStd column reads 0.01 across all 300 iterations of Experiments B and C
- MeanRew hovers between -0.005 and +0.002 — noise-floor territory
- Baseline achieves 4.0 per episode; PPO never exceeds ~0.002 per step

### Root cause

Returns have std ~ 0.0002. A target return over 16 steps is roughly 0.0002 * sqrt(16) ~ 0.0008. With reward_scale = 10.0 and position ~0.3:

    signal_reward ~ 10 * 0.3 * 0.0008 = 0.0024 per step
    penalty       ~ 0.01 * 0.09       = 0.0009 per step
    net reward    ~ 0.001 per step

This is the same order of magnitude as noise. PPO cannot extract a useful gradient from rewards this small.

Additionally, in train_rl.py line 493:

    rewards_norm = rewards / reward_rms.std

The RunningMeanStd converges to a tiny denominator, creating an extra layer of unstable normalization on top of already-tiny rewards.

### Fix

Increase reward_scale to 1000-10000 (not 10). The reward needs to be O(1) per step for PPO to have a clear gradient. Alternatively, normalize returns inside the environment before applying reward_scale. Consider turning off reward normalization entirely for Phase 1, or using a much longer warm-up for the reward RMS — the reward_scale already handles scaling, the extra normalization adds instability and lag.

---

## Problem 2: Entropy Collapse Kills Exploration

### Evidence

- Entropy drops monotonically from ~0.9 to deeply negative values (-1.58 in Exp B)
- log_std hits the floor at log_std_min = -3.0 (std ~ 0.05) and stays clamped
- LOW-ENT warnings start at iteration ~35, before the policy has learned anything useful
- By iteration 100+, the policy is nearly deterministic with |action| ~ 0.2

### Root cause

ent_coef starts at 5e-4 and anneals toward 1e-5. This is far too weak. The entropy bonus is fighting against a gradient that consistently rewards reducing variance — because rewards are tiny and noisy, the safest thing PPO can do is reduce exploration. The policy becomes deterministic before it learns how much to bet.

### Fix

Increase ent_coef to 0.01-0.05 and do not anneal it during Phase 1. Set log_std_min = -1.0 (std ~ 0.37) to prevent premature collapse. You are trying to learn a basic directional mapping — you need sustained exploration, not convergence to a point estimate.

---

## Problem 3: Value Function Never Learns

### Evidence

- VLoss stays between 0.6 and 1.0 across all 300 iterations — essentially random prediction
- The critic never learns to predict returns

### Root cause

Partly caused by Problem 1 (tiny rewards mean the target values are noise-dominated). Partly caused by the return normalization in ppo_update:

    returns_norm = (returns - ret_mean) / ret_std

When rewards are ~1e-3, the GAE returns are also tiny, and normalizing them to zero-mean unit-variance inflates noise to the same scale as signal. The critic is fitting noise.

### Fix

Fixing Problem 1 (larger reward_scale) should largely resolve this. The critic needs a signal-to-noise ratio above 1 to learn anything meaningful.

---

## Secondary Issue: Low Environment Diversity

n_envs = 4 with rollout_steps = 2048 gives 8192 samples per iteration. But each env runs a full 2000-step episode, so you only see 4 stock paths per iteration out of 20 available. Gradient estimates are noisy due to low path diversity.

Consider n_envs = 8-16 for better gradient estimates per iteration.

---

## What the Logs Actually Show

The policy IS learning something — the Corr column reaches 0.75-0.80, meaning actions track the signal direction well. But the reward scale is so tiny relative to noise that PPO cannot convert directional correctness into reward improvement.

The baseline takes position = clip(z_h) which gives it ~0.33 average position. PPO's positions are stuck at ~0.2 with collapsed entropy. The policy knows WHERE to go but has not learned HOW MUCH to bet, and it cannot explore that dimension because entropy is dead.

---

## Recommended Action Plan

All three fixes should be applied together — they interact:

1. Set reward_scale = 3000-5000 (calibrate so per-step reward is O(1))
2. Set ent_coef = 0.02, do not anneal, set log_std_min = -1.0
3. Disable reward normalization (set rewards_norm = rewards) or use a fixed normalizer
4. Increase n_envs to 8-16
5. Re-run Experiment B and compare MeanRew trajectory against baseline

Success criteria remain the same as the action plan: PPO should match or beat the baseline reward of ~4.0 per episode, with mean absolute position comparable to baseline (~0.33) and sustained positive correlation with the signal.
