# Phase 1 RL Refactor Plan for Coding Agent

## Objective

Refactor the RL training setup so the policy first learns the simplest useful skill:

> map predictive signal -> sensible directional exposure

The current evidence suggests:
- the transformer signal is real and tradable
- PPO is stable enough
- the current reward encourages near-zero action / inactivity

So Phase 1 should:
1. simplify the reward
2. align reward with the signal horizon
3. add diagnostics and a baseline
4. avoid solving full market making immediately

---

# Scope

Do **not** change the transformer for this task.

Focus only on:
- RL environment reward
- RL configs
- baseline evaluation
- diagnostics
- optional warm-start hooks (only if cheap)

---

# Implementation Tasks

## 1. Add reward mode switch

Add a config field:
```python
reward_mode = "signal_exposure"
```

Supported values:
- `"signal_exposure"` -> new simplified Phase 1 reward
- `"full_market_making"` -> existing reward
- `"hybrid"` -> reserved for later

Default to:
```python
reward_mode = "signal_exposure"
```

---

## 2. Add target return definition aligned to signal horizon

Add config:
```python
target_horizon = 16
```

Define:
```python
target_return_t = mid[t + target_horizon] - mid[t]
```

Requirements:
- no leakage into the state
- reward only
- safe boundary handling near episode end
- no off-by-one mistakes

If the environment already has precomputed future returns, reuse them.

---

## 3. Implement Phase 1 reward

In `reward_mode == "signal_exposure"`, use:

```python
reward_t = reward_scale * position_t * target_return_t \
           - alpha_pos * position_t**2 \
           - beta_trade * (position_t - position_prev)**2
```

Where:
- `position_t` is the post-action position
- `position_prev` is previous position
- `target_return_t` is realized future return over chosen horizon

### Initial parameter values for Phase 1

Use these as the first run:

```python
reward_scale = 10.0
alpha_pos = 0.01
beta_trade = 0.005
gamma = 0.99
target_horizon = 16
max_position = 1.0
reward_mode = "signal_exposure"
```

Notes:
- these are starting values, not assumed optimum
- expose all of them through config
- print them clearly in logs

---

## 4. De-emphasize penalties that push inactivity

In `signal_exposure` mode:
- disable heavy inventory penalties beyond `alpha_pos * position^2`
- disable spread-capture reward terms
- disable complex execution / fill effects if they dominate learning
- keep reward focused on directional correctness + mild regularization

The point is to make the learning target dense and interpretable.

---

## 5. Add baseline sanity policy

Implement a deterministic baseline policy evaluated in the exact same environment and reward mode.

### Baseline policy
Use:
```python
position_t = clip(baseline_k * z_h16, -max_position, max_position)
```

Where:
```python
z_h16 = mu_h16 / (sigma_h16 + eps)
```

Initial value:
```python
baseline_k = 1.0
```

Requirements:
- evaluate baseline under the same environment
- report mean episode reward
- report turnover
- report mean absolute position
- report correlation with signal if easy

Purpose:
- verify the reward/environment actually rewards signal-following
- provide a lower bound PPO should approximate or beat

---

## 6. Add action / position diagnostics during PPO training

For each iteration, log:
- mean reward
- reward std
- mean absolute action
- mean absolute position
- turnover
- fraction of near-zero actions
- correlation(action, z_h16) if practical
- entropy
- KL
- log_std

Add a collapse warning if:
- mean absolute action trends toward zero
- entropy collapses
- reward stays near zero

---

## 7. Keep multi-horizon observations

Do not remove current multi-horizon state.

Continue passing:
- `mu_h` for all configured horizons
- `sigma_h` for all configured horizons

Optional but recommended derived features:
- `z_h = mu_h / (sigma_h + eps)`
- `mu_long - mu_short`

These can be added if simple.

---

## 8. Phase 1 experiment protocol

Run the following:

### Experiment A
Current PPO setup with old reward  
(for reference only)

### Experiment B
PPO with:
```python
reward_mode = "signal_exposure"
target_horizon = 16
reward_scale = 10.0
alpha_pos = 0.01
beta_trade = 0.005
gamma = 0.99
```

### Experiment C
Linear baseline in the same `signal_exposure` reward mode

For each experiment, report:
- mean reward
- reward std
- mean absolute action
- mean absolute position
- turnover
- best validation reward if available

---

## 9. Prepare for grid search later, but do not run it yet

After the first Phase 1 run, prepare a small grid over:

```python
reward_scale in [3.0, 10.0, 30.0]
alpha_pos in [0.003, 0.01, 0.03]
beta_trade in [0.001, 0.005, 0.02]
target_horizon in [8, 16]
```

But do **not** run the full grid now.
Run the single initial config first and inspect behaviour.

---

## 10. Success criteria for Phase 1

Phase 1 is successful if:
- PPO no longer collapses to near-zero action
- mean reward is clearly positive or at least materially above trivial inactivity
- actions correlate sensibly with signal sign / magnitude
- PPO is competitive with or improving toward the baseline

If these conditions fail, do not move to hybrid/full reward yet.

---

# File / Code Targets

Likely files to edit:
- `placing/market_env.py`
- `placing/train_rl.py`
- config file(s) or experiment config objects
- optional diagnostics / evaluation helper files

---

# Constraints

- keep changes minimal and localized
- do not rewrite the transformer
- do not add unnecessary abstractions
- do not silently mix reward modes
- ensure logs clearly show reward mode and all parameter values

---

# Deliverables

1. reward mode switch
2. `signal_exposure` reward implementation
3. aligned target horizon return
4. baseline linear/z-score evaluator
5. diagnostics for action collapse / signal usage
6. updated config with initial Phase 1 values
7. short summary of exact code changes
