# HANDOFF

This file summarizes the current state of the RL market-making work so a new
chat can resume quickly.

## Repo / Branch

- Repo: `/Users/oric/git/tsaitrans`
- Current branch: likely `codex/market-making-refresh`

## What Changed

### Environment / RL design

We redesigned the market-making setup around:

- aggressive action: `buy`, `sell`, or `none`
- passive bid toggle + passive offer toggle
- passive quote distances as part of the action space
- one-tick passive order lifetime
- hard inventory cap

Key files:

- [placing/market_env.py](/Users/oric/git/tsaitrans/placing/market_env.py)
- [RL_DESIGN.md](/Users/oric/git/tsaitrans/RL_DESIGN.md)
- [MARKET_ENV.md](/Users/oric/git/tsaitrans/MARKET_ENV.md)

### Current reward

Implemented reward in [placing/market_env.py](/Users/oric/git/tsaitrans/placing/market_env.py):

```text
reward_t
= realized_pnl_t
- inventory_penalty_t
+ alignment_reward_t
- trade_penalty_t
- terminal_cost_t
```

with:

```text
z_long_t = clip(mu_H(t) / sigma_H(t), -alignment_clip, alignment_clip)
alignment_reward_t = alignment_coef * position_after_t * half_spread * z_long_t
```

Current knobs in config:

- `kappa_base`
- `kappa_close`
- `lambda2`
- `alignment_coef`
- `alignment_clip`
- `trade_penalty`
- `max_position`

### Logging

Structured logging was added across the pipeline. RL logs include:

- `Iter`
- `MeanRew`
- `CumPnL`
- `Sharpe`
- `PolLoss`
- `VLoss`
- `Entropy`
- `KL`
- `|Act|`
- `|Pos|`
- `Corr`
- `LogStd`
- `RawRewStd`
- `RewStd`
- `Time`

Relevant files:

- [placing/train_rl.py](/Users/oric/git/tsaitrans/placing/train_rl.py)
- [run_experiment.py](/Users/oric/git/tsaitrans/run_experiment.py)

## Main Experiment

Primary experiment config:

- [experiments/mm_stable_reuse_latest_transformer.json](/Users/oric/git/tsaitrans/experiments/mm_stable_reuse_latest_transformer.json)

Important recent values used there:

- `half_spread = 0.001`
- `kappa_base` was tuned multiple times; latest main experiment file was adjusted during debugging
- `kappa_close = 0.0`
- `lambda2 = 1.5`
- `max_position` in the main experiment was still `3`

There were repeated single-run experiments, but the user shifted to a grid
search after the policy kept learning inventory-hugging behavior.

## Key Finding Before Grid Search

The policy was mechanically using transformer predictions, but behaviorally was
not learning a good selective market-making policy.

Typical issue:

- `Corr` improved with alignment reward
- but `|Pos|` stayed near `1.0`
- meaning the agent often sat near max inventory continuously

So reward shaping alone did not cleanly solve the problem.

## Grid Search Setup

We created a new config-driven grid search runner:

- [grid_search.py](/Users/oric/git/tsaitrans/grid_search.py)
- [experiments/mm_reward_grid.json](/Users/oric/git/tsaitrans/experiments/mm_reward_grid.json)

Important fixes made during setup:

1. `grid_search.py` now writes a per-run raw log:
   - `output/<run_name>/grid_run.log`
2. `grid_search.py` now launches `run_experiment.py` non-interactively using
   `stdin=subprocess.DEVNULL`
3. [run_experiment.py](/Users/oric/git/tsaitrans/run_experiment.py) was fixed so
   `base_experiment` reuses the base DGP / checkpoint while preserving explicit
   `data` overrides in the current config

### Final grid that was run

Sweep dimensions:

- `max_position`: `[3, 10]`
- `kappa_base`: `[1e-6, 5e-6, 2e-5]`
- `alignment_coef`: `[0.01, 0.02, 0.04]`
- `trade_penalty`: `[0.0, 0.0002]`

Fixed for the sweep:

- `kappa_close = 0.0`
- `alignment_clip = 3.0`
- `lambda2 = 1.5`
- `n_envs = 4`
- `n_iterations = 200`
- `rollout_steps = 1024`
- reduced RL split:
  - `stocks_rl_train = 12`
  - `stocks_rl_val = 4`

Total runs:

- `36`

## Grid Search Results

Master sweep log:

- [log_mm_reward_grid.txt](/Users/oric/git/tsaitrans/log_mm_reward_grid.txt)

Aggregate summary:

- [experiments/generated/mm_reward_grid/summary.json](/Users/oric/git/tsaitrans/experiments/generated/mm_reward_grid/summary.json)

The full 36-run sweep completed.

### Best run by training-tail metric

Best run according to the grid runner:

- `mm_reward_grid__max_position10__kappa_base5em06__alignment_coef0p04__trade_penalty0`

Path:

- [best run dir](/Users/oric/git/tsaitrans/output/mm_reward_grid__max_position10__kappa_base5em06__alignment_coef0p04__trade_penalty0)

Training-tail metrics:

- `tail_cum_pnl = 0.0225`
- `tail_sharpe = 5.63`
- `tail_corr = 0.5214`
- `tail_|pos| = 0.1202`

Interpretation:

- `max_position = 10` plus stronger alignment and moderate `kappa_base`
  produced far less inventory hugging than the `max_position = 3` regime
- `trade_penalty = 0.0002` generally did not help

## Fresh 50-Simulation Evaluation

The user then asked for **50 simulations** on the best grid result.

### Important simulator fix

[simulate.py](/Users/oric/git/tsaitrans/simulate.py) was stale and had to be
updated to:

- work with `base_experiment` runs
- use current observation dim `6 + 2H`
- use action dim `5`
- construct the current `MarketMakingEnv`
- export bid/offer prices from `bid_prices` / `offer_prices`

### 50 deterministic simulations run

Evaluated run:

- [output/mm_reward_grid__max_position10__kappa_base5em06__alignment_coef0p04__trade_penalty0](/Users/oric/git/tsaitrans/output/mm_reward_grid__max_position10__kappa_base5em06__alignment_coef0p04__trade_penalty0)

Saved summary:

- [sim_summary_50.json](/Users/oric/git/tsaitrans/output/mm_reward_grid__max_position10__kappa_base5em06__alignment_coef0p04__trade_penalty0/checkpoints_rl/sim_summary_50.json)

Results over 50 deterministic fresh sims:

- mean PnL: `-0.2361`
- median PnL: `-0.2278`
- PnL std: `1.1079`
- mean reward: `-0.5673`
- mean avg absolute position: `4.85`
- mean fills: `36.96`
- best PnL: `2.0668`
- worst PnL: `-4.3464`

### Important conclusion

This run looked best by the training-tail metric, but it is still **negative on
average on fresh simulations**.

So the next step should not be “declare success”. The next step should be one
of:

1. rank top grid runs by fresh simulation performance rather than training-tail
   metrics
2. run the same 50-sim evaluation on the next 2-3 best grid candidates
3. design a narrower second-round grid

## Suggested Next Step

Most sensible next action in a new chat:

1. read [experiments/generated/mm_reward_grid/summary.json](/Users/oric/git/tsaitrans/experiments/generated/mm_reward_grid/summary.json)
2. identify top 3-5 candidates by training-tail metrics
3. run 50 fresh sims on each
4. compare out-of-sample mean/median PnL and position usage

This is preferable to further tweaking the reward based only on training logs.

## Useful Artifacts

- [grid_search.py](/Users/oric/git/tsaitrans/grid_search.py)
- [simulate.py](/Users/oric/git/tsaitrans/simulate.py)
- [experiments/mm_reward_grid.json](/Users/oric/git/tsaitrans/experiments/mm_reward_grid.json)
- [experiments/generated/mm_reward_grid/summary.json](/Users/oric/git/tsaitrans/experiments/generated/mm_reward_grid/summary.json)
- [best run log](/Users/oric/git/tsaitrans/output/mm_reward_grid__max_position10__kappa_base5em06__alignment_coef0p04__trade_penalty0/grid_run.log)
- [50-sim summary](/Users/oric/git/tsaitrans/output/mm_reward_grid__max_position10__kappa_base5em06__alignment_coef0p04__trade_penalty0/checkpoints_rl/sim_summary_50.json)
