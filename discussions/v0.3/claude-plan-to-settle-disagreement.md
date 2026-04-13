# Phase-1 RL Fix: Implementation Plan

## Goal

Resolve the core disagreement between the three reviews by running a small, ordered experimental ladder. Each rung tests one hypothesis and either unblocks the next rung or kills a candidate diagnosis. Stop as soon as PPO matches or beats the linear baseline (≈2.7–4.0 per episode).

All experiments live under `output/phase1_<expN>_<label>/` and are defined as JSON configs in `experiments/phase1/`.

---

## Shared Setup (applies to every experiment below)

Before running any experiment, the coding agent should:

1. **Create a `phase1` branch of configs** — copy `experiments/example.json` into `experiments/phase1/base.json` and strip it to only the RL block fields we intend to override.
2. **Add missing knobs** to `run_experiment.py:DEFAULTS` and to `placing/train_rl.py` if they don't already exist:
   - `rl.reward_mode` — string, one of `"stepwise_pnl"` (current), `"horizon_aligned"` (new), `"sign_times_future_return"` (new)
   - `rl.reward_horizon` — int, default 8
   - `rl.disable_reward_norm` — bool, default False
   - `rl.ent_coef_anneal` — bool, default True (so we can turn it off)
   - `rl.log_std_min` — float, default -3.0
   - `rl.ablate_penalties` — bool; when True, forces `alpha_pos = 0`, `beta_trade = 0`, any variance penalty = 0
   - `rl.action_space` — string, `"continuous"` or `"discrete3"`
   - `rl.use_uncertainty` — bool, default False; when True, policy receives `mu/sigma` from the transformer instead of `mu`
3. **Every added knob must have a single call site** — no sprinkled `if` statements. Concentrate the branches in `market_env.py` (reward + action space) and `train_rl.py` (ent schedule, normalization, observation wiring).
4. **Log schema must not regress** — keep the existing columns (Entropy, LOW-ENT warnings, Corr, VLoss, MeanRew, RewStd). Add two new columns: `MeanAbsPos` and `BaselineRew` (baseline evaluated once at the start, then held constant in the log for easy eyeballing).
5. **Deterministic seeds** — fix `seed = 0` across all Phase-1 experiments so trajectories are comparable.
6. **Success gate (all experiments)** — a run is "successful" if any of the last 20 iterations has `MeanRew >= 0.7 * BaselineRew` AND `MeanAbsPos >= 0.25`. Log this as a single boolean at the end of each run.

---

## Experiment Ladder

### Exp 1 — `phase1_exp1_ablate_penalties`

**Hypothesis tested:** penalties alone are suppressing the signal (all three reviews implicated this).

**Config diff vs current baseline:**
```json
"rl": {
  "ablate_penalties": true,
  "ent_coef": 0.02,
  "ent_coef_anneal": false,
  "log_std_min": -1.0,
  "n_envs": 8,
  "n_iterations": 200
}
```

**What the agent must implement:**
- Wire `ablate_penalties` through `market_env.py` so `alpha_pos`, `beta_trade`, and any variance penalty are forced to 0 when the flag is set — do NOT remove the code paths, just zero them.
- Wire `ent_coef_anneal=False` through the PPO training loop (it currently anneals toward 1e-5; when disabled, keep `ent_coef` at its initial value for the whole run).
- Wire `log_std_min` through `policy.py` (currently hardcoded to -3.0).

**Stop condition:** if success gate is hit, skip Exps 2–4 and go to Exp 5 (stress test). If not, proceed to Exp 2.

---

### Exp 2 — `phase1_exp2_reward_scale`

**Hypothesis tested:** Review 1's claim that the current reward is structurally fine but numerically drowned.

**Config diff (built on Exp 1):**
```json
"rl": {
  "reward_scale": 3000,
  "disable_reward_norm": true
}
```

**What the agent must implement:**
- `disable_reward_norm` should bypass `rewards_norm = rewards / reward_rms.std` in `train_rl.py` (the coding agent needs to locate this line; per the Claude review it is around `train_rl.py:493`). When True, feed raw rewards to the PPO update.
- Also disable the `(returns - ret_mean)/ret_std` step inside `ppo_update` when `disable_reward_norm` is True — the rationale from Review 1 is that both normalizations amplify noise when rewards are tiny.
- Keep everything from Exp 1 (penalties still ablated, entropy still boosted). This is an additive test.

**Decision:** If success gate hit → Exp 5. Else → Exp 3.

---

### Exp 3 — `phase1_exp3_horizon_aligned_reward`

**Hypothesis tested:** Review 3's claim that the reward is structurally misaligned with a multi-horizon signal.

**Config diff (built on Exp 1, NOT Exp 2 — reset `reward_scale` to 10 and re-enable normalization):**
```json
"rl": {
  "reward_mode": "horizon_aligned",
  "reward_horizon": 8,
  "reward_scale": 10,
  "disable_reward_norm": false
}
```

**What the agent must implement in `market_env.py`:**
- Add a `reward_mode == "horizon_aligned"` branch that computes, at step `t`:
  ```
  reward_t = action_t * (price_{t+H} - price_t) / price_t
  ```
  where `H = reward_horizon`. Use log-returns or simple returns — be consistent with the existing reward unit.
- At the end of the episode, pad the last `H` steps with zero reward (or truncate the episode by `H` steps — pick the simpler option and document it in a comment).
- Keep the discount factor `gamma` at its current value; this experiment tests whether a horizon-aligned *instantaneous* reward is enough, without separately re-tuning γ.

**Decision:** If success gate hit → Exp 5. Else → Exp 4.

---

### Exp 4 — `phase1_exp4_discrete_actions_plus_uncertainty`

**Hypothesis tested:** Reviews 3's remaining two recommendations — continuous actions collapse with weak signal, and the transformer's σ is being wasted.

**Config diff (built on Exp 3):**
```json
"rl": {
  "action_space": "discrete3",
  "use_uncertainty": true
}
```

**What the agent must implement:**
- **Discrete action space:** add a `discrete3` mode to `market_env.py` with actions `{-1, 0, +1}`. In `policy.py`, this requires a categorical head instead of the Gaussian head. Keep the existing continuous policy intact — gate on `action_space` at construction time.
- **Uncertainty in observation:** currently the policy receives `mu` from the transformer. When `use_uncertainty=True`, replace that feature with `mu / (sigma + 1e-8)`. This is a one-line change in whatever function builds the env observation vector from the transformer output.
- **Observation dimensionality must not change** — we are replacing `mu` with `mu/sigma`, not adding a channel. If the coding agent prefers to add `sigma` as an extra channel, they must update the policy input width accordingly; default to the replacement approach unless they find a reason not to.

**Decision:** If success gate hit → Exp 5. Else → escalate: the problem is elsewhere and we need a new diagnosis round.

---

### Exp 5 — `phase1_exp5_stress_reintroduce`

Only run this if one of Exps 1–4 succeeded. Purpose: figure out which of the removed-in-Exp-1 penalties we can bring back without breaking learning, so Phase 2 has a sensible starting point.

Run three sub-runs, each starting from the successful config:

- `5a_pos_penalty`: reintroduce `alpha_pos` at its original value, keep trade cost at 0.
- `5b_trade_cost`: reintroduce `beta_trade` at its original value, keep `alpha_pos` at 0.
- `5c_both`: reintroduce both.

Report which sub-runs still pass the success gate. No new knobs — this is purely a config sweep.

---

## Diagnostics to add to every run

Regardless of the reward mode, the training logger must emit per iteration:

- `MeanAbsPos` — mean absolute policy action across the rollout (the reviews agreed this is the tell-tale for "learned direction but not magnitude").
- `RewScaleObserved` — the empirical std of raw rewards in the rollout, **before** any normalization. This is the single number that settles Exp 1 vs Exp 3: if it's ~1e-3 under horizon-aligned reward, Review 1 was right; if it's ~1e-1 but PPO still fails, Review 3 was right.
- `SignalCorr` — already logged as `Corr`, keep it.
- `LogStd` — mean `log_std` over the rollout, so we can see if the entropy floor fix actually held.

Add these as plain extra columns in the existing log table; no new file formats.

---

## Deliverables the coding agent must produce

1. Five JSON configs under `experiments/phase1/`: `exp1_ablate_penalties.json`, `exp2_reward_scale.json`, `exp3_horizon_aligned.json`, `exp4_discrete_uncertainty.json`, `exp5a_pos_penalty.json`, `exp5b_trade_cost.json`, `exp5c_both.json`.
2. Code changes concentrated in:
   - `run_experiment.py` (defaults for new knobs)
   - `placing/market_env.py` (reward modes, penalty ablation, discrete action space)
   - `placing/policy.py` (`log_std_min` param, optional categorical head)
   - `placing/train_rl.py` (ent-coef anneal flag, reward-norm bypass, new log columns, observation uncertainty wiring)
3. A short `discussions/v0.3/phase1_results.md` with one table per experiment: final `MeanRew`, `BaselineRew`, `MeanAbsPos`, `LogStd`, `SignalCorr`, pass/fail against the success gate, and a one-line conclusion about which hypothesis the experiment supported or killed.
4. No changes to the transformer, data generation, or RL val/test splits — Phase 1 is strictly about the policy's reward and exploration surface.

---

## Guardrails for the coding agent

- **Do not refactor unrelated code.** If something looks ugly but isn't in the path of one of these experiments, leave it alone.
- **Do not change the action-to-order-size mapping** in `market_env.py` — only add a new branch for `discrete3`. Scaling/clipping of continuous actions must stay identical so Exps 1–3 are true A/B tests against the current behavior.
- **Do not change γ, GAE λ, learning rates, rollout length, or minibatch size** in any of Exps 1–4. Those are orthogonal to what we are testing and will confound results.
- **Run each experiment once, fixed seed.** Phase 1 is diagnostic, not a hyperparameter search. If Exp 1 produces borderline results, add a second seed for that one experiment only.
- **Commit after each passing experiment** on branch `claude/compare-agent-reviews-ZzBkT`, with a message of the form `phase1 exp<N>: <one-line result>`.
- **Do not open a PR** unless explicitly asked.

---

## Why this ordering

Exp 1 is the cheapest and is endorsed by all three reviews, so it is the correct first cut. Exps 2 and 3 isolate the Review 1 vs Review 3 disagreement: each adds exactly one intervention on top of Exp 1, so whichever one passes directly identifies the dominant cause. Exp 4 is the "Review 3 was half-right but needed more" fallback. Exp 5 exists because a Phase-1 win with penalties zeroed is not a Phase-2-ready configuration, and we need to know which penalty the policy can actually tolerate before Phase 2 starts layering execution cost back in.
