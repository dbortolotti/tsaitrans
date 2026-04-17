# Agent instructions: run the RL convergence A/B experiment

## Goal

Run a baseline vs. "fixed" RL training comparison to validate whether the
five Opus 4.7 review fixes (see `documents/v4.0/opus4.7-review.md`) actually
unblock convergence. Produce training logs for both runs, then compare.

All configs already exist — do **not** re-create them. Your job is to run the
pipeline, collect logs, and summarize.

## Prerequisites

1. Repository: `/home/user/tsaitrans` (or wherever it's cloned). `cd` there.
2. Checkout the right branch:
   ```
   git fetch origin
   git checkout claude/debug-rl-convergence-MwU7x
   git pull
   ```
3. Install dependencies (torch is large — expect 1–3 min):
   ```
   pip install torch numpy matplotlib gymnasium
   ```
4. Verify imports work:
   ```
   python3 -c "import torch, gymnasium, numpy, matplotlib; print(torch.__version__)"
   ```

## Configs (already committed)

- `experiments/ab_base.json` — reduced-scale data + transformer (CPU-friendly).
- `experiments/ab_baseline.json` — RL with current defaults (expected: no convergence).
- `experiments/ab_fixed.json` — RL with all five fixes applied.

The two RL configs set `base_experiment: ab_base`, so they reuse the same
data and transformer checkpoint — only RL hyperparameters differ.

## Run sequence

Create a logs directory and run each stage with `nohup`, capturing stdout+stderr:

```
mkdir -p logs/v4.0

# Stage 1: data + transformer (one time, ~5-15 min on CPU)
nohup python3 -u run_experiment.py experiments/ab_base.json \
  > logs/v4.0/ab_base.log 2>&1 &
echo $! > logs/v4.0/ab_base.pid

# Wait for stage 1 to finish before running RL. Poll the log:
tail -f logs/v4.0/ab_base.log
# Exit tail (Ctrl-C) once you see "Experiment 'ab_base' complete."
```

Confirm stage 1 finished cleanly:
```
grep -E "Experiment 'ab_base' complete|Traceback|Error" logs/v4.0/ab_base.log | tail -5
ls output/ab_base/checkpoints/best_model.pt   # must exist
```

Then launch both RL runs **in parallel** (they're independent):

```
nohup python3 -u run_experiment.py experiments/ab_baseline.json \
  > logs/v4.0/ab_baseline.log 2>&1 &
echo $! > logs/v4.0/ab_baseline.pid

nohup python3 -u run_experiment.py experiments/ab_fixed.json \
  > logs/v4.0/ab_fixed.log 2>&1 &
echo $! > logs/v4.0/ab_fixed.pid
```

Notes:
- `python3 -u` disables output buffering — needed for live log tailing.
- The `.pid` files let you kill a run with `kill $(cat logs/v4.0/ab_fixed.pid)` if needed.
- `run_experiment.py` prompts "overwrite?" if the output folder exists. Running via
  nohup is non-interactive, so it auto-proceeds (see `run_experiment.py:131-137`).

## Monitoring

Watch progress live (runs print one line per PPO iteration):
```
tail -f logs/v4.0/ab_fixed.log
```

Key columns to watch on each iteration row:
- `MeanRew`   — should grow positive if the policy is learning.
- `|Pos|`     — mean absolute position. Should move above ~0.2, not stay near 0.
- `Corr`      — correlation between action and the predictive signal. Should go positive.
- `Entropy`   — should stay above ~0.1 (LOW-ENT flag fires below).
- `LogStd`    — policy std floor. In the fixed run should sit around -1.0, not collapse.

Verify both processes finished:
```
grep -E "complete|Traceback|Error" logs/v4.0/ab_baseline.log logs/v4.0/ab_fixed.log | tail -10
```

## Summarize

Extract the last 10 iterations of each run:
```
grep -E "^\s*[0-9]+\s" logs/v4.0/ab_baseline.log | tail -10
grep -E "^\s*[0-9]+\s" logs/v4.0/ab_fixed.log    | tail -10
```

Write a short markdown summary to `documents/v4.0/run-results.md` covering:
- Final `MeanRew`, `|Pos|`, `Corr` for each run.
- Whether the fixed run shows the expected behavior (positive Corr, |Pos| > 0.2,
  Entropy > 0.1, sustained LogStd).
- Any surprises (e.g. a fix that over-corrected, new instability).

Commit and push to the current branch:
```
git add logs/v4.0/ documents/v4.0/run-results.md
git commit -m "Add A/B run logs + summary for v4.0 RL fixes"
git push origin claude/debug-rl-convergence-MwU7x
```

## Failure modes to watch for

- `ModuleNotFoundError: torch` — dependencies not installed; redo prerequisites.
- `No such file or directory: output/ab_base/...` — baseline/fixed launched before
  ab_base finished. Re-run once stage 1 completes.
- `CUDA`/`MPS` errors — CPU fallback is automatic via `get_device()`, but verify
  the device line in the log.
- Runs that finish in seconds — likely `n_iterations` was overridden to a tiny
  value. Confirm configs match what's in git.
- `output/<name>` already exists prompt blocks non-interactive run — delete the
  folder first: `rm -rf output/ab_fixed`.

## Scope

Do not modify code, configs, or the review documents. Only: install, run, log,
and summarize. If you discover a bug that prevents a run from completing, stop
and report it rather than patching silently.
