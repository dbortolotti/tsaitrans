#!/bin/bash
# Experiment 3: Retrain transformer with probabilistic head + multi-horizon
# on the same data generated for experiment 2.
#
# Usage: bash run_experiment3.sh [--skip-rl]

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

EXP2_DIR="output/experiment2"
EXP3_DIR="output/experiment3"
DATA_FILE="$EXP2_DIR/data/returns_experiment2.npy"

if [ ! -f "$DATA_FILE" ]; then
    echo "ERROR: experiment2 data not found at $DATA_FILE"
    exit 1
fi

# Create output dir and symlink experiment2's data
mkdir -p "$EXP3_DIR/data"
for f in "$EXP2_DIR/data/"*; do
    base="$(basename "$f")"
    # Rename experiment2 -> experiment3 in symlink names
    target="$(echo "$base" | sed 's/experiment2/experiment3/g')"
    ln -sfn "$(cd "$EXP2_DIR/data" && pwd)/$base" "$EXP3_DIR/data/$target"
done

echo "Symlinked experiment2 data into $EXP3_DIR/data/"

# Run pipeline: skip data generation, train new transformer + backtest + optional RL
python run_experiment.py experiments/experiment3.json --skip-data "$@"
