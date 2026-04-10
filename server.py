"""
server.py

Lightweight Flask server that serves the trading visualizer and runs
simulations on demand.

Usage:
    python server.py output/example
    python server.py output/example --port 5001

Then open http://localhost:5000 in a browser.
"""

import argparse
import os
import sys
import random

from flask import Flask, jsonify, request, send_file

# Add module dirs to path
repo_root = os.path.dirname(os.path.abspath(__file__))
for d in ["prediction", "placing"]:
    p = os.path.join(repo_root, d)
    if p not in sys.path:
        sys.path.insert(0, p)

from simulate import main as run_simulation

app = Flask(__name__)
EXPERIMENT_DIR = None
OUTPUT_DIR = os.path.join(repo_root, "output")


@app.route("/")
def index():
    return send_file(os.path.join(repo_root, "trading_visualizer.html"))


@app.route("/experiments")
def list_experiments():
    if not os.path.isdir(OUTPUT_DIR):
        return jsonify({"experiments": [], "current": None})
    exps = sorted(d for d in os.listdir(OUTPUT_DIR) if os.path.isdir(os.path.join(OUTPUT_DIR, d)))
    current = os.path.basename(EXPERIMENT_DIR) if EXPERIMENT_DIR else None
    return jsonify({"experiments": exps, "current": current})


@app.route("/simulate")
def simulate():
    global EXPERIMENT_DIR
    name = request.args.get("experiment")
    if name:
        EXPERIMENT_DIR = os.path.join(OUTPUT_DIR, name)
    seed = random.randint(1, 100000)
    result = run_simulation(EXPERIMENT_DIR, seed=seed, deterministic=True)
    return jsonify(result)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Serve trading visualizer with live simulation")
    parser.add_argument("experiment", type=str, nargs="?", help="Path to experiment output dir (e.g. output/example); defaults to first found in output/")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()

    if args.experiment:
        EXPERIMENT_DIR = args.experiment
    elif os.path.isdir(OUTPUT_DIR):
        found = sorted(d for d in os.listdir(OUTPUT_DIR) if os.path.isdir(os.path.join(OUTPUT_DIR, d)))
        EXPERIMENT_DIR = os.path.join(OUTPUT_DIR, found[0]) if found else None

    print(f"Serving experiment: {EXPERIMENT_DIR}")
    print(f"Open http://localhost:{args.port}")
    app.run(host="0.0.0.0", port=args.port, debug=False)
