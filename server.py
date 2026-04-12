"""
server.py

Lightweight Flask server that serves the trading visualizer and runs
simulations on demand.

Usage:
    python server.py
    python server.py --port 5001

Then open http://localhost:8000 in a browser and pick an experiment from the dropdown.
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
    if EXPERIMENT_DIR is None:
        return jsonify({"error": "No experiment selected"}), 400
    seed = random.randint(1, 100000)
    result = run_simulation(EXPERIMENT_DIR, seed=seed, deterministic=True)
    return jsonify(result)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Serve trading visualizer with live simulation")
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()

    print(f"Serving on http://{args.host}:{args.port}")
    app.run(host=args.host, port=args.port, debug=False)
