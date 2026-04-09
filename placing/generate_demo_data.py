"""
generate_demo_data.py

Generates a demo sim_results.json so the visualizer can be tested
without training the full RL pipeline first.
"""

import json
import numpy as np


def generate():
    rng = np.random.default_rng(42)
    T = 400
    
    # Generate a realistic-looking price path
    returns = rng.normal(0.0002, 0.015, T)
    prices = [100.0]
    for r in returns:
        prices.append(prices[-1] * (1.0 + r))
    
    mid_prices = prices[1:]  # T prices
    
    # Generate bid/ask around mid
    vol = 0.015
    bids = [p - rng.uniform(0.05, 0.3) for p in mid_prices]
    asks = [p + rng.uniform(0.05, 0.3) for p in mid_prices]
    
    # Generate positions that look like market-making
    position = 0
    positions = []
    fills_buy_t, fills_buy_p = [], []
    fills_sell_t, fills_sell_p = [], []
    pnls = []
    
    for t in range(T):
        # Random fills
        if rng.random() < 0.15 and position < 5:
            position += 1
            fills_buy_t.append(t)
            fills_buy_p.append(bids[t])
        if rng.random() < 0.15 and position > -5:
            position -= 1
            fills_sell_t.append(t)
            fills_sell_p.append(asks[t])
        
        positions.append(position)
        pnl = rng.normal(0.002, 0.05)
        pnls.append(pnl)
    
    cum_pnl = np.cumsum(pnls).tolist()
    
    result = {
        "summary": {
            "total_pnl": cum_pnl[-1],
            "sharpe": 1.8,
            "max_drawdown": -0.5,
            "n_buys": len(fills_buy_t),
            "n_sells": len(fills_sell_t),
            "n_total_fills": len(fills_buy_t) + len(fills_sell_t),
            "avg_abs_position": float(np.mean(np.abs(positions))),
            "total_reward": cum_pnl[-1] * 0.9,
            "n_steps": T,
        },
        "times": list(range(T)),
        "mid_prices": [float(p) for p in mid_prices],
        "bids": [float(b) for b in bids],
        "asks": [float(a) for a in asks],
        "positions": positions,
        "pnls": pnls,
        "cum_pnl": cum_pnl,
        "rewards": [p * 0.9 for p in pnls],
        "fills_buy_t": fills_buy_t,
        "fills_buy_p": [float(p) for p in fills_buy_p],
        "fills_sell_t": fills_sell_t,
        "fills_sell_p": [float(p) for p in fills_sell_p],
        "full_prices": [float(p) for p in prices],
    }
    
    with open("sim_results.json", "w") as f:
        json.dump(result, f)
    
    print("Generated sim_results.json with demo data")


if __name__ == "__main__":
    generate()
