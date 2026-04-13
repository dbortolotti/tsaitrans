## Analysis of RL Performance Issues (Experiment 5 & Phase 1)

[span_0](start_span)[span_1](start_span)[span_2](start_span)[span_3](start_span)Based on the provided training logs[span_0](end_span)[span_1](end_span)[span_2](end_span)[span_3](end_span), the Reinforcement Learning (RL) agent is consistently failing to outperform the analytical baseline. Below are the primary technical reasons for this failure.

---

### 1. Rapid Entropy Collapse
[span_4](start_span)[span_5](start_span)The policy is suffering from premature convergence, where it becomes overly confident in suboptimal actions[span_4](end_span)[span_5](end_span).
* **[span_6](start_span)Observation**: Entropy drops from **0.8956** at Iteration 0[span_6](end_span) [span_7](start_span)to a negative value (**-1.5811**) by Iteration 174[span_7](end_span).
* **[span_8](start_span)Warning Signs**: The logs show frequent `⚠ LOW-ENT` warnings starting as early as Iteration 34[span_8](end_span). 
* **[span_9](start_span)[span_10](start_span)Impact**: Negative entropy indicates the policy's standard deviation has collapsed, effectively stopping exploration and causing the agent to get stuck in a local optimum[span_9](end_span)[span_10](end_span).

### 2. Underperformance Relative to Linear Baseline
[span_11](start_span)The PPO agent is unable to replicate the success of a simple linear strategy, despite high correlation with the input signal[span_11](end_span).
* **[span_12](start_span)The Gap**: The baseline evaluation (k=1.0) shows a mean reward of **2.7244**[span_12](end_span). [span_13](start_span)In contrast, the PPO agent's best mean reward reached only **0.0031**[span_13](end_span).
* **[span_14](start_span)Correlation Paradox**: By Iteration 299, the agent achieved a high signal correlation of **0.8957**[span_14](end_span). [span_15](start_span)However, this "correct" directional alignment did not translate into meaningful PnL, suggesting the agent is taking positions that are too small or being eaten by penalties[span_15](end_span).

### 3. Conflicting Reward Structures and Horizons
[span_16](start_span)[span_17](start_span)The RL agent's objectives may be poorly aligned with the data it is receiving[span_16](end_span)[span_17](end_span).
* **[span_18](start_span)[span_19](start_span)Noisy Long-Term Signals**: The agent uses a high discount factor ($\gamma=0.97$ to $0.99$) to optimize long-term rewards[span_18](end_span)[span_19](end_span). [span_20](start_span)However, the underlying transformer signal for the $h=16$ horizon has an $R^2$ of only **0.1037**[span_20](end_span). [span_21](start_span)[span_22](start_span)The agent is effectively trying to optimize a long-term strategy based on very noisy predictions[span_21](end_span)[span_22](end_span).
* **[span_23](start_span)Aggressive Penalties**: The inclusion of trade penalties ($\beta_{trade} = 0.005$) and position penalties ($\alpha_{pos} = 0.01$)[span_23](end_span) [span_24](start_span)appears to discourage the agent from taking any significant risk, leading to the near-zero mean rewards[span_24](end_span).

### 4. Complexity Bottleneck in Phase 1A
[span_25](start_span)In the "full market making" mode, the agent's performance was significantly more unstable[span_25](end_span).
* **[span_26](start_span)Extreme Loss**: The initial mean reward was a massive **-114.29**[span_26](end_span). [span_27](start_span)While it improved to **-0.0124**[span_27](end_span)[span_28](start_span)[span_29](start_span), it never achieved positive territory, suggesting the action dimension (dim=2) and complex reward mode are currently too difficult for the policy to navigate[span_28](end_span)[span_29](end_span).

---

### Suggested Fixes
1.  **Increase Entropy Regularization**: Boost the entropy coefficient to force the agent to keep its standard deviation higher for longer.
2.  **[span_30](start_span)Reward Rescaling**: The current `reward_scale = 10.0`[span_30](end_span) may be interacting poorly with the penalties; consider normalizing the reward components.
3.  **[span_31](start_span)Ablation of Penalties**: Temporarily set $\alpha_{pos}$ and $\beta_{trade}$ to zero to verify the agent can learn the basic linear mapping before reintroducing transaction costs[span_31](end_span).
