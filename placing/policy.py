"""
policy.py

Small MLP policy + value network for PPO, and the PPO update logic.
Designed to be lightweight enough for MPS on a Mac Mini M3/M4.
"""

import torch
import torch.nn as nn
import numpy as np
from torch.distributions import Normal, Categorical


class ActorCritic(nn.Module):
    """
    Shared-trunk MLP with separate actor (policy) and critic (value) heads.

    Actor outputs: mean of bid_offset, ask_offset (2D continuous)
    Log-std is a learnable parameter (state-independent).
    """

    def __init__(self, obs_dim=4, act_dim=2, hidden=64,
                 log_std_init=-0.5, log_std_min=-3.0, log_std_max=1.0):
        super().__init__()

        self.log_std_min = log_std_min
        self.log_std_max = log_std_max

        self.trunk = nn.Sequential(
            nn.Linear(obs_dim, hidden),
            nn.Tanh(),
            nn.Linear(hidden, hidden),
            nn.Tanh(),
        )

        self.actor_mean = nn.Linear(hidden, act_dim)
        self.actor_log_std = nn.Parameter(torch.full((act_dim,), log_std_init))

        self.critic = nn.Linear(hidden, 1)

        self._init_weights()

    def _init_weights(self):
        for m in self.trunk:
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
                nn.init.zeros_(m.bias)
        nn.init.orthogonal_(self.actor_mean.weight, gain=0.01)
        nn.init.zeros_(self.actor_mean.bias)
        nn.init.orthogonal_(self.critic.weight, gain=1.0)
        nn.init.zeros_(self.critic.bias)

    def forward(self, obs):
        h = self.trunk(obs)
        mean = self.actor_mean(h)
        # mean is raw (unsquashed) — tanh applied after sampling
        log_std = torch.clamp(self.actor_log_std, self.log_std_min, self.log_std_max)
        std = log_std.exp().expand_as(mean)
        value = self.critic(h).squeeze(-1)
        return mean, std, value

    def get_action(self, obs, deterministic=False):
        """For inference / rollout. Returns numpy arrays."""
        with torch.no_grad():
            mean, std, value = self.forward(obs)
            # Guard against NaN from gradient explosion
            if torch.isnan(mean).any() or torch.isinf(mean).any():
                mean = torch.nan_to_num(mean, nan=0.0, posinf=5.0, neginf=-5.0)
            if deterministic:
                action = torch.tanh(mean)
                log_prob = torch.zeros(mean.shape[0], device=mean.device)
            else:
                dist = Normal(mean, std)
                z = dist.sample()
                action = torch.tanh(z)
                # Tanh log-prob correction (Jacobian of the squashing)
                log_prob = (dist.log_prob(z) - torch.log(1.0 - action.pow(2) + 1e-6)).sum(-1)
        return action.cpu().numpy(), log_prob.cpu().numpy(), value.cpu().numpy()

    def evaluate(self, obs, actions):
        """For PPO update. Returns log_prob, entropy, value."""
        mean, std, value = self.forward(obs)
        # Guard against NaN from gradient explosion — clamp mean to finite range
        if torch.isnan(mean).any() or torch.isinf(mean).any():
            mean = torch.nan_to_num(mean, nan=0.0, posinf=5.0, neginf=-5.0)
        # Invert tanh to recover pre-squash z
        actions_c = actions.clamp(-0.999, 0.999)
        z = torch.atanh(actions_c)
        dist = Normal(mean, std)
        log_prob = (dist.log_prob(z) - torch.log(1.0 - actions_c.pow(2) + 1e-6)).sum(-1)
        entropy = dist.entropy().sum(-1)  # base Normal entropy (standard approx)
        return log_prob, entropy, value

    def actor_params(self):
        """Actor parameters (trunk + actor head + log_std)."""
        return list(self.trunk.parameters()) + list(self.actor_mean.parameters()) + [self.actor_log_std]

    def critic_params(self):
        """Critic parameters (critic head only)."""
        return list(self.critic.parameters())


class CategoricalActorCritic(nn.Module):
    """ActorCritic with discrete action space {-1, 0, +1} for Phase 1 Exp 4."""

    def __init__(self, obs_dim=4, n_actions=3, hidden=64):
        super().__init__()

        self.n_actions = n_actions

        self.trunk = nn.Sequential(
            nn.Linear(obs_dim, hidden),
            nn.Tanh(),
            nn.Linear(hidden, hidden),
            nn.Tanh(),
        )

        self.actor_logits = nn.Linear(hidden, n_actions)
        self.critic = nn.Linear(hidden, 1)

        # For compatibility with log_std logging
        self.actor_log_std = nn.Parameter(torch.zeros(1), requires_grad=False)

        self._init_weights()

    def _init_weights(self):
        for m in self.trunk:
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
                nn.init.zeros_(m.bias)
        nn.init.orthogonal_(self.actor_logits.weight, gain=0.01)
        nn.init.zeros_(self.actor_logits.bias)
        nn.init.orthogonal_(self.critic.weight, gain=1.0)
        nn.init.zeros_(self.critic.bias)

    def forward(self, obs):
        h = self.trunk(obs)
        logits = self.actor_logits(h)
        value = self.critic(h).squeeze(-1)
        return logits, value

    def get_action(self, obs, deterministic=False):
        with torch.no_grad():
            logits, value = self.forward(obs)
            dist = Categorical(logits=logits)
            if deterministic:
                action = logits.argmax(dim=-1)
            else:
                action = dist.sample()
            log_prob = dist.log_prob(action)
        return action.cpu().numpy(), log_prob.cpu().numpy(), value.cpu().numpy()

    def evaluate(self, obs, actions):
        logits, value = self.forward(obs)
        dist = Categorical(logits=logits)
        log_prob = dist.log_prob(actions.long().squeeze(-1))
        entropy = dist.entropy()
        return log_prob, entropy, value

    def actor_params(self):
        return list(self.trunk.parameters()) + list(self.actor_logits.parameters())

    def critic_params(self):
        return list(self.critic.parameters())


class RunningMeanStd:
    """Welford's online algorithm for running mean/variance."""

    def __init__(self, shape=()):
        self.mean = np.zeros(shape, dtype=np.float64)
        self.var = np.ones(shape, dtype=np.float64)
        self.count = 1e-4

    @property
    def std(self):
        return np.sqrt(self.var + 1e-8)

    def update(self, batch):
        batch = np.asarray(batch, dtype=np.float64)
        batch_mean = batch.mean(axis=0)
        batch_var = batch.var(axis=0)
        batch_count = batch.shape[0]
        self._update_from_moments(batch_mean, batch_var, batch_count)

    def _update_from_moments(self, batch_mean, batch_var, batch_count):
        delta = batch_mean - self.mean
        total = self.count + batch_count
        new_mean = self.mean + delta * batch_count / total
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        m2 = m_a + m_b + delta**2 * self.count * batch_count / total
        self.mean = new_mean
        self.var = m2 / total
        self.count = total


class RolloutBuffer:
    """Stores rollout data for PPO updates."""

    def __init__(self):
        self.obs = []
        self.actions = []
        self.log_probs = []
        self.rewards = []
        self.values = []
        self.dones = []

    def add(self, obs, action, log_prob, reward, value, done):
        self.obs.append(obs)
        self.actions.append(action)
        self.log_probs.append(log_prob)
        self.rewards.append(reward)
        self.values.append(value)
        self.dones.append(done)

    def compute_returns_and_advantages(self, last_values, gamma=0.99, lam=0.95):
        """GAE-Lambda advantage estimation."""
        n_steps = len(self.rewards)
        n_envs = self.rewards[0].shape[0]

        advantages = np.zeros((n_steps, n_envs), dtype=np.float64)
        returns = np.zeros((n_steps, n_envs), dtype=np.float64)
        last_gae = np.zeros(n_envs, dtype=np.float64)

        for t in reversed(range(n_steps)):
            if t == n_steps - 1:
                next_values = last_values
            else:
                next_values = self.values[t + 1]

            next_non_terminal = 1.0 - self.dones[t]
            delta = self.rewards[t] + gamma * next_values * next_non_terminal - self.values[t]
            last_gae = delta + gamma * lam * next_non_terminal * last_gae
            advantages[t] = last_gae
            returns[t] = advantages[t] + self.values[t]

        self.advantages = advantages
        self.returns = returns

    def get_batches(self, batch_size, device):
        """Yield minibatches for PPO update."""
        n_steps = len(self.obs)
        n_envs = self.obs[0].shape[0]

        # Flatten (steps, envs) -> (steps*envs,)
        obs = np.concatenate(self.obs, axis=0)          # (steps*envs, obs_dim)
        actions = np.concatenate(self.actions, axis=0)
        old_log_probs = np.concatenate(self.log_probs, axis=0)
        advantages = self.advantages.reshape(-1)
        returns = self.returns.reshape(-1)

        total = len(obs)
        indices = np.random.permutation(total)

        for start in range(0, total, batch_size):
            end = start + batch_size
            idx = indices[start:end]

            yield (
                torch.tensor(obs[idx], dtype=torch.float32, device=device),
                torch.tensor(actions[idx], dtype=torch.float32, device=device),
                torch.tensor(old_log_probs[idx], dtype=torch.float32, device=device),
                torch.tensor(advantages[idx], dtype=torch.float32, device=device),
                torch.tensor(returns[idx], dtype=torch.float32, device=device),
            )

    def clear(self):
        self.__init__()


def ppo_update(
    policy,
    optimizer: torch.optim.Optimizer,
    buffer: RolloutBuffer,
    device: torch.device,
    n_epochs: int = 10,
    batch_size: int = 512,
    clip_eps: float = 0.2,
    vf_coef: float = 0.5,
    ent_coef: float = 5e-4,
    max_grad_norm: float = 0.5,
    target_kl: float = 0.02,
    disable_reward_norm: bool = False,
):
    """Run PPO update on collected rollout data."""
    # Normalize advantages
    adv_flat = buffer.advantages.reshape(-1)
    adv_mean, adv_std = adv_flat.mean(), adv_flat.std() + 1e-8

    # Normalize value targets (returns)
    ret_flat = buffer.returns.reshape(-1)
    ret_mean, ret_std = ret_flat.mean(), ret_flat.std() + 1e-8

    metrics = {"policy_loss": 0, "value_loss": 0, "entropy": 0, "approx_kl": 0, "n_batches": 0}

    early_stopped = False
    for _ in range(n_epochs):
        for obs, actions, old_lp, advantages, returns in buffer.get_batches(batch_size, device):
            # Normalize advantages per minibatch
            advantages = (advantages - adv_mean) / adv_std

            log_prob, entropy, values = policy.evaluate(obs, actions)

            # Policy loss (clipped surrogate)
            ratio = (log_prob - old_lp).exp()
            surr1 = ratio * advantages
            surr2 = ratio.clamp(1.0 - clip_eps, 1.0 + clip_eps) * advantages
            policy_loss = -torch.min(surr1, surr2).mean()

            # Value loss (optionally normalized targets)
            if disable_reward_norm:
                value_loss = nn.functional.mse_loss(values, returns)
            else:
                returns_norm = (returns - ret_mean) / ret_std
                value_loss = nn.functional.mse_loss(values, returns_norm)

            # Entropy bonus
            entropy_loss = -entropy.mean()

            loss = policy_loss + vf_coef * value_loss + ent_coef * entropy_loss

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(policy.parameters(), max_grad_norm)
            optimizer.step()

            with torch.no_grad():
                approx_kl = (old_lp - log_prob).mean().item()

            metrics["policy_loss"] += policy_loss.item()
            metrics["value_loss"] += value_loss.item()
            metrics["entropy"] += entropy.mean().item()
            metrics["approx_kl"] += approx_kl
            metrics["n_batches"] += 1

            if target_kl is not None and approx_kl > 1.5 * target_kl:
                early_stopped = True
                break
        if early_stopped:
            break

    # Average
    n = max(metrics["n_batches"], 1)
    return {
        "policy_loss": metrics["policy_loss"] / n,
        "value_loss": metrics["value_loss"] / n,
        "entropy": metrics["entropy"] / n,
        "approx_kl": metrics["approx_kl"] / n,
        "early_stopped": early_stopped,
    }
