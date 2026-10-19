"""Tabular Q-learning for Gymnasium CliffWalking-v1."""

from dataclasses import dataclass

import gymnasium as gym
import numpy as np


@dataclass(frozen=True)
class Config:
    alpha: float = 0.5
    gamma: float = 0.99
    epsilon_initial: float = 1.0
    epsilon_min: float = 0.05
    epsilon_decay: float = 0.995
    episodes: int = 3000
    max_steps: int = 200


class ActionReplacement(gym.Wrapper):
    """Action error: probability p of choosing uniformly from the other 3 actions.

    Update Q for the requested action using the observed state and reward.
    Built-in slippery mode off.
    """

    def __init__(self, env, noise_p=0.0):
        super().__init__(env)
        if not 0.0 <= noise_p <= 1.0:
            raise ValueError("noise_p must lie in [0, 1]")
        self.noise_p = noise_p
        self.noise_rng = np.random.default_rng()

    def reset(self, *, seed=None, options=None, noise_seed=None):
        if noise_seed is not None:
            self.noise_rng = np.random.default_rng(noise_seed)
        return self.env.reset(seed=seed, options=options)

    def step(self, action):
        executed = int(action)
        if self.noise_p > 0 and self.noise_rng.random() < self.noise_p:
            # Offsets 1, 2, 3: equal chance for each other action.
            executed = (executed + int(self.noise_rng.integers(1, 4))) % 4
        return self.env.step(executed)


def make_env(noise_p=0.0, max_steps=200, render_mode=None):
    """Stop at max_steps; a time limit isn't a terminal state."""
    env = gym.make("CliffWalking-v1", is_slippery=False,
                   max_episode_steps=max_steps, render_mode=render_mode)
    return ActionReplacement(env, noise_p)


def update_q(q, state, action, reward, next_state, terminated, alpha, gamma):
    """Bootstrap at the time limit, not at the goal."""
    target = reward if terminated else reward + gamma * np.max(q[next_state])
    q[state, action] += alpha * (target - q[state, action])


def choose_action(q, state, epsilon, rng):
    if rng.random() < epsilon:
        return int(rng.integers(q.shape[1]))
    # Random ties so training doesn't favor action 0 (up) from the start.
    best = np.flatnonzero(q[state] == np.max(q[state]))
    return int(rng.choice(best))


def episode(env, q, reset_seed, noise_seed, *, config=None, epsilon=0.0,
            rng=None, frames=None):
    """One training or greedy eval episode."""
    state, _ = env.reset(seed=int(reset_seed), noise_seed=int(noise_seed))
    if frames is not None:
        frames.append(env.render())
    total_reward = 0.0
    terminated = truncated = False
    while not (terminated or truncated):
        if config is None:
            action = int(np.argmax(q[state]))  # deterministic, epsilon-free
        else:
            action = choose_action(q, state, epsilon, rng)
        next_state, reward, terminated, truncated, _ = env.step(action)
        if config is not None:
            update_q(q, state, action, reward, next_state, terminated,
                     config.alpha, config.gamma)
        total_reward += float(reward)
        state = next_state
        if frames is not None:
            frames.append(env.render())
    return {"reward": total_reward, "success": bool(terminated)}


def train(config, seed, noise_p=0.0):
    env = make_env(noise_p, config.max_steps)
    q = np.zeros((env.observation_space.n, env.action_space.n), dtype=np.float64)
    action_rng = np.random.default_rng(seed)
    reset_rng = np.random.default_rng(seed + 1000)
    noise_rng = np.random.default_rng(seed + 2000)
    rewards = []
    try:
        for index in range(config.episodes):
            epsilon = max(config.epsilon_min,
                          config.epsilon_initial * config.epsilon_decay ** index)
            row = episode(env, q, reset_rng.integers(0, 2**31),
                          noise_rng.integers(0, 2**31), config=config,
                          epsilon=epsilon, rng=action_rng)
            rewards.append(row["reward"])
    finally:
        env.close()
    return q, rewards


def evaluate(q, noise_p=0.0, max_steps=200, episodes=30):
    env = make_env(noise_p, max_steps)
    try:
        # Same episode seeds for every policy.
        return [episode(env, q, 700000 + i, 800000 + i) for i in range(episodes)]
    finally:
        env.close()
