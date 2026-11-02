from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .env import OBS_NAMES


class FixedPolicy:
    def __init__(self, level: int, name: str | None = None):
        self.level = int(level)
        self.name = name or f"fixed-{level}"

    def act(self, obs) -> int:
        return self.level

_GRID = {"active": [1e-6, 0.26, 0.63], "queue": [1e-6, 0.13], "load": [0.10, 0.275]}


class QLearningAgent:

    name = "qlearning"

    def __init__(self, n_actions: int, grid: dict[str, list[float]] | None = None):
        self.grid = grid or _GRID
        self.dims = [OBS_NAMES.index(k) for k in self.grid]
        self.edges = [np.asarray(v, dtype=float) for v in self.grid.values()]
        self.shape = tuple(len(e) + 1 for e in self.edges)
        self.q = np.zeros((int(np.prod(self.shape)), n_actions))
        self.visits = np.zeros_like(self.q, dtype=np.int64)

    def state(self, obs) -> int:
        idx = [int(np.digitize(obs[d], e)) for d, e in zip(self.dims, self.edges)]
        return int(np.ravel_multi_index(idx, self.shape))

    def act(self, obs, epsilon: float = 0.0, rng: np.random.Generator | None = None) -> int:
        if rng is not None and rng.random() < epsilon:
            return int(rng.integers(self.q.shape[1]))
        row = self.q[self.state(obs)]
        return int(len(row) - 1 - np.argmax(row[::-1]))

    def update(self, obs, action: int, reward: float, next_obs, alpha: float, gamma: float) -> None:
        s, s2 = self.state(obs), self.state(next_obs)
        self.visits[s, action] += 1
        step = max(alpha, self.visits[s, action] ** -0.5)
        self.q[s, action] += step * (reward + gamma * self.q[s2].max() - self.q[s, action])

    def save(self, path: str | Path) -> None:
        data = {"grid": self.grid, "q": self.q.tolist(), "visits": self.visits.tolist()}
        Path(path).write_text(json.dumps(data), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "QLearningAgent":
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        q = np.asarray(data["q"], dtype=float)
        agent = cls(q.shape[1], data["grid"])
        agent.q, agent.visits = q, np.asarray(data["visits"], dtype=np.int64)
        return agent


class SB3Policy:

    def __init__(self, model, name: str):
        self.model, self.name = model, name

    def act(self, obs) -> int:
        action, _ = self.model.predict(obs, deterministic=True)
        return int(action)


REWARD_SCALE = 0.1

SB3_KWARGS = {
    "dqn": dict(
        learning_rate=5e-4, buffer_size=100_000, learning_starts=2_000, batch_size=128,
        gamma=0.95, train_freq=1, gradient_steps=1, target_update_interval=1_000,
        exploration_fraction=0.4, exploration_final_eps=0.03, policy_kwargs=dict(net_arch=[64, 64]),
    ),
    "ppo": dict(
        learning_rate=3e-4, n_steps=2048, batch_size=128, n_epochs=10, gamma=0.95,
        gae_lambda=0.95, ent_coef=0.01, policy_kwargs=dict(net_arch=[64, 64]),
    ),
}


def make_sb3(algo: str, venv, steps: int, seed: int = 0):
    from stable_baselines3 import DQN, PPO

    kwargs = dict(SB3_KWARGS[algo])
    if algo == "dqn":  
        kwargs["learning_starts"] = int(min(kwargs["learning_starts"], max(100, steps // 20)))
    else:  
        per_env = kwargs["n_steps"] // venv.num_envs
        kwargs["n_steps"] = int(min(per_env, max(64, steps // (4 * venv.num_envs) // 64 * 64)))
        if (kwargs["n_steps"] * venv.num_envs) % kwargs["batch_size"]:
            kwargs["batch_size"] = 64
    return {"dqn": DQN, "ppo": PPO}[algo]("MlpPolicy", venv, seed=seed, device="cpu", verbose=0, **kwargs)


def load_policy(spec: str, n_levels: int):
    kind, _, arg = spec.partition(":")
    if kind == "default":
        return FixedPolicy(n_levels - 1, "default")
    if kind == "fixed":
        return FixedPolicy(int(arg))
    if kind == "qlearning":
        return QLearningAgent.load(arg)
    if kind in ("dqn", "ppo"):
        from stable_baselines3 import DQN, PPO

        return SB3Policy({"dqn": DQN, "ppo": PPO}[kind].load(arg, device="cpu"), kind)
    raise ValueError(f"Unknown policy spec: {spec!r}")
