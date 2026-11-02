from __future__ import annotations

import csv
import json
import time
from pathlib import Path

import numpy as np

from .agents import REWARD_SCALE, QLearningAgent, SB3Policy, make_sb3
from .backends import SimBackend
from .config import Config
from .env import LLMServingEnv
from .profile import HardwareProfile

CURVE_FIELDS = ["step", "episode", "return", "energy_j", "j_per_token", "slo_both", "mean_level", "wall_s"]
EVAL_SEED0 = 1_000_000
VALID_SEED0 = 2_000_000


def make_sim_env(profile: HardwareProfile, cfg: Config, seed: int = 0, drain: bool = False) -> LLMServingEnv:
    return LLMServingEnv(SimBackend(profile, cfg.sim_noise, seed), profile, cfg, drain=drain)


class CurveLog:

    def __init__(self, path: str | Path | None):
        self.path = Path(path) if path else None
        self.rows: list[dict] = []
        self.t0 = time.perf_counter()
        if self.path:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.open("w", newline="", encoding="utf-8") as f:
                csv.DictWriter(f, CURVE_FIELDS).writeheader()

    def add(self, step: int, metrics: dict) -> None:
        row = {k: metrics.get(k) for k in CURVE_FIELDS}
        row.update(step=step, episode=len(self.rows) + 1, wall_s=round(time.perf_counter() - self.t0, 1))
        self.rows.append(row)
        if self.path:
            with self.path.open("a", newline="", encoding="utf-8") as f:
                csv.DictWriter(f, CURVE_FIELDS).writerow(row)


def validation_return(env: LLMServingEnv, policy, episodes: int = 4) -> float:
    total = 0.0
    for i in range(episodes):
        obs, _ = env.reset(options={"trace_seed": VALID_SEED0 + i})
        while True:
            obs, _, terminated, truncated, info = env.step(policy.act(obs))
            if terminated or truncated:
                total += info["metrics"]["return"]
                break
    return total / episodes


def train_qlearning(
    env: LLMServingEnv,
    steps: int,
    seed: int = 0,
    agent: QLearningAgent | None = None,
    alpha: float = 0.005,
    gamma: float = 0.9,
    eps_start: float = 1.0,
    eps_end: float = 0.05,
    eps_fraction: float = 0.6,
    curve: str | Path | None = None,
    progress=None,
    validate=None,
    validate_every: int = 20_000,
) -> QLearningAgent:
    rng = np.random.default_rng(seed)
    agent = agent or QLearningAgent(env.action_space.n)
    log = CurveLog(curve)
    best_score, best_q = -np.inf, None
    obs, _ = env.reset(seed=seed)
    for step in range(1, steps + 1):
        eps = eps_start + (eps_end - eps_start) * min(1.0, step / max(1.0, eps_fraction * steps))
        action = agent.act(obs, eps, rng)
        next_obs, reward, terminated, truncated, info = env.step(action)
        agent.update(obs, action, reward, next_obs, alpha, gamma)
        obs = next_obs
        if terminated or truncated:
            log.add(step, info["metrics"])
            if progress:
                progress(step, steps, info["metrics"])
            obs, _ = env.reset()
        if validate and (step % validate_every == 0 or step == steps):
            score = validate(agent)
            if score > best_score:
                best_score, best_q = score, agent.q.copy()
    if best_q is not None:
        agent.q = best_q
    return agent


def train_sb3(
    algo: str,
    envs,
    steps: int,
    seed: int = 0,
    model=None,
    curve: str | Path | None = None,
    progress=None,
    validate=None,
    validate_every: int = 10_000,
):
    import copy

    import torch
    from stable_baselines3.common.callbacks import BaseCallback
    from stable_baselines3.common.vec_env import DummyVecEnv

    envs = list(envs) if isinstance(envs, (list, tuple)) else [envs]
    venv = DummyVecEnv([lambda e=e: e for e in envs])
    log = CurveLog(curve)

    best = {"score": -np.inf, "weights": None, "next": validate_every}

    def checkpoint(model) -> None:
        score = validate(SB3Policy(model, algo))
        if score > best["score"]:
            best["score"], best["weights"] = score, copy.deepcopy(model.policy.state_dict())

    class _Episodes(BaseCallback):
        def _on_step(self) -> bool:
            for info in self.locals.get("infos", []):
                if "metrics" in info:
                    log.add(self.num_timesteps, info["metrics"])
                    if progress:
                        progress(self.num_timesteps - start, steps, info["metrics"])
            if validate and self.num_timesteps - start >= best["next"]:
                best["next"] += validate_every
                checkpoint(self.model)
            return True

    if model is None:
        model = make_sb3(algo, venv, steps, seed)
    elif model.n_envs != venv.num_envs:  
        fresh = make_sb3(algo, venv, steps, seed)
        fresh.policy.load_state_dict(model.policy.state_dict())
        fresh.num_timesteps = model.num_timesteps
        model = fresh
    else:
        model.set_env(venv)
    start = model.num_timesteps
    threads = torch.get_num_threads()
    torch.set_num_threads(1) 
    for env in envs:
        env.reward_scale = REWARD_SCALE
    try:
        model.learn(total_timesteps=steps, callback=_Episodes(), reset_num_timesteps=False)
        if validate:
            checkpoint(model)
            model.policy.load_state_dict(best["weights"])
    finally:
        torch.set_num_threads(threads)
        for env in envs:
            env.reward_scale = 1.0
    return model


def run_episode(env: LLMServingEnv, policy, trace_seed: int) -> dict:
    obs, _ = env.reset(options={"trace_seed": trace_seed})
    while True:
        obs, _, terminated, truncated, info = env.step(policy.act(obs))
        if terminated or truncated:
            return {"metrics": info["metrics"], "history": env.history, "timeline": env.timeline}


def evaluate(env: LLMServingEnv, policies: dict, episodes: int = 5, between=None, progress=None) -> dict:
    env.drain = True
    names = list(policies)
    runs: dict[str, list[dict]] = {n: [] for n in names}
    first: dict[str, dict] = {}
    for ep in range(episodes):
        for name in names if ep % 2 == 0 else names[::-1]:
            if between:
                between()
            out = run_episode(env, policies[name], EVAL_SEED0 + ep)
            runs[name].append(out["metrics"])
            if ep == 0:
                first[name] = {"history": out["history"], "timeline": out["timeline"]}
            if progress:
                progress(name, ep, out["metrics"])
    result = {"episodes": episodes, "policies": {}, "first_episode": first}
    for name in names:
        keys = runs[name][0].keys()
        result["policies"][name] = {
            "runs": runs[name],
            "mean": {k: float(np.nanmean([m[k] for m in runs[name]])) for k in keys},
            "std": {k: float(np.nanstd([m[k] for m in runs[name]])) for k in keys},
        }
    if "default" in runs:
        base = np.array([m["energy_j"] for m in runs["default"]])
        for name in names:
            mine = np.array([m["energy_j"] for m in runs[name]])
            per_trace = 100.0 * (1.0 - mine / base)
            result["policies"][name]["energy_saving_pct"] = float(100.0 * (1.0 - mine.sum() / base.sum()))
            result["policies"][name]["energy_saving_pct_std"] = float(per_trace.std())
    return result


def save_json(obj, path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(obj, indent=1, default=float), encoding="utf-8")
