from __future__ import annotations

from dataclasses import dataclass

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from .config import Config
from .profile import HardwareProfile
from .server import Server, WindowStats
from .workload import make_trace, mean_lengths

TRAIN_SEED0 = 10_000_000  

OBS_NAMES = [
    "queue",  # waiting requests
    "active",  # requests being decoded
    "wait",  # age of the longest-waiting request, relative to the TTFT target
    "load",  # smoothed arrival rate, relative to capacity
    "inter_token",  # time between tokens, relative to the TPOT target
    "busy",  # share of the interval the GPU was working
    "level",  # cap that was in force
    "power",  # mean GPU power
    "decode_rate",  # output tokens per second
    "prefill_rate",  # prompt tokens per second
]


@dataclass
class Scales:
    tpot_slo: float  # seconds per output token
    ttft_slo: float  # seconds to first token
    capacity_rps: float  # requests/s the server sustains with default clocks
    capacity_tps: float  # output tokens/s at that rate
    p_ref: float  # GPU watts at full batch with default clocks
    mean_prompt: float
    mean_out: float

    @classmethod
    def resolve(cls, profile: HardwareProfile, cfg: Config) -> "Scales":
        mean_prompt, mean_out = mean_lengths(cfg.workload)
        top, batch = profile.top, cfg.max_batch
        steps = mean_out - 1.0
        per_request = steps * profile.decode_time(top, batch) / batch + profile.prefill_time(top, mean_prompt)
        capacity_rps = 1.0 / per_request
        return cls(
            tpot_slo=cfg.tpot_factor * batch * per_request / steps,
            ttft_slo=cfg.ttft_factor * batch * per_request,
            capacity_rps=capacity_rps,
            capacity_tps=capacity_rps * mean_out,
            p_ref=profile.decode_power(top, batch),
            mean_prompt=mean_prompt,
            mean_out=mean_out,
        )


def _pressure(value: float, target: float, start: float) -> float:
    return float(np.clip((value / target - start) / (1.0 - start), 0.0, 3.0))


def _pct(values, q: float) -> float:
    return float(np.percentile(values, q)) if len(values) else float("nan")


class LLMServingEnv(gym.Env):

    metadata = {"render_modes": []}

    def __init__(self, backend, profile: HardwareProfile, cfg: Config, drain: bool = False):
        if backend.n_levels != profile.n_levels:
            raise ValueError("backend and profile disagree on the number of cap levels")
        self.backend, self.profile, self.cfg = backend, profile, cfg
        self.drain = drain  
        self.reward_scale = 1.0  
        self.scales = Scales.resolve(profile, cfg)
        self.server = Server(backend, cfg.max_batch)
        self.action_space = spaces.Discrete(profile.n_levels)
        self.observation_space = spaces.Box(0.0, 1.0, (len(OBS_NAMES),), np.float32)
        self.trace, self.timeline, self.history = [], [], []

    # ------------------------------------------------------------------ gym API
    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        trace_seed = (options or {}).get("trace_seed")
        if trace_seed is None:  
            trace_seed = int(self.np_random.integers(TRAIN_SEED0, 2**31 - 1))
        self.trace, self.timeline = make_trace(
            self.cfg.workload, self.cfg.episode_s, self.scales.capacity_rps, int(trace_seed)
        )
        self.backend.reset(int(trace_seed))
        self.t0 = self._last_t = self.backend.now()
        self._e0 = self._last_e = self.backend.energy_j()
        self.server.reset(self.trace, self.t0)
        self._load_ema = 0.0
        self.history = []
        return self._observe(WindowStats(), self.profile.top, 0.0, self.cfg.interval_s), {"trace_seed": trace_seed}

    def step(self, action):
        level = int(action)
        self.backend.set_level(level)
        st = self.server.run_until(self.backend.now() + self.cfg.interval_s)
        now, e = self.backend.now(), self.backend.energy_j()
        dt, energy = max(now - self._last_t, 1e-9), e - self._last_e
        self._last_t, self._last_e = now, e

        s = self.scales
        tpot, ttft = (1.0 - self.cfg.headroom) * s.tpot_slo, (1.0 - self.cfg.headroom) * s.ttft_slo
        violation = (
            _pressure(st.inter_token_s, tpot, 0.9)
            + (float(np.mean([_pressure(x, ttft, 0.9) for x in st.ttft])) if st.ttft else 0.0)
            + _pressure(st.oldest_wait, ttft, 0.5)
        )
        reward = -energy / (s.p_ref * self.cfg.interval_s) - self.cfg.slo_weight * violation

        elapsed = now - self.t0
        over = elapsed >= self.cfg.episode_s - 1e-9
        if self.drain:
            truncated = (over and self.server.all_done) or elapsed > 1.5 * self.cfg.episode_s + 120.0
        else:
            truncated = over
        row = {
            "t": elapsed, "level": level, "energy_j": energy, "power_w": energy / dt,
            "tokens": st.tokens, "arrivals": st.arrivals, "queue": st.queue_len,
            "active": st.active, "violation": violation, "reward": reward,
        }
        row.update(self.backend.telemetry())
        self.history.append(row)
        info = dict(row)
        if truncated:
            info["metrics"] = self.summary()
        return self._observe(st, level, energy, dt), float(reward) * self.reward_scale, False, bool(truncated), info

    def _observe(self, st: WindowStats, level: int, energy: float, dt: float) -> np.ndarray:
        s, batch = self.scales, self.cfg.max_batch
        self._load_ema = 0.7 * self._load_ema + 0.3 * (st.arrivals / dt) / s.capacity_rps
        obs = [
            min(st.queue_len, 2 * batch) / (2 * batch),
            st.active / batch,
            min(st.oldest_wait / s.ttft_slo, 2.0) / 2.0,
            min(self._load_ema, 2.0) / 2.0,
            min(st.inter_token_s / s.tpot_slo, 2.0) / 2.0,
            min(st.busy_time / dt, 1.0),
            level / (self.profile.n_levels - 1),
            min(energy / (s.p_ref * dt), 1.5) / 1.5,
            min(st.tokens / dt / s.capacity_tps, 1.5) / 1.5,
            min(st.prefill_tokens / dt / (s.capacity_rps * s.mean_prompt), 3.0) / 3.0,
        ]
        return np.asarray(obs, dtype=np.float32)

    def summary(self) -> dict:
        s, t0, now = self.scales, self.t0, self._last_t
        arrived = self.trace[: self.server.next]
        done = self.server.done
        ttft = [r.first_token_t - (t0 + r.arrival) for r in arrived if r.first_token_t is not None]
        tpot = [(r.finish_t - r.first_token_t) / (r.out_len - 1) for r in done if r.out_len > 1]
        e2e = [r.finish_t - (t0 + r.arrival) for r in done]

        def meets(r) -> tuple[bool, bool]:
            fast_start = r.first_token_t - (t0 + r.arrival) <= s.ttft_slo
            fast_tokens = r.out_len <= 1 or (r.finish_t - r.first_token_t) / (r.out_len - 1) <= s.tpot_slo
            return fast_start, fast_tokens

        checks = [meets(r) for r in done]
        stuck = sum(1 for r in arrived if r.first_token_t is None and now - (t0 + r.arrival) > s.ttft_slo)
        judged = len(done) + stuck
        energy = self._last_e - self._e0
        tokens = sum(r.n_out for r in arrived)
        return {
            "energy_j": energy,
            "duration_s": now - t0,
            "mean_power_w": energy / max(now - t0, 1e-9),
            "requests": len(arrived),
            "completed": len(done),
            "tokens": tokens,
            "j_per_token": energy / tokens if tokens else float("nan"),
            "ttft_p50_s": _pct(ttft, 50), "ttft_p95_s": _pct(ttft, 95), "ttft_p99_s": _pct(ttft, 99),
            "tpot_mean_s": float(np.mean(tpot)) if tpot else float("nan"),
            "tpot_p95_s": _pct(tpot, 95),
            "e2e_mean_s": float(np.mean(e2e)) if e2e else float("nan"),
            "slo_ttft": sum(a for a, _ in checks) / judged if judged else 1.0,
            "slo_tpot": sum(b for _, b in checks) / len(done) if done else 1.0,
            "slo_both": sum(a and b for a, b in checks) / judged if judged else 1.0,
            "edp": (energy / len(done)) * float(np.mean(e2e)) if done else float("nan"),
            "mean_level": float(np.mean([h["level"] for h in self.history])) if self.history else float("nan"),
            "return": float(np.sum([h["reward"] for h in self.history])),
        }

    def close(self):
        self.backend.close()
