from __future__ import annotations

import math
import time

import numpy as np

from .profile import HardwareProfile


class SimBackend:

    def __init__(self, profile: HardwareProfile, noise: float = 0.0, seed: int = 0):
        self.profile = profile
        self.noise = noise
        self.rng = np.random.default_rng(seed)
        self.n_levels = profile.n_levels
        self.t = 0.0
        self.energy = 0.0
        self.level = profile.top
        self.batch = 0

    def _jitter(self) -> float:
        return float(self.rng.lognormal(0.0, self.noise)) if self.noise > 0 else 1.0

    def reset(self, seed: int | None = None) -> None:
        self.t, self.energy, self.batch = 0.0, 0.0, 0
        self.level = self.profile.top
        if seed is not None:
            self.rng = np.random.default_rng(seed)

    def now(self) -> float:
        return self.t

    def set_level(self, level: int) -> None:
        level = int(level)
        if level != self.level:  # changing the cap is a driver call that takes time
            self.idle_until(self.t + self.profile.switch_s)
            self.level = level

    def energy_j(self) -> float:
        return self.energy

    def idle_until(self, t: float) -> None:
        if t > self.t:
            self.energy += self.profile.idle_w[self.level] * (t - self.t)
            self.t = t

    def prefill(self, requests) -> None:
        tokens = sum(r.prompt_len for r in requests)
        dt = self.profile.prefill_time(self.level, tokens, len(requests)) * self._jitter()
        self.energy += self.profile.prefill_w[self.level] * self._jitter() * dt
        self.t += dt
        self.batch += len(requests)

    def decode(self, max_steps: int, until: float) -> tuple[int, float]:
        tau = self.profile.decode_time(self.level, self.batch) * self._jitter()
        steps = int(min(max_steps, max(1, math.ceil((until - self.t) / tau))))
        dt = steps * tau
        self.energy += self.profile.decode_power(self.level, self.batch) * self._jitter() * dt
        self.t += dt
        return steps, dt

    def remove(self, rows: list[int]) -> None:
        self.batch -= len(rows)

    def telemetry(self) -> dict:
        return {}

    def close(self) -> None:
        pass


class RealBackend:

    def __init__(self, engine, gpu, caps: list[float | None]):
        self.engine = engine
        self.gpu = gpu
        self.caps = list(caps)
        self.n_levels = len(caps)
        self.level: int | None = None

    def reset(self, seed: int | None = None) -> None:
        self.engine.reset()
        self.set_level(self.n_levels - 1)

    def now(self) -> float:
        return time.perf_counter()

    def set_level(self, level: int) -> None:
        level = int(level)
        if level != self.level:  # talking to the driver costs time, skip no-ops
            self.gpu.set_cap(self.caps[level])
            self.level = level

    def energy_j(self) -> float:
        return self.gpu.energy_j()

    def idle_until(self, t: float) -> None:
        while (dt := t - time.perf_counter()) > 0:
            time.sleep(min(dt, 0.05))

    def prefill(self, requests) -> None:
        self.engine.prefill(requests)

    def decode(self, max_steps: int, until: float) -> tuple[int, float]:
        return self.engine.decode(max_steps, until)

    def remove(self, rows: list[int]) -> None:
        self.engine.remove(rows)

    def telemetry(self) -> dict:
        return self.gpu.telemetry()

    def close(self) -> None:
        self.engine.reset()
        self.gpu.release()
        self.level = None
