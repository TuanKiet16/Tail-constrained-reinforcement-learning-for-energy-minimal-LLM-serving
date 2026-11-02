from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

import numpy as np

from .config import WorkloadConfig


@dataclass
class Request:
    rid: int
    arrival: float  # seconds after the start of the episode
    prompt_len: int  # prompt tokens
    out_len: int  # tokens to generate (fixed, so every policy does identical work)
    n_out: int = 0
    first_token_t: float | None = None
    finish_t: float | None = None


def _lengths(rng: np.random.Generator, median: float, sigma: float, lo: int, hi: int, n: int) -> np.ndarray:
    return np.clip(np.rint(rng.lognormal(np.log(median), sigma, n)), lo, hi).astype(int)


@lru_cache(maxsize=32)
def _mean_lengths(key: tuple) -> tuple[float, float]:
    p_med, p_sig, p_lo, p_hi, o_med, o_sig, o_lo, o_hi = key
    rng = np.random.default_rng(12345)
    return (
        float(_lengths(rng, p_med, p_sig, p_lo, p_hi, 50_000).mean()),
        float(_lengths(rng, o_med, o_sig, o_lo, o_hi, 50_000).mean()),
    )


def mean_lengths(cfg: WorkloadConfig) -> tuple[float, float]:
    """Mean prompt and output length of the (clipped) length distributions."""
    return _mean_lengths(
        (cfg.prompt_median, cfg.prompt_sigma, cfg.prompt_min, cfg.prompt_max,
         cfg.output_median, cfg.output_sigma, cfg.output_min, cfg.output_max)
    )


def make_trace(
    cfg: WorkloadConfig, duration_s: float, capacity_rps: float, seed: int
) -> tuple[list[Request], list[tuple[float, float, str, float]]]:
    rng = np.random.default_rng(seed)
    regimes = cfg.regimes
    dwell = np.array([r["mean_dwell_s"] for r in regimes], dtype=float)
    k = int(rng.choice(len(regimes), p=dwell / dwell.sum()))
    arrivals: list[float] = []
    timeline: list[tuple[float, float, str, float]] = []
    t = 0.0
    while t < duration_s:
        end = min(duration_s, t + rng.exponential(dwell[k]))
        rate = regimes[k]["load"] * capacity_rps
        timeline.append((t, end, regimes[k]["name"], float(regimes[k]["load"])))
        if rate > 0:
            a = t + rng.exponential(1.0 / rate)
            while a < end:
                arrivals.append(a)
                a += rng.exponential(1.0 / rate)
        t = end
        if len(regimes) > 1:
            k = int(rng.choice([i for i in range(len(regimes)) if i != k]))
    n = len(arrivals)
    prompts = _lengths(rng, cfg.prompt_median, cfg.prompt_sigma, cfg.prompt_min, cfg.prompt_max, n)
    outputs = _lengths(rng, cfg.output_median, cfg.output_sigma, cfg.output_min, cfg.output_max, n)
    trace = [Request(i, float(arrivals[i]), int(prompts[i]), int(outputs[i])) for i in range(n)]
    return trace, timeline
