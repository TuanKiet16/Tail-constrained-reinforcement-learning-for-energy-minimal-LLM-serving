from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np


@dataclass
class HardwareProfile:
    knob: str  # "clock" (caps in MHz) or "power" (caps in W)
    caps: list[float | None]  # one per action, low to high; None = driver default
    batch_sizes: list[int]
    decode_s: list[list[float]]  # [level][batch] seconds per decode iteration
    decode_w: list[list[float]]  # [level][batch] GPU watts while decoding
    prefill_base_s: list[float]  # [level] fixed seconds per prefill call
    prefill_s_per_token: list[float]  # [level] extra seconds per prompt token
    prefill_w: list[float]  # [level] GPU watts while prefilling
    idle_w: list[float]  # [level] GPU watts with nothing to do
    switch_s: float = 0.0  # seconds the driver call takes when the cap changes
    request_overhead_s: float = 0.0  # extra seconds per admitted request seen under full load
    meta: dict = field(default_factory=dict)

    @property
    def n_levels(self) -> int:
        return len(self.caps)

    @property
    def top(self) -> int:
        return len(self.caps) - 1

    def decode_time(self, level: int, batch: int) -> float:
        return float(np.interp(batch, self.batch_sizes, self.decode_s[level]))

    def decode_power(self, level: int, batch: int) -> float:
        return float(np.interp(batch, self.batch_sizes, self.decode_w[level]))

    def prefill_time(self, level: int, n_tokens: int, n_requests: int = 1) -> float:
        """Seconds to admit `n_requests` whose prompts total `n_tokens` tokens."""
        return (self.prefill_base_s[level] + self.prefill_s_per_token[level] * n_tokens
                + self.request_overhead_s * n_requests)

    def label(self, level: int) -> str:
        cap = self.caps[level]
        if cap is None:
            return "default"
        return f"{cap:.0f} {'MHz' if self.knob == 'clock' else 'W'}"

    def labels(self) -> list[str]:
        return [self.label(i) for i in range(self.n_levels)]

    def save(self, path: str | Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(json.dumps(asdict(self), indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "HardwareProfile":
        return cls(**json.loads(Path(path).read_text(encoding="utf-8")))


def synthetic_profile(
    cap_fractions: list[float] | tuple[float, ...] = (0.5, 0.6, 0.7, 0.8, 0.9),
    max_clock_mhz: float = 1900.0,
) -> HardwareProfile:
    phis = [float(f) for f in cap_fractions] + [1.0]
    batch_sizes = [1, 2, 4, 8, 16]
    idle = 30.0
    decode_s, decode_w = [], []
    prefill_base, prefill_tok, prefill_w, idle_w = [], [], [], []
    for phi in phis:
        scale = 0.15 + 0.85 * phi**2.3  
        row_s, row_w = [], []
        for b in batch_sizes:
            gpu_s = 0.011 * (1.0 + 0.10 * (b - 1))
            row_s.append(0.005 + gpu_s * (0.35 + 0.65 / phi))
            row_w.append(idle + min(120.0 + 18.0 * (b - 1), 290.0) * scale)
        decode_s.append(row_s)
        decode_w.append(row_w)
        prefill_base.append(0.008)
        prefill_tok.append(6e-5 * (0.10 + 0.90 / phi))
        prefill_w.append(idle + 250.0 * scale)
        idle_w.append(idle)
    return HardwareProfile(
        knob="clock",
        caps=[round(f * max_clock_mhz) for f in cap_fractions] + [None],
        batch_sizes=batch_sizes,
        decode_s=decode_s,
        decode_w=decode_w,
        prefill_base_s=prefill_base,
        prefill_s_per_token=prefill_tok,
        prefill_w=prefill_w,
        idle_w=idle_w,
        switch_s=0.005,
        meta={"source": "synthetic", "model": "placeholder profile (no real model)"},
    )
