from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


def _default_regimes() -> list[dict]:
    return [
        {"name": "low", "load": 0.08, "mean_dwell_s": 40.0},
        {"name": "normal", "load": 0.35, "mean_dwell_s": 40.0},
        {"name": "burst", "load": 0.80, "mean_dwell_s": 20.0},
    ]


@dataclass
class WorkloadConfig:
  
    prompt_median: int = 128
    prompt_sigma: float = 0.6
    prompt_min: int = 16
    prompt_max: int = 512
    output_median: int = 64
    output_sigma: float = 0.6
    output_min: int = 8
    output_max: int = 256
    regimes: list[dict] = field(default_factory=_default_regimes)


@dataclass
class Config:
    model: str = "Qwen/Qwen2.5-1.5B-Instruct" 
    dtype: str = "float16"
    gpu_index: int = 0
    knob: str = "auto"  
    lock_mode: str = "range" 
    min_clock_mhz: int = 210
    cap_fractions: list[float] = field(default_factory=lambda: [0.5, 0.6, 0.7, 0.8, 0.9])
    max_batch: int = 8
    interval_s: float = 1.0  
    episode_s: float = 300.0
    tpot_factor: float = 1.25  
    ttft_factor: float = 1.0  
    slo_weight: float = 4.0  
    headroom: float = 0.10 
    sim_noise: float = 0.03 
    workload: WorkloadConfig = field(default_factory=WorkloadConfig)

    @property
    def n_levels(self) -> int:
        return len(self.cap_fractions) + 1


def load_config(path: str | Path | None = None, **overrides: Any) -> Config:
    data: dict[str, Any] = {}
    if path is not None:
        import yaml

        data = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    data.update({k: v for k, v in overrides.items() if v is not None})
    workload = data.pop("workload", None) or {}
    try:
        cfg = Config(**data)
        cfg.workload = WorkloadConfig(**workload)
    except TypeError as err:
        raise ValueError(f"Bad config ({path or 'defaults'}): {err}") from None
    return cfg
