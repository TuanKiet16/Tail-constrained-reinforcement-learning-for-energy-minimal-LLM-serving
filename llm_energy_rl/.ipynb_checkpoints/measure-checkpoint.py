from __future__ import annotations

import statistics
import time

from .config import Config
from .profile import HardwareProfile
from .workload import Request, mean_lengths

_FOREVER = 10**9


def _fill(engine, batch: int, prompt_len: int = 128) -> None:
    engine.reset()
    engine.prefill([Request(i, 0.0, prompt_len, _FOREVER) for i in range(batch)])


def _decode_for(engine, gpu, seconds: float) -> tuple[float, float, list[float]]:
    clocks: list[float] = []
    e0, t0, steps = gpu.energy_j(), time.perf_counter(), 0
    while time.perf_counter() - t0 < seconds:
        n, _ = engine.decode(_FOREVER, time.perf_counter() + 0.25)
        steps += n
        clock = gpu.clock_mhz()
        if clock:
            clocks.append(float(clock))
    elapsed = time.perf_counter() - t0
    return elapsed / steps, (gpu.energy_j() - e0) / elapsed, clocks


def _prefill_time(engine, prompt_len: int, seconds: float) -> float:
    request = [Request(0, 0.0, prompt_len, _FOREVER)]
    t0, n = time.perf_counter(), 0
    while n < 3 or time.perf_counter() - t0 < seconds:
        engine.reset()
        engine.prefill(request)
        n += 1
    return (time.perf_counter() - t0) / n


def measure_throughput(engine, gpu, cfg: Config, seconds: float = 12.0) -> float:
    import numpy as np

    from .backends import RealBackend
    from .server import Server
    from .workload import _lengths

    wl, rng, n = cfg.workload, np.random.default_rng(0), 20_000
    prompts = _lengths(rng, wl.prompt_median, wl.prompt_sigma, wl.prompt_min, wl.prompt_max, n)
    outputs = _lengths(rng, wl.output_median, wl.output_sigma, wl.output_min, wl.output_max, n)
    backlog = [Request(i, 0.0, int(prompts[i]), int(outputs[i])) for i in range(n)]
    backend = RealBackend(engine, gpu, [None])
    backend.reset()
    server = Server(backend, cfg.max_batch)
    server.reset(backlog, backend.now())
    server.run_until(backend.now() + seconds / 3)  # reach a steady mix of young and old requests
    start = backend.now()
    stats = server.run_until(start + 2 * seconds / 3)
    rate = stats.tokens / (backend.now() - start)
    engine.reset()
    return rate


def batch_grid(max_batch: int) -> list[int]:
    sizes, b = [], 1
    while b < max_batch:
        sizes.append(b)
        b *= 2
    return sizes + [max_batch]


def warm_up(engine, gpu, seconds: float, batch: int) -> float | None:
    gpu.release()
    _fill(engine, batch)
    _, _, clocks = _decode_for(engine, gpu, seconds)
    engine.reset()
    return statistics.median(clocks[len(clocks) // 2:]) if clocks else None


def _clock_under_cap(engine, gpu, cap: float, batch: int, seconds: float) -> float | None:
    gpu.set_cap(cap)
    _fill(engine, batch)
    _, _, clocks = _decode_for(engine, gpu, seconds)
    engine.reset()
    return statistics.median(clocks[len(clocks) // 2:]) if clocks else None


def settle_lock_mode(engine, gpu, cap: float, batch: int, say=print, seconds: float = 1.5) -> None:
    def follows(seen):
        return seen is None or abs(seen - cap) <= 0.08 * cap

    seen = _clock_under_cap(engine, gpu, cap, batch, seconds)
    if not follows(seen) and gpu.lock_mode == "range":
        gpu.lock_mode = "exact"
        first, seen = seen, _clock_under_cap(engine, gpu, cap, batch, seconds)
        if follows(seen):
            say(f"  a range lock left the clock at {first:.0f} MHz for a {cap:.0f} MHz cap; pinning the clock instead")
    if not follows(seen):
        say(f"  WARNING: asked for a {cap:.0f} MHz cap but the clock ran at {seen:.0f} MHz under load. "
            "The cap may have no effect on this driver; consider knob: power in the config.")
    gpu.release()


def measure_profile(
    engine,
    gpu,
    cfg: Config,
    warmup_s: float = 15.0,
    settle_s: float = 1.5,
    window_s: float = 3.0,
    say=print,
) -> HardwareProfile:
    gpu.require_knob()
    say(f"Warming up {gpu.name} for {warmup_s:.0f} s ...")
    ref_clock = warm_up(engine, gpu, warmup_s, cfg.max_batch) or gpu.max_clock_mhz()
    if gpu.knob == "clock":
        if not ref_clock:
            raise RuntimeError("The driver does not report the GPU clock; set knob: power in the config.")
        caps: list[float | None] = [float(round(f * ref_clock)) for f in cfg.cap_fractions]
        unit = "MHz"
        say(f"Clock under load with default settings: {ref_clock:.0f} MHz")
        settle_lock_mode(engine, gpu, caps[len(caps) // 2], cfg.max_batch, say, seconds=max(settle_s, 0.3))
    else:
        limits = gpu.power_limits_w()
        floor = limits["min"] or 0.0
        caps = [float(round(max(floor, f * limits["current"]))) for f in cfg.cap_fractions]
        unit = "W"
        say(f"Power limit with default settings: {limits['current']:.0f} W (minimum {floor:.0f} W)")
    caps.append(None)

    batches = batch_grid(cfg.max_batch)
    context = int(mean_lengths(cfg.workload)[0])  
    out = {k: [] for k in ("decode_s", "decode_w", "prefill_base_s", "prefill_s_per_token", "prefill_w", "idle_w", "clock")}
    short, long_ = 32, 384
    try:
        for cap in caps:
            label = "default" if cap is None else f"{cap:.0f} {unit}"
            gpu.set_cap(cap)
            engine.reset()
            time.sleep(settle_s)
            e0, t0 = gpu.energy_j(), time.perf_counter()
            time.sleep(2.0)
            out["idle_w"].append((gpu.energy_j() - e0) / (time.perf_counter() - t0))

            row_s, row_w, clocks = [], [], []
            for b in batches:
                _fill(engine, b, context)
                _decode_for(engine, gpu, settle_s)  # let clock, voltage and the power reading settle
                sec, watts, seen = _decode_for(engine, gpu, window_s)
                row_s.append(sec)
                row_w.append(watts)
                clocks += seen
            out["decode_s"].append(row_s)
            out["decode_w"].append(row_w)
            out["clock"].append(statistics.median(clocks) if clocks else None)

            t_short = _prefill_time(engine, short, 0.5)
            e0, t0 = gpu.energy_j(), time.perf_counter()
            t_long = _prefill_time(engine, long_, 1.5)
            out["prefill_w"].append((gpu.energy_j() - e0) / (time.perf_counter() - t0))
            per_token = max((t_long - t_short) / (long_ - short), 0.0)
            out["prefill_s_per_token"].append(per_token)
            out["prefill_base_s"].append(max(t_short - per_token * short, 0.0))

            tok_s = batches[-1] / row_s[-1]
            say(
                f"  {label:>9}: {row_s[-1] * 1000:6.1f} ms/step at batch {batches[-1]} ({tok_s:5.0f} tok/s), "
                f"{row_w[-1]:5.0f} W, {row_w[-1] / tok_s:5.2f} J/token, idle {out['idle_w'][-1]:4.0f} W"
                + (f", clock {out['clock'][-1]:.0f} MHz" if out["clock"][-1] else "")
            )
        gpu.release()
        sustained = measure_throughput(engine, gpu, cfg, seconds=4 * window_s)
        switches = []
        for i in range(12):
            t0 = time.perf_counter()
            gpu.set_cap(caps[i % 2])
            switches.append(time.perf_counter() - t0)
        switch_s = statistics.median(switches)
        say(f"  changing the cap takes {switch_s * 1000:.1f} ms")
    finally:
        gpu.release()
        engine.reset()

    notes = []
    if gpu.knob == "clock":
        for cap, seen in zip(caps[:-1], out["clock"]):
            if seen and seen > cap * 1.05:
                notes.append(f"cap {cap:.0f} MHz was not honoured (clock stayed at {seen:.0f} MHz)")
    for note in notes:
        say(f"  WARNING: {note}")
    profile = HardwareProfile(
        knob=gpu.knob,
        caps=caps,
        batch_sizes=batches,
        decode_s=out["decode_s"],
        decode_w=out["decode_w"],
        prefill_base_s=out["prefill_base_s"],
        prefill_s_per_token=out["prefill_s_per_token"],
        prefill_w=out["prefill_w"],
        idle_w=out["idle_w"],
        switch_s=switch_s,
        meta={
            "source": "measured",
            "gpu": gpu.name,
            "driver": gpu.driver,
            "model": engine.model_id,
            "device": engine.device,
            "ref_clock_mhz": ref_clock,
            "clock_mhz": out["clock"],
            "lock_mode": gpu.lock_mode,
            "energy_counter": gpu.has_energy_counter,
            "warnings": notes,
            "sustained_tokens_per_s": sustained,
            "measured_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        },
    )
    mean_prompt, mean_out = mean_lengths(cfg.workload)
    stepwise = ((mean_out - 1.0) * profile.decode_time(profile.top, cfg.max_batch) / cfg.max_batch
                + profile.prefill_time(profile.top, mean_prompt))
    profile.request_overhead_s = max(0.0, mean_out / sustained - stepwise)
    say(f"  sustained throughput {sustained:.0f} tokens/s; overhead per request {profile.request_overhead_s * 1000:.1f} ms")
    return profile
