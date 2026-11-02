from __future__ import annotations

import gc
import re
import time
from dataclasses import asdict, replace
from pathlib import Path

from . import report
from .agents import FixedPolicy, QLearningAgent, SB3Policy, load_policy
from .backends import RealBackend
from .config import Config
from .env import LLMServingEnv, Scales
from .experiment import evaluate, make_sim_env, save_json, train_qlearning, train_sb3, validation_return
from .profile import HardwareProfile, synthetic_profile

ALGOS = ("qlearning", "dqn", "ppo")
SIM_STEPS = {"qlearning": 300_000, "dqn": 80_000, "ppo": 300_000}


def slug(model: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", model).strip("_")


def policy_file(out: Path, algo: str) -> Path:
    return out / ("qlearning.json" if algo == "qlearning" else f"{algo}.zip")


def _progress(say, algo: str):
    last = {"shown": -1}

    def show(step: int, total: int, m: dict) -> None:
        tenth = int(10 * step / total)
        if tenth > last["shown"]:
            last["shown"] = tenth
            say(f"  [{algo}] step {step:>7}/{total}  return {m['return']:8.1f}  "
                f"energy {m['energy_j'] / 1000:6.1f} kJ  on target {m['slo_both'] * 100:5.1f}%")

    return show


def train_in_sim(algo: str, profile: HardwareProfile, cfg: Config, steps: int, out: Path, seed: int = 0, say=print):
    judge = make_sim_env(profile, cfg, seed + 100)

    def validate(policy) -> float:
        return validation_return(judge, policy)

    return train(algo, sim_envs(algo, profile, cfg, seed), steps, out, seed, say=say, validate=validate)


def sim_envs(algo: str, profile: HardwareProfile, cfg: Config, seed: int):
    copies = 4 if algo == "ppo" else 1
    envs = [make_sim_env(profile, cfg, seed + i) for i in range(copies)]
    return envs if copies > 1 else envs[0]


def train(algo: str, env, steps: int, out: Path, seed: int = 0, init: Path | None = None,
          tag: str = "sim", say=print, validate=None):
    out.mkdir(parents=True, exist_ok=True)
    curve = out / f"curve_{algo}_{tag}.csv"
    if algo == "qlearning":
        agent = QLearningAgent.load(init) if init else None
        extra = dict(eps_start=0.10, eps_end=0.02, eps_fraction=1.0) if init else {}
        agent = train_qlearning(env, steps, seed, agent=agent, curve=curve, progress=_progress(say, algo),
                                validate=validate, **extra)
        agent.save(policy_file(out, algo))
        return agent
    model = load_policy(f"{algo}:{init}", 0).model if init else None
    replay = out / "dqn_replay.pkl"
    if algo == "dqn" and model is not None and replay.exists():
        model.load_replay_buffer(replay)
    model = train_sb3(algo, env, steps, seed, model=model, curve=curve, progress=_progress(say, algo),
                      validate=validate)
    model.save(policy_file(out, algo))
    if algo == "dqn":
        model.save_replay_buffer(replay)
    return SB3Policy(model, algo)


def describe(profile: HardwareProfile, cfg: Config, where: str) -> list[str]:
    s = Scales.resolve(profile, cfg)
    notes = [
        f"Evaluated {where}",
        f"Model: {profile.meta.get('model', cfg.model)}",
        f"GPU: {profile.meta.get('gpu', 'none')}",
        f"Caps ({profile.knob}): {', '.join(profile.labels())}",
        f"Targets: time per output token <= {s.tpot_slo * 1000:.1f} ms, time to first token <= {s.ttft_slo:.2f} s",
        f"Load: up to {cfg.max_batch} requests at once, arrival rate between "
        f"{min(r['load'] for r in cfg.workload.regimes):.0%} and {max(r['load'] for r in cfg.workload.regimes):.0%} "
        f"of capacity ({s.capacity_rps:.2f} requests/s)",
    ]
    if profile.meta.get("source") == "synthetic":
        notes.append("The hardware profile is the built-in placeholder, so these numbers are not a result about any real GPU")
    return notes


def evaluate_and_report(env: LLMServingEnv, policies: dict, episodes: int, out: Path, tag: str,
                        profile: HardwareProfile, cfg: Config, where: str, say=print, between=None) -> dict:
    """Run the comparison, then write eval_<tag>.json and report_<tag>/."""
    def show(name, ep, m):
        say(f"  trace {ep + 1}/{episodes}  {name:<10} {m['energy_j'] / 1000:7.2f} kJ  {m['j_per_token']:.3f} J/token  "
            f"TTFT p99 {m['ttft_p99_s']:.2f} s  TPOT {m['tpot_mean_s'] * 1000:.1f} ms  on target {m['slo_both'] * 100:.1f}%")

    result = evaluate(env, policies, episodes, between=between, progress=show)
    result["scales"] = asdict(env.scales)
    result["labels"] = profile.labels()
    result["where"] = where
    save_json(result, out / f"eval_{tag}.json")
    write_report(result, out, tag, profile, cfg, say)
    return result


def write_report(result: dict, out: Path, tag: str, profile: HardwareProfile, cfg: Config, say=print) -> None:
    rep = out / f"report_{tag}"
    rep.mkdir(parents=True, exist_ok=True)
    where = result.get("where", tag)
    model = profile.meta.get("model", cfg.model)
    sub = f"{model} · {where} · mean of {result['episodes']} held-out trace(s) of {cfg.episode_s:.0f} s"
    text = report.write_summary(result, rep / "summary.md", f"RL vs no RL: {model}", describe(profile, cfg, where))
    report.plot_energy(result, rep / "energy.png", sub)
    report.plot_latency(result, result["scales"], rep / "latency.png", sub)
    for name in result["policies"]:
        if name != "default":
            report.plot_timeline(result, name, result["labels"], rep / f"timeline_{name}.png", f"{model} · {where}")
    curves = {}
    for algo in ALGOS:
        rows = []
        for stage in ("sim", "real"):
            path = out / f"curve_{algo}_{stage}.csv"
            if path.exists():
                offset = rows[-1]["step"] if rows else 0.0
                rows += [{**r, "step": r["step"] + offset} for r in report.read_curve(path)]
        if rows:
            curves[algo] = rows
    report.plot_learning(curves, rep / "learning.png", f"{model} · one point per training episode, smoothed")
    report.plot_profile(profile, rep / "profile.png", f"{model} · {profile.meta.get('gpu', 'no GPU')}")
    say("")
    say(text)
    say(f"Report written to {rep}")


def run_demo(cfg: Config, out: Path, algos=ALGOS, episodes: int = 10, scale: float = 1.0, seed: int = 0, say=print) -> dict:
    out.mkdir(parents=True, exist_ok=True)
    profile = synthetic_profile(cfg.cap_fractions)
    profile.save(out / "profile.json")
    policies = {"default": FixedPolicy(profile.top, "default")}
    for algo in algos:
        steps = max(2_000, int(SIM_STEPS[algo] * scale))
        say(f"Training {algo} in the simulator for {steps} steps ...")
        policies[algo] = train_in_sim(algo, profile, cfg, steps, out, seed, say)
    say("Evaluating on held-out traces ...")
    return evaluate_and_report(make_sim_env(profile, cfg, seed + 1, drain=True), policies, episodes, out, "sim",
                               profile, cfg, "in the simulator", say)


def open_gpu(cfg: Config, dry_run: bool = False):
    from .gpu import NvmlGpu

    nvml = None
    if dry_run:
        from .fake_nvml import FakeNvml

        nvml = FakeNvml()
    return NvmlGpu(cfg.gpu_index, cfg.knob, cfg.lock_mode, cfg.min_clock_mhz, nvml=nvml)


def open_engine(cfg: Config, dry_run: bool = False):
    from .engine import HFEngine

    return HFEngine(cfg.model, cfg.dtype, device="cpu" if dry_run else None)


def run_model(cfg: Config, out: Path, algos=ALGOS, episodes: int = 2, sim_scale: float = 1.0,
              real_steps: int = 0, dry_run: bool = False, force: bool = False, retrain: bool = False,
              seed: int = 0, only_profile: bool = False, say=print) -> dict | None:
    from .measure import measure_profile, warm_up

    out.mkdir(parents=True, exist_ok=True)
    gpu = open_gpu(cfg, dry_run)
    engine = None
    try:
        gpu.require_knob()
        say(f"GPU: {gpu.name} (driver {gpu.driver}), capping by {gpu.knob}")
        limits = gpu.power_limits_w()
        if limits["current"] and limits["default"] and limits["current"] < limits["default"] - 1:
            say(f"NOTE: the power limit is {limits['current']:.0f} W, below the card's default of "
                f"{limits['default']:.0f} W. If that is left over from an interrupted run, restore it with: "
                f"nvidia-smi -pl {limits['default']:.0f}")
        say(f"Loading {cfg.model} ...")
        engine = open_engine(cfg, dry_run)
        ppath = out / "profile.json"
        if ppath.exists() and not force:
            profile = HardwareProfile.load(ppath)
            if profile.n_levels != cfg.n_levels or cfg.max_batch > max(profile.batch_sizes):
                raise ValueError(f"{ppath} was measured with different settings; run again with --force")
            say(f"Using the existing profile {ppath}")
        else:
            quick = dict(warmup_s=1.0, settle_s=0.2, window_s=0.5) if dry_run else {}
            profile = measure_profile(engine, gpu, cfg, say=say, **quick)
            profile.save(ppath)
            for stale in [*out.glob("curve_*.csv"), out / "dqn_replay.pkl", *(policy_file(out, a) for a in ALGOS)]:
                stale.unlink(missing_ok=True)  # policies trained for the old profile
        gpu.lock_mode = profile.meta.get("lock_mode", gpu.lock_mode)  # cap the way the profile was measured
        s = Scales.resolve(profile, cfg)
        say(f"Targets: TPOT <= {s.tpot_slo * 1000:.1f} ms, TTFT <= {s.ttft_slo:.2f} s; "
            f"capacity {s.capacity_rps:.2f} requests/s ({s.capacity_tps:.0f} tokens/s)")
        report.plot_profile(profile, out / "profile.png", f"{cfg.model} · {gpu.name}")
        if only_profile:
            say(f"Profile saved to {ppath} (figure: {out / 'profile.png'})")
            return None

        policies = {"default": FixedPolicy(profile.top, "default")}
        for algo in algos:
            if policy_file(out, algo).exists() and not (force or retrain):
                say(f"Using the existing {algo} policy")
                policies[algo] = load_policy(f"{algo}:{policy_file(out, algo)}", profile.n_levels)
            elif sim_scale > 0:
                steps = max(2_000, int(SIM_STEPS[algo] * sim_scale))
                say(f"Training {algo} in the calibrated simulator for {steps} steps ...")
                policies[algo] = train_in_sim(algo, profile, cfg, steps, out, seed, say)
            elif real_steps <= 0:
                raise ValueError(f"No {algo} policy to evaluate: --no-sim needs --real-steps")

        real_env = LLMServingEnv(RealBackend(engine, gpu, profile.caps), profile, cfg)
        if real_steps > 0:
            for algo in algos:
                start = policy_file(out, algo) if algo in policies else None
                say(f"{'Fine-tuning' if start else 'Training'} {algo} on the GPU for {real_steps} steps "
                    f"(about {real_steps * cfg.interval_s / 60:.0f} min) ...")
                real_env.drain = False
                policies[algo] = train(algo, real_env, real_steps, out, seed, init=start, tag="real", say=say)

        minutes = episodes * len(policies) * cfg.episode_s / 60
        say(f"Evaluating on the GPU: {episodes} trace(s) x {len(policies)} policies, about {minutes:.0f} min ...")
        if not dry_run:
            warm_up(engine, gpu, 20.0, cfg.max_batch)
        where = "on a pretend GPU (dry run, energy values are meaningless)" if dry_run else f"on {gpu.name}"
        return evaluate_and_report(real_env, policies, episodes, out, "real", profile, cfg, where, say,
                                   between=lambda: time.sleep(0.2 if dry_run else 3.0))
    finally:
        if engine is not None:
            engine.reset()
        gpu.close()


def _free_gpu_memory() -> None:
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass


def run_models(cfg: Config, models: list[str], out: Path, say=print, **kwargs) -> None:
    out.mkdir(parents=True, exist_ok=True)
    savings, rows = {}, []
    for model in models:
        say(f"\n=== {model} ===")
        result = run_model(replace(cfg, model=model), out / slug(model), say=say, **kwargs)
        _free_gpu_memory()
        if result is None:
            continue
        base = result["policies"]["default"]["mean"]
        savings[model] = {}
        for name, p in result["policies"].items():
            if name == "default":
                continue
            savings[model][name] = p["energy_saving_pct"]
            rows.append(
                f"| {model} | {report.NAMES.get(name, name)} | {base['energy_j'] / 1000:.2f} | {p['mean']['energy_j'] / 1000:.2f} "
                f"| {-p['energy_saving_pct']:+.1f}% | {base['slo_both'] * 100:.1f}% | {p['mean']['slo_both'] * 100:.1f}% |"
            )
    if not rows:
        return
    lines = [
        "# RL vs no RL across models", "",
        "| Model | Learner | Energy, no RL (kJ) | Energy, RL (kJ) | Change | On target, no RL | On target, RL |",
        "|---|---|---|---|---|---|---|",
    ] + rows
    (out / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    caption = ("Dry run on a pretend GPU: these values are meaningless" if kwargs.get("dry_run")
               else "Measured GPU energy, same request traces for every policy")
    report.plot_models(savings, out / "models.png", caption)
    say("\n" + "\n".join(lines))
    say(f"\nAll results are in {out}")
