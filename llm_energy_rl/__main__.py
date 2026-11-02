from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

from .config import load_config


def _say(text: str = "") -> None:
    print(text, flush=True)


def cmd_probe(args) -> int:
    from .gpu import GpuError, NvmlGpu

    try:
        gpu = NvmlGpu(args.gpu, "auto")
    except GpuError as err:
        _say(f"No usable NVIDIA GPU: {err}")
        return 1
    d = gpu.describe()
    limits = d["power_limits_w"]
    period = gpu.energy_counter_period_s()
    _say(f"GPU              {d['name']} (driver {d['driver']})")
    _say(f"Now              {d['power_w']:.0f} W, {d['clock_mhz']} MHz, {d['temp_c']} C, {d['util_pct']}% busy")
    _say(f"Max clock        {d['max_clock_mhz']} MHz")
    _say(f"Power limit      {limits['current']} W (default {limits['default']}, allowed {limits['min']} to {limits['max']})")
    _say("Energy counter   " + (f"yes, updates every {period * 1000:.0f} ms" if period else
                                 "yes" if d["energy_counter"] else "no, energy will be integrated from power samples"))
    for knob in ("clock", "power"):
        state = "OK" if knob in d["usable_knobs"] else f"not usable: {d['knob_errors'].get(knob, 'unknown')}"
        _say(f"Cap by {knob:<6}    {state}")
    gpu.close()
    if limits["current"] and limits["default"] and limits["current"] < limits["default"] - 1:
        _say(f"\nNOTE: the power limit is below the card's default. If you did not lower it yourself, "
             f"restore it with: nvidia-smi -pl {limits['default']:.0f}")
    if d["knob"] == "none":
        if any("permission" in v for v in d["knob_errors"].values()):
            _say("\nThis terminal may not change GPU settings. Open it with administrator rights "
                 "(Windows: right-click PowerShell > Run as administrator; Linux: sudo) and probe again.")
        else:
            _say("\nThis GPU or driver accepts neither a clock cap nor a power limit (common on laptops). "
                 "The real-GPU experiment cannot run here; the simulator (demo) still works.")
        return 2
    _say(f"\nReady. Experiments will cap the GPU by {d['knob']}.")
    return 0


def cmd_demo(args) -> int:
    from .pipeline import run_demo

    cfg = load_config(args.config, episode_s=args.episode_s)
    run_demo(cfg, Path(args.out), args.algos, args.episodes, 0.1 if args.quick else 1.0, args.seed, _say)
    return 0


def cmd_run(args) -> int:
    from .gpu import GpuError
    from .pipeline import run_models

    overrides = dict(episode_s=args.episode_s)
    if args.dry_run:
        overrides.update(episode_s=args.episode_s or 20.0, interval_s=0.5)
    cfg = load_config(args.config, **overrides)
    models = ["tiny-random"] if args.dry_run else (args.models or [cfg.model])
    try:
        run_models(
            cfg, models, Path(args.out), say=_say, algos=args.algos, episodes=args.episodes,
            sim_scale=0.0 if args.no_sim else (0.05 if args.dry_run else 1.0),
            real_steps=args.real_steps, dry_run=args.dry_run, force=args.force, retrain=args.retrain,
            seed=args.seed,
            only_profile=args.only_profile,
        )
    except GpuError as err:
        _say(f"\nGPU problem: {err}")
        return 2
    except ValueError as err:
        _say(f"\nCannot continue: {err}")
        return 2
    return 0


def cmd_report(args) -> int:
    import json

    from .pipeline import write_report
    from .profile import HardwareProfile

    out = Path(args.dir)
    cfg = load_config(args.config)
    profile = HardwareProfile.load(out / "profile.json")
    result = json.loads((out / f"eval_{args.tag}.json").read_text(encoding="utf-8"))
    write_report(result, out, args.tag, profile, cfg, _say)
    return 0


def main(argv: list[str] | None = None) -> int:
    warnings.filterwarnings("ignore", category=UserWarning)
    for stream in (sys.stdout, sys.stderr): 
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(errors="replace")
    algos = ["qlearning", "dqn", "ppo"]
    p = argparse.ArgumentParser(prog="python -m llm_energy_rl", description=__doc__)
    sub = p.add_subparsers(dest="command", required=True)

    s = sub.add_parser("probe", help="check what your GPU and driver allow (run this first)")
    s.add_argument("--gpu", type=int, default=0)
    s.set_defaults(fn=cmd_probe)

    s = sub.add_parser("demo", help="whole pipeline in the simulator, no GPU needed")
    s.add_argument("--config")
    s.add_argument("--out", default="runs/demo")
    s.add_argument("--algos", nargs="+", default=algos, choices=algos)
    s.add_argument("--episodes", type=int, default=10, help="held-out traces for the comparison")
    s.add_argument("--episode-s", type=float)
    s.add_argument("--quick", action="store_true", help="a tenth of the training, to see it run")
    s.add_argument("--seed", type=int, default=0)
    s.set_defaults(fn=cmd_demo)

    s = sub.add_parser("run", help="the real experiment: profile, train, compare on the GPU")
    s.add_argument("--config", default=None)
    s.add_argument("--models", nargs="+", help="Hugging Face model ids (default: the one in the config)")
    s.add_argument("--out", default="runs/gpu")
    s.add_argument("--algos", nargs="+", default=algos, choices=algos)
    s.add_argument("--episodes", type=int, default=2, help="traces per policy in the GPU comparison")
    s.add_argument("--episode-s", type=float, help="length of one trace in seconds")
    s.add_argument("--real-steps", type=int, default=0, help="also train this many steps on the GPU itself")
    s.add_argument("--no-sim", action="store_true", help="skip simulator training (use with --real-steps)")
    s.add_argument("--only-profile", action="store_true", help="measure the hardware profile and stop")
    s.add_argument("--force", action="store_true", help="redo the profile and the training even if files exist")
    s.add_argument("--retrain", action="store_true", help="keep the profile, train the agents again (after changing reward settings)")
    s.add_argument("--dry-run", action="store_true", help="tiny random model on the CPU and a pretend GPU")
    s.add_argument("--seed", type=int, default=0)
    s.set_defaults(fn=cmd_run)

    s = sub.add_parser("report", help="rebuild tables and figures from a finished run")
    s.add_argument("--dir", required=True, help="a model folder inside the run folder")
    s.add_argument("--tag", default="real", choices=["real", "sim"])
    s.add_argument("--config")
    s.set_defaults(fn=cmd_report)

    args = p.parse_args(argv)
    return args.fn(args)


if __name__ == "__main__":
    sys.exit(main())
