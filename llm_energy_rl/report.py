from __future__ import annotations

import csv
import math
from pathlib import Path

import numpy as np

SURFACE, INK, INK2, MUTED, GRID, AXIS = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
COLORS = {"default": MUTED, "qlearning": "#2a78d6", "dqn": "#eb6834", "ppo": "#1baf7a"}
NAMES = {"default": "No RL (driver default)", "qlearning": "Q-learning", "dqn": "DQN", "ppo": "PPO"}
RAMP = ["#86b6ef", "#5598e7", "#2a78d6", "#1c5cab", "#0d366b"]


def _name(policy: str) -> str:
    return NAMES.get(policy, policy)


def _color(policy: str) -> str:
    return COLORS.get(policy, AXIS)


def _plt():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "figure.dpi": 150, "savefig.dpi": 150,
        "font.family": "sans-serif",
        "font.sans-serif": ["Segoe UI", "Helvetica Neue", "Arial", "DejaVu Sans"],
        "font.size": 9.5, "text.color": INK,
        "axes.titlesize": 10.5, "axes.titleweight": "bold", "axes.titlelocation": "left",
        "axes.titlepad": 10, "axes.labelsize": 9, "axes.labelcolor": INK2,
        "axes.edgecolor": AXIS, "axes.linewidth": 0.75, "axes.axisbelow": True,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.75, "grid.linestyle": "-",
        "xtick.color": AXIS, "ytick.color": AXIS, "xtick.labelcolor": MUTED, "ytick.labelcolor": MUTED,
        "xtick.labelsize": 8.5, "ytick.labelsize": 8.5, "xtick.major.size": 3, "ytick.major.size": 0,
        "lines.linewidth": 1.5, "lines.solid_capstyle": "round", "lines.solid_joinstyle": "round",
        "legend.frameon": False, "legend.fontsize": 8.5, "legend.labelcolor": INK2,
    })
    return plt


def _finish(fig, path, title: str, caption: str = "") -> None:
    """Left-aligned title above, grey caption below, then save."""
    import matplotlib.pyplot as plt

    fig.suptitle(title, x=0.01, ha="left", fontsize=10.5, fontweight="bold")
    if caption:
        fig.text(0.01, -0.01, caption, fontsize=8.5, color=MUTED, va="top", ha="left")
    fig.savefig(path, bbox_inches="tight", pad_inches=0.15)
    plt.close(fig)


def _hbars(ax, rows: list[tuple[float, float, str]], thickness_pt: float = 14.0, radius_pt: float = 3.0) -> None:
    from matplotlib.patches import PathPatch
    from matplotlib.path import Path as MPath

    fig = ax.figure
    fig.canvas.draw()  
    box = ax.get_window_extent()
    px_per_pt = fig.dpi / 72.0
    x_per_pt = abs(ax.get_xlim()[1] - ax.get_xlim()[0]) / box.width * px_per_pt
    y_per_pt = abs(ax.get_ylim()[1] - ax.get_ylim()[0]) / box.height * px_per_pt
    h, ry = thickness_pt * y_per_pt / 2, radius_pt * y_per_pt
    line, curve = MPath.LINETO, MPath.CURVE3
    for y, v, color in rows:
        if not math.isfinite(v) or v == 0:
            continue
        rx = math.copysign(min(radius_pt * x_per_pt, abs(v)), v)
        verts = [(0, y - h), (v - rx, y - h), (v, y - h), (v, y - h + ry), (v, y + h - ry),
                 (v, y + h), (v - rx, y + h), (0, y + h), (0, y - h)]
        codes = [MPath.MOVETO, line, curve, curve, line, curve, curve, line, MPath.CLOSEPOLY]
        ax.add_patch(PathPatch(MPath(verts, codes), facecolor=color, linewidth=0))


def _order(policies) -> list[str]:
    names = list(policies)
    return [n for n in ("default",) if n in names] + [n for n in names if n != "default"]


def write_summary(result: dict, path: str | Path, heading: str, notes: list[str]) -> str:
    lines = [f"# {heading}", ""] + [f"- {n}" for n in notes] + [
        "",
        "| Policy | Energy (kJ) | Energy vs no RL | J / token | Mean power (W) | TTFT p50 / p99 (s) | TPOT mean / p95 (ms) | Requests on target | Mean cap level |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for name in _order(result["policies"]):
        p = result["policies"][name]
        m = p["mean"]
        change = "–" if name == "default" or "energy_saving_pct" not in p else (
            f"{-p['energy_saving_pct']:+.1f}% (±{p['energy_saving_pct_std']:.1f})"
        )
        lines.append(
            f"| {_name(name)} | {m['energy_j'] / 1000:.2f} | {change} | {m['j_per_token']:.3f} | {m['mean_power_w']:.0f} "
            f"| {m['ttft_p50_s']:.2f} / {m['ttft_p99_s']:.2f} | {m['tpot_mean_s'] * 1000:.1f} / {m['tpot_p95_s'] * 1000:.1f} "
            f"| {m['slo_both'] * 100:.1f}% | {m['mean_level']:.2f} |"
        )
    text = "\n".join(lines) + "\n"
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(text, encoding="utf-8")
    return text


def plot_energy(result: dict, path: str | Path, caption: str = "") -> None:
    plt = _plt()
    names = _order(result["policies"])
    values = [result["policies"][n]["mean"]["energy_j"] / 1000 for n in names]
    fig, ax = plt.subplots(figsize=(7.2, 0.95 + 0.46 * len(names)), layout="constrained")
    ax.set_xlim(0, max(values) * 1.32)
    ax.set_ylim(len(names) - 0.5, -0.5)
    ax.set_yticks(range(len(names)), [_name(n) for n in names])
    ax.tick_params(axis="y", labelcolor=INK2, labelsize=9.5)
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("GPU energy per trace (kJ)")
    fig.suptitle("GPU energy for the same requests, with and without RL", x=0.01, ha="left", fontsize=10.5, fontweight="bold")
    _hbars(ax, [(i, v, _color(n)) for i, (n, v) in enumerate(zip(names, values))])
    for i, (n, v) in enumerate(zip(names, values)):
        label = f"{v:.1f} kJ"
        saving = result["policies"][n].get("energy_saving_pct")
        if n != "default" and saving is not None:
            label += f"   {-saving:+.1f}%".replace("-", "−")
        ax.annotate(label, (v, i), xytext=(6, 0), textcoords="offset points", va="center", fontsize=9, color=INK)
    _finish(fig, path, "GPU energy for the same requests, with and without RL", caption)


def plot_latency(result: dict, scales: dict, path: str | Path, caption: str = "") -> None:
    plt = _plt()
    from matplotlib.lines import Line2D

    names = _order(result["policies"])
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 1.6 + 0.48 * len(names)), layout="constrained", sharey=True)
    panels = [
        ("Time to first token (s)", "ttft_p50_s", "ttft_p99_s", scales["ttft_slo"], 1.0, "99th percentile"),
        ("Time per output token (ms)", "tpot_mean_s", "tpot_p95_s", scales["tpot_slo"], 1000.0, "95th percentile"),
    ]
    for ax, (title, lo_key, hi_key, target, unit, tail) in zip(axes, panels):
        target *= unit
        hi_max = 0.0
        for i, n in enumerate(names):
            m = result["policies"][n]["mean"]
            lo, hi = m[lo_key] * unit, m[hi_key] * unit
            hi_max = max(hi_max, hi)
            ax.plot([lo, hi], [i, i], color=AXIS, linewidth=1.5, zorder=2)
            ax.plot(lo, i, "o", markersize=6.5, markerfacecolor=SURFACE, markeredgecolor=_color(n), markeredgewidth=1.5, zorder=3)
            ax.plot(hi, i, "o", markersize=8, markerfacecolor=_color(n), markeredgecolor=SURFACE, markeredgewidth=1.5, zorder=3)
        ax.axvline(target, color=INK2, linewidth=0.75, zorder=1)
        ax.annotate(f"target {target:.3g}", (target, -0.5), xytext=(-4, -3), textcoords="offset points",
                    ha="right", va="top", fontsize=8, color=INK2)
        ax.set_xlim(0, max(hi_max, target) * 1.12)
        ax.set_ylim(len(names) - 0.5, -0.5)
        ax.grid(axis="y", visible=False)
        ax.set_title(title, fontsize=9.5, fontweight="normal", color=INK2)
        handles = [
            Line2D([], [], marker="o", linestyle="", markersize=6.5, markerfacecolor=SURFACE, markeredgecolor=MUTED, markeredgewidth=1.5),
            Line2D([], [], marker="o", linestyle="", markersize=8, markerfacecolor=MUTED, markeredgecolor=SURFACE, markeredgewidth=1.5),
        ]
        ax.legend(handles, ["median" if "ttft" in lo_key else "mean", tail], loc="upper left",
                  bbox_to_anchor=(0.0, -0.14), ncols=2, handletextpad=0.2, columnspacing=1.2, borderaxespad=0.0)
    labels = [f"{_name(n)}\n{result['policies'][n]['mean']['slo_both'] * 100:.1f}% on target" for n in names]
    axes[0].set_yticks(range(len(names)), labels)
    axes[0].tick_params(axis="y", labelcolor=INK2, labelsize=9)
    _finish(fig, path, "Latency against its targets", caption)


def _binned(times, values, width: float):
    times, values = np.asarray(times, float), np.asarray(values, float)
    edges = np.arange(0.0, times.max() + width, width)
    idx = np.clip(np.digitize(times, edges) - 1, 0, len(edges) - 2)
    sums = np.bincount(idx, weights=values, minlength=len(edges) - 1)
    counts = np.bincount(idx, minlength=len(edges) - 1)
    return edges[:-1] + width / 2, sums, np.maximum(counts, 1)


def plot_timeline(result: dict, policy: str, labels: list[str], path: str | Path, caption: str = "") -> None:
    plt = _plt()
    first = result["first_episode"]
    if policy not in first or "default" not in first:
        return
    mine, base = first[policy]["history"], first["default"]["history"]
    t = np.array([h["t"] for h in mine])
    fig, axes = plt.subplots(3, 1, figsize=(7.2, 5.6), layout="constrained", sharex=True,
                             gridspec_kw={"height_ratios": [1, 1.25, 1.4]})
    for ax in axes:
        ax.grid(axis="x", visible=False)

    centres, arrivals, _ = _binned(t, [h["arrivals"] for h in mine], 5.0)
    axes[0].bar(centres, arrivals / 5.0, width=3.4, color=MUTED, linewidth=0)
    axes[0].set_title("Requests arriving per second", fontsize=9.5, fontweight="normal", color=INK2)
    axes[0].set_ylim(bottom=0)

    axes[1].step(t, [h["level"] for h in mine], where="pre", color=_color(policy), linewidth=1.5)
    axes[1].set_yticks(range(len(labels)), labels)
    axes[1].set_ylim(-0.4, len(labels) - 0.6)
    axes[1].set_title(f"GPU cap chosen by {_name(policy)}", fontsize=9.5, fontweight="normal", color=INK2)

    for name, hist in (("default", base), (policy, mine)):
        c, energy, _ = _binned([h["t"] for h in hist], [h["energy_j"] for h in hist], 5.0)
        _, seconds, _ = _binned([h["t"] for h in hist], np.diff([0.0] + [h["t"] for h in hist]), 5.0)
        power = energy / np.maximum(seconds, 1e-9)
        axes[2].plot(c, power, color=_color(name), linewidth=1.5, label=_name(name))
    axes[2].set_ylim(bottom=0)
    axes[2].set_title("GPU power (W), 5-second averages", fontsize=9.5, fontweight="normal", color=INK2)
    axes[2].legend(loc="upper right", ncols=2, borderaxespad=0.1)
    axes[2].set_xlabel("Time (s)")
    _finish(fig, path, "One trace: the load, the cap the agent chose, GPU power", caption)


def read_curve(path: str | Path) -> list[dict]:
    with Path(path).open(encoding="utf-8") as f:
        return [{k: float(v) for k, v in row.items() if v not in ("", None)} for row in csv.DictReader(f)]


def plot_learning(curves: dict[str, list[dict]], path: str | Path, caption: str = "") -> None:
    plt = _plt()
    curves = {k: sorted(v, key=lambda r: r["step"]) for k, v in curves.items() if len(v) >= 2}
    if not curves:
        return
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.0), layout="constrained")
    for ax, key, title, scale in ((axes[0], "return", "Episode return", 1.0), (axes[1], "slo_both", "Requests on target (%)", 100.0)):
        for algo, rows in curves.items():
            steps = sorted({r["step"] for r in rows})
            x = np.array(steps) / 1000.0
            y = np.array([np.mean([r[key] for r in rows if r["step"] == st]) for st in steps]) * scale
            k = max(3, len(y) // 15)  # running mean over the last k episodes
            smooth = np.convolve(np.pad(y, (k - 1, 0), mode="edge"), np.ones(k) / k, mode="valid")
            ax.plot(x, smooth, color=_color(algo), linewidth=1.5, label=_name(algo))
        ax.set_title(title, fontsize=9.5, fontweight="normal", color=INK2)
        ax.set_xlabel("Training steps (thousands)")
        ax.grid(axis="x", visible=False)
    axes[1].set_ylim(top=100.5)
    axes[0].legend(loc="lower right")
    _finish(fig, path, "Learning curves", caption)


def plot_profile(profile, path: str | Path, caption: str = "") -> None:
    plt = _plt()
    x = np.arange(profile.n_levels)
    sizes = profile.batch_sizes
    shades = [RAMP[round(i * (len(RAMP) - 1) / max(1, len(sizes) - 1))] for i in range(len(sizes))]
    s, w = np.array(profile.decode_s), np.array(profile.decode_w)
    panels = [
        ("Output tokens per second", np.array(sizes)[None, :] / s),
        ("GPU power (W)", w),
        ("Energy per output token (J)", w * s / np.array(sizes)[None, :]),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(9.6, 3.1), layout="constrained")
    for ax, (title, data) in zip(axes, panels):
        for j, b in enumerate(sizes):
            ax.plot(x, data[:, j], color=shades[j], linewidth=1.5, marker="o", markersize=6.5,
                    markeredgecolor=SURFACE, markeredgewidth=1.5, label=f"{b}", clip_on=False, zorder=3)
        ax.set_xticks(x, [lab.replace(" ", "\n") for lab in profile.labels()])
        ax.set_ylim(bottom=0)
        ax.grid(axis="x", visible=False)
        ax.set_title(title, fontsize=9.5, fontweight="normal", color=INK2)
    handles, names = axes[0].get_legend_handles_labels()
    fig.legend(handles, names, title="Requests in the batch", title_fontsize=8.5, loc="outside upper right",
               ncols=len(sizes), columnspacing=1.0, handletextpad=0.3, alignment="right")
    axes[1].set_xlabel("GPU cap" + (" (clock)" if profile.knob == "clock" else " (power limit)"))
    _finish(fig, path, "What the cap does on this GPU", caption)


def plot_models(savings: dict[str, dict[str, float]], path: str | Path, caption: str = "") -> None:
    plt = _plt()
    from matplotlib.lines import Line2D

    models = list(savings)
    algos = [a for a in ("qlearning", "dqn", "ppo") if any(a in savings[m] for m in models)]
    if not models or not algos:
        return
    step = 0.27
    fig, ax = plt.subplots(figsize=(7.2, 1.1 + (0.25 + 0.3 * len(algos)) * len(models)), layout="constrained")
    values = [v for m in models for v in savings[m].values()]
    ax.set_xlim(min(0.0, min(values) * 1.3), max(max(values), 1.0) * 1.22)
    ax.set_ylim(len(models) - 0.5, -0.5)
    ax.set_yticks(range(len(models)), models)
    ax.tick_params(axis="y", labelcolor=INK2, labelsize=9)
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("GPU energy saved compared with no RL (%)")
    title = "Energy saved by RL, per model"
    fig.suptitle(title, x=0.01, ha="left", fontsize=10.5, fontweight="bold")
    handles = [Line2D([], [], marker="s", linestyle="", markersize=7, markerfacecolor=_color(a), markeredgewidth=0) for a in algos]
    fig.legend(handles, [_name(a) for a in algos], loc="outside upper right", ncols=len(algos), handletextpad=0.2, columnspacing=1.0)
    rows = []
    for i, m in enumerate(models):
        for j, a in enumerate(algos):
            if a in savings[m]:
                rows.append((i + (j - (len(algos) - 1) / 2) * step, savings[m][a], _color(a)))
    if min(values) < 0:  # bars can point left: draw the zero line they grow from
        ax.spines["left"].set_visible(False)
        ax.axvline(0, color=AXIS, linewidth=0.75)
    _hbars(ax, rows, thickness_pt=11.0)
    for y, v, _ in rows:
        ax.annotate(f"{v:.1f}%".replace("-", "−"), (max(v, 0.0), y), xytext=(6, 0), textcoords="offset points",
                    va="center", fontsize=8.5, color=INK)
    _finish(fig, path, title, caption)
