"""Plot GRPO training curves and held-out test results for the active_space_candidates experiment (oracle excluded).

Reads what `bash brev/run_on_brev.sh pull` copies back:
  * logs/grpo-active-space-candidates-<run>/exp_*/tensorboard/  NeMo RL training metrics, one dir per model
    (lightning35, qwen3-4b, ...); per model the exp_* with the most logged steps is used (earlier ones are failed runs)
  * results/candidates_*_test.jsonl                              test rollouts (base and GRPO, every model)

and writes to results/plots/:
  training_curves.png  loss and mean training reward per GRPO step (raw and smoothed), one line per model
  test_results.png     mean test reward and near-oracle rate per method, with 95% bootstrap CIs over materials
  test_per_material.png  mean test reward per held-out material, per method
  summary.csv          the numbers behind test_results.png

    uv run --with tensorboard --with matplotlib --with numpy python scripts/candidates/plot_results.py
"""

import argparse
import csv
import json
import random
import re
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))
from evaluate import ORBITALS, SCORES, load_rollouts, rule_picks  # noqa: E402  (same scorer as the table)


def run_label(run: str) -> str:
    """lightning35 -> 'Nemotron 3.5 Lightning v1', lightning35-v2 -> '... v2', qwen3-4b-v2 -> 'Qwen3-4B v2'."""
    base, _, version = run.partition("-v")
    if base == "lightning35":
        name = "Nemotron 3.5 Lightning"
    else:
        m = re.fullmatch(r"qwen3-([\d.]+)b", base)
        name = f"Qwen3-{m.group(1)}B" if m else base
    return f"{name} v{version or 1}"


def method_label(stem: str) -> str:
    """candidates_base_test -> 'Lightning base', candidates_qwen3-4b_grpo_test -> 'Qwen3-4B GRPO'."""
    body = stem.removeprefix("candidates_").removesuffix("_test")
    model, _, kind = body.rpartition("_")
    name = run_label(model).rsplit(" v", 1)[0] if model else "Lightning"
    name = name.replace("Nemotron 3.5 ", "")
    return f"{name} {'GRPO' if kind == 'grpo' else kind}"


# ---------------------------------------------------------------- training curves

def read_scalars(exp_dir: Path) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

    out = {}
    for f in sorted(exp_dir.rglob("events.out.tfevents.*")):
        ea = EventAccumulator(str(f), size_guidance={"scalars": 0})
        ea.Reload()
        for tag in ea.Tags()["scalars"]:
            ev = ea.Scalars(tag)
            # some tags are logged twice per step; keep the last value for each step
            by_step = {e.step: e.value for e in ev}
            steps = np.array(sorted(by_step))
            out[tag] = (steps, np.array([by_step[s] for s in steps]))
    return out


def find_runs(logs: Path) -> dict[str, dict]:
    """{run name: scalars of its most complete exp_*}."""
    runs = {}
    for run_dir in sorted(logs.glob("grpo-active-space-candidates-*")):
        best, best_key = None, (-1, "")
        for exp in sorted(run_dir.glob("exp_*")):
            scalars = read_scalars(exp)
            n = len(scalars.get("train/loss", ([], []))[0])
            if (n, exp.name) > best_key:
                best, best_key = scalars, (n, exp.name)
        if best and best_key[0] > 0:
            runs[run_dir.name.removeprefix("grpo-active-space-candidates-")] = best
            print(f"training run {run_dir.name}: {run_dir / best_key[1]} ({best_key[0]} steps)")
    return runs


def ema(y: np.ndarray, alpha: float) -> np.ndarray:
    out, acc = np.empty_like(y, dtype=float), y[0]
    for i, v in enumerate(y):
        acc = alpha * v + (1 - alpha) * acc
        out[i] = acc
    return out


def rolling_median(y: np.ndarray, window: int = 5) -> np.ndarray:
    h = window // 2
    return np.array([np.median(y[max(0, i - h): i + h + 1]) for i in range(len(y))])


def plot_training(runs: dict, out: Path, alpha: float):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))
    for i, (run, sc) in enumerate(runs.items()):
        color = f"C{i}"
        for ax, tag in zip(axes, ["train/loss", "train/reward"]):
            if tag not in sc:
                continue
            x, y = sc[tag]
            ax.plot(x, y, color=color, alpha=0.3, lw=1, marker="o", ms=3)
            # the loss has rare huge spikes that would drag an EMA for many steps; a median ignores them
            smooth = rolling_median(y) if tag == "train/loss" else ema(y, alpha)
            ax.plot(x, smooth, color=color, lw=2, label=run_label(run))
        if "train/reward" in sc and "train/total_reward/stddev" in sc:
            x, y = sc["train/reward"]
            sx, sd = sc["train/total_reward/stddev"]
            sd = np.interp(x, sx, sd)
            axes[1].fill_between(x, np.clip(y - sd, 0, 1), np.clip(y + sd, 0, 1), color=color, alpha=0.08)
    axes[0].set(title="GRPO loss (thick: rolling median of 5)", xlabel="training step", ylabel="loss")
    # occasional large spikes (importance-ratio outliers) would flatten everything else on a linear axis
    axes[0].set_yscale("symlog", linthresh=1.0)
    axes[0].axhline(0, color="grey", lw=0.5)
    axes[1].set(title="Mean training reward (band: ±1 std over rollouts)", xlabel="training step",
                ylabel="reward (0 = worst, 1 = oracle)", ylim=(0, 1))
    for ax in axes:
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
    fig.suptitle(f"Training curves (thin: per step; reward thick: EMA α={alpha})")
    fig.tight_layout()
    fig.savefig(out, dpi=160)
    plt.close(fig)


# ---------------------------------------------------------------- test results

def per_material_rewards(picks: dict, mats: dict, tol: float) -> dict[str, tuple[float, float]]:
    """{material: (mean reward, near-oracle rate)} over that material's answers."""
    res = {}
    for mid, choices in picks.items():
        cands = mats[mid]["candidates"]
        rs = np.array([cands[c]["reward"] if c in cands else 0.0 for c in choices])
        res[mid] = (rs.mean(), (rs >= 1 - tol).mean())
    return res


def bootstrap_ci(values: np.ndarray, n: int = 5000, seed: int = 0) -> tuple[float, float]:
    """95% CI of the mean, resampling materials (the main source of variance with 8 test materials)."""
    rng = np.random.default_rng(seed)
    means = rng.choice(values, size=(n, len(values)), replace=True).mean(axis=1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def collect_methods(results: Path, mats: dict, tol: float) -> dict[str, dict]:
    methods = {}
    # Same random baseline as scripts/candidates/evaluate.py (64 seeded draws per material), so the numbers match the table.
    rng = random.Random(0)
    rnd = {mid: [rng.choice(list(m["candidates"])) for _ in range(64)] for mid, m in mats.items()}
    methods["random menu entry"] = per_material_rewards(rnd, mats, tol)
    for name, picks in rule_picks(json.loads(ORBITALS.read_text()), mats).items():
        methods[name] = per_material_rewards(picks, mats, tol)
    files = sorted(results.glob("candidates_*_test.jsonl"), key=lambda p: (p.stem.count("_"), p.stem))
    for path in files:
        picks = {mid: v for mid, v in load_rollouts(path).items() if mid in mats}
        if picks:
            methods[method_label(path.stem)] = per_material_rewards(picks, mats, tol)
    return methods


def plot_test(methods: dict, mats: dict, out_dir: Path, tol: float):
    names = list(methods)
    n_baselines = sum(not n[0].isupper() for n in names)   # "random ..." and "rule: ..." rows, drawn in greys
    colors = [str(0.45 + 0.15 * i) for i in range(n_baselines)] + [f"C{i}" for i in range(len(names) - n_baselines)]
    rows = []
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    for ax, k, title in [(axes[0], 0, "Mean test reward"), (axes[1], 1, f"Near-oracle rate (reward ≥ {1 - tol:.2f})")]:
        means, lo, hi = [], [], []
        for name in names:
            vals = np.array([v[k] for v in methods[name].values()])
            m = vals.mean()
            a, b = bootstrap_ci(vals)
            means.append(m), lo.append(m - a), hi.append(b - m)
            if k == 0:
                rows.append({"method": name, "mean_reward": m, "ci_low": a, "ci_high": b})
            else:
                rows[names.index(name)].update({"near_oracle_rate": m, "near_ci_low": a, "near_ci_high": b})
        bars = ax.bar(names, means, yerr=[lo, hi], capsize=4, color=colors)
        ax.bar_label(bars, fmt="%.3f", padding=3, fontsize=8)
        ax.set(title=title, ylim=(0, 1))
        ax.tick_params(axis="x", rotation=25)
        ax.grid(axis="y", alpha=0.3)
    fig.suptitle(f"Held-out test materials (n={len(mats)}), oracle = 1.0 not shown; error bars: 95% bootstrap CI over materials")
    fig.tight_layout()
    fig.savefig(out_dir / "test_results.png", dpi=160)
    plt.close(fig)

    with open(out_dir / "summary.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    # per material: shows whether a gain is broad or carried by one or two materials
    mids = sorted(mats, key=lambda m: mats[m]["name"])
    width = 0.8 / len(names)
    fig, ax = plt.subplots(figsize=(max(10, 1.4 * len(mids)), 4.5))
    for i, name in enumerate(names):
        y = [methods[name].get(m, (np.nan,))[0] for m in mids]
        ax.bar(np.arange(len(mids)) + (i - (len(names) - 1) / 2) * width, y, width, label=name, color=colors[i])
    ax.set_xticks(np.arange(len(mids)), [f"{mats[m]['name']}\n{m}" for m in mids], fontsize=8)
    ax.set(ylabel="mean reward", ylim=(0, 1.05), title="Test reward per material (oracle = 1.0)")
    ax.grid(axis="y", alpha=0.3)
    ax.legend(fontsize=8, ncol=min(len(names), 4))
    fig.tight_layout()
    fig.savefig(out_dir / "test_per_material.png", dpi=160)
    plt.close(fig)
    return rows


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--logs", type=Path, default=ROOT / "logs", help="dir holding grpo-active-space-candidates-*/")
    p.add_argument("--results", type=Path, default=ROOT / "results", help="dir holding candidates_*_test.jsonl")
    p.add_argument("--out", type=Path, default=ROOT / "results" / "plots")
    p.add_argument("--tol", type=float, default=0.05, help="near-oracle tolerance, as in scripts/candidates/evaluate.py")
    p.add_argument("--ema", type=float, default=0.3, help="smoothing weight of the newest step")
    args = p.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    scores = json.loads(SCORES.read_text())
    mats = {mid: m for mid, m in scores["materials"].items() if m["split"] == "test"}

    runs = find_runs(args.logs)
    if runs:
        plot_training(runs, args.out / "training_curves.png", args.ema)
    else:
        print(f"no training metrics under {args.logs}/grpo-active-space-candidates-*/ (run `bash brev/run_on_brev.sh pull`)")

    methods = collect_methods(args.results, mats, args.tol)
    rows = plot_test(methods, mats, args.out, args.tol)
    print(f"\n{'method':<30} {'reward':>7}  {'95% CI':>15}  {'near-oracle':>11}")
    for r in rows:
        print(f"{r['method']:<30} {r['mean_reward']:7.3f}  [{r['ci_low']:.3f}, {r['ci_high']:.3f}]  {r['near_oracle_rate']:11.3f}")
    print(f"\nwrote {', '.join(sorted(f.name for f in args.out.iterdir()))} to {args.out}")


if __name__ == "__main__":
    main()
