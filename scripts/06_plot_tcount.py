"""Plot the fault-tolerant gate cost of the active space each baseline picks.

The cost model (active_space/cost.py) counts Toffolis (Toffoli + logical AND + CSWAP) for
double-factorized qubitized QPE. T count = --t-per-toffoli x Toffolis: 4 with Gidney's
measurement-based uncomputation (the usual convention for these estimates), 7 for a textbook
Clifford+T Toffoli.

Left: per material, the T count of the oracle, AVAS, MP2 natural-orbital threshold and best fixed
size (chosen on train), with the full menu range as a gray bar. Right: median over materials.

    python scripts/06_plot_tcount.py [--split test] [--t-per-toffoli 4] [--out results/tcount.png]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
SCORES = ROOT / "resources_servers" / "active_space" / "data" / "scores.json"

# Fixed categorical order (blue, orange, aqua, yellow); identity is also carried by marker shape.
METHODS = [("oracle", "#2a78d6", "o"), ("avas", "#eb6834", "s"), ("no_threshold", "#1baf7a", "^"),
           ("best fixed", "#eda100", "D")]
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"


def best_fixed(materials: dict) -> str | None:
    """Frontier menu entry with the highest mean reward on the train split (as in 05_evaluate.py)."""
    train = [m for m in materials.values() if m["split"] == "train"]
    fixed = {c for m in train for c in m["candidates"] if c.startswith("frontier:")}
    if not fixed:
        return None
    return max(fixed, key=lambda c: np.mean([m["candidates"].get(c, {"reward": 0})["reward"] for m in train]))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--split", default="all", help="train / validation / test / all")
    p.add_argument("--t-per-toffoli", type=float, default=4.0)
    p.add_argument("--out", type=Path, default=ROOT / "results" / "tcount.png")
    args = p.parse_args()

    scores = json.loads(SCORES.read_text())
    materials = scores["materials"]
    fixed = best_fixed(materials)
    mats = {mid: m for mid, m in materials.items() if args.split == "all" or m["split"] == args.split}
    k = args.t_per_toffoli

    def t_of(m, cid):
        c = m["candidates"].get(cid)
        return c["toffolis"] * k if c else np.nan

    picks = {
        "oracle": lambda m: m["oracle"],
        "avas": lambda m: m["baselines"]["avas"],
        "no_threshold": lambda m: m["baselines"]["no_threshold"],
        "best fixed": lambda m: fixed,
    }
    order = sorted(mats, key=lambda mid: t_of(mats[mid], mats[mid]["oracle"]))
    t = {name: np.array([t_of(mats[mid], picks[name](mats[mid])) for mid in order]) for name, _, _ in METHODS}

    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(14, 6), gridspec_kw={"width_ratios": [3.2, 1]})
    x = np.arange(len(order))
    lo = [min(c["toffolis"] for c in mats[mid]["candidates"].values()) * k for mid in order]
    hi = [max(c["toffolis"] for c in mats[mid]["candidates"].values()) * k for mid in order]
    ax.vlines(x, lo, hi, color=GRID, lw=6, zorder=1, label="menu range")
    for i, (name, color, marker) in enumerate(METHODS):
        label = f"best fixed ({fixed})" if name == "best fixed" else name
        ax.scatter(x + (i - 1.5) * 0.12, t[name], s=40, color=color, marker=marker, label=label,
                   edgecolors="white", linewidths=1, zorder=3)
    ax.set_yscale("log")
    ax.set_xticks(x, [mats[mid]["name"] for mid in order], rotation=90, fontsize=8)
    ax.set_ylabel(f"T count  (= {k:g} × Toffolis)")
    ax.set_title(f"T count of the chosen active space per material  (split: {args.split}, n={len(order)})",
                 loc="left", color=INK)
    ax.legend(frameon=False, fontsize=9, loc="upper left")

    names = [n for n, _, _ in METHODS]
    med = [np.nanmedian(t[n]) for n in names]
    ax2.barh(names[::-1], med[::-1], color=[c for _, c, _ in METHODS][::-1], height=0.6)
    ax2.set_xscale("log")
    for y, v in enumerate(med[::-1]):
        ax2.text(v, y, f" {v:.2e}", va="center", fontsize=9, color=INK)
    ax2.set_title("Median T count", loc="left", color=INK)

    for a in (ax, ax2):
        a.grid(axis="y" if a is ax else "x", color=GRID, lw=0.8)
        a.set_axisbelow(True)
        a.tick_params(colors=MUTED)
        for s in ("top", "right"):
            a.spines[s].set_visible(False)

    fig.tight_layout()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=150)
    print(f"wrote {args.out}")
    for n, v in zip(names, med):
        print(f"  {n:<14} median T = {v:.3e}")


if __name__ == "__main__":
    main()
