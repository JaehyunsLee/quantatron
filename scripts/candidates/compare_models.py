"""Head-to-head on the held-out test materials: random menu entry vs. GRPO fine-tuned Nemotron 3.5 Lightning vs.
GRPO fine-tuned Qwen3 (same data, GRPO settings, LoRA rank and scorer as Lightning).

The rollouts come from the Brev pipeline (`bash brev/run_on_brev.sh detach wait evaltrained "MODEL=qwen3-1.7b all"`,
then `pull`). This script only reads them; it prints a table and saves a bar chart:

    python scripts/candidates/compare_models.py                       # Qwen3-1.7B (default)
    python scripts/candidates/compare_models.py --qwen qwen3-4b       # another Qwen size you ran with MODEL=qwen3-4b
    python scripts/candidates/compare_models.py --with-base           # also the untrained Lightning / Qwen rows

Needs numpy and matplotlib (e.g. `uv run --with matplotlib --with numpy python scripts/candidates/compare_models.py`).
"""

import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))
from evaluate import SCORES, load_rollouts  # noqa: E402  (same scorer as the printed table)


def per_material(picks: dict, mats: dict, tol: float) -> dict[str, tuple[float, float, float]]:
    """{material: (mean reward, near-oracle rate, valid rate)} over that material's answers."""
    out = {}
    for mid, choices in picks.items():
        cands = mats[mid]["candidates"]
        rs = np.array([cands[c]["reward"] if c in cands else 0.0 for c in choices])
        out[mid] = (rs.mean(), (rs >= 1 - tol).mean(), np.mean([c in cands for c in choices]))
    return out


def bootstrap_ci(values: np.ndarray, n: int = 5000, seed: int = 0) -> tuple[float, float]:
    """95% CI of the mean, resampling materials (the dominant source of variance with 8 test materials)."""
    means = np.random.default_rng(seed).choice(values, size=(n, len(values))).mean(axis=1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--results", type=Path, default=ROOT / "results")
    p.add_argument("--qwen", default="qwen3-1.7b", help="MODEL name the Qwen run used (qwen3-1.7b, qwen3-4b, qwen3-8b)")
    p.add_argument("--with-base", action="store_true", help="add the untrained Lightning and Qwen rows")
    p.add_argument("--tol", type=float, default=0.05)
    p.add_argument("--out", type=Path, default=ROOT / "results" / "plots" / "compare_models.png")
    args = p.parse_args()

    scores = json.loads(SCORES.read_text())
    mats = {mid: m for mid, m in scores["materials"].items() if m["split"] == "test"}
    qwen_name = args.qwen.replace("qwen3-", "Qwen3-").replace("b", "B")

    # Same seeded random baseline as scripts/candidates/evaluate.py, so the numbers match its table.
    rng = random.Random(0)
    methods = {"Random menu entry": {mid: [rng.choice(list(m["candidates"])) for _ in range(64)] for mid, m in mats.items()}}
    files = {"Lightning GRPO (fine-tuned)": "candidates_grpo_test.jsonl",
             f"{qwen_name} GRPO (fine-tuned)": f"candidates_{args.qwen}_grpo_test.jsonl"}
    if args.with_base:
        files = {"Lightning base": "candidates_base_test.jsonl", **files, f"{qwen_name} base": f"candidates_{args.qwen}_base_test.jsonl"}
    missing = []
    for label, name in files.items():
        path = args.results / name
        if path.exists():
            methods[label] = {mid: v for mid, v in load_rollouts(path).items() if mid in mats}
        else:
            missing.append(f"{label}: {path} not found")

    rows = []
    for label, picks in methods.items():
        pm = per_material(picks, mats, args.tol)
        rewards = np.array([v[0] for v in pm.values()])
        lo, hi = bootstrap_ci(rewards)
        rows.append({"method": label, "reward": rewards.mean(), "lo": lo, "hi": hi,
                     "near": np.mean([v[1] for v in pm.values()]), "valid": np.mean([v[2] for v in pm.values()]),
                     "n": sum(len(c) for c in picks.values())})

    print(f"held-out test materials: {len(mats)}   (oracle = 1.000 by definition)")
    print(f"| {'method':<30} | {'reward':>6} | {'95% CI':>15} | {'near-oracle':>11} | {'valid':>5} | {'answers':>7} |")
    print(f"|{'-' * 32}|{'-' * 8}|{'-' * 17}|{'-' * 13}|{'-' * 7}|{'-' * 9}|")
    for r in rows:
        print(f"| {r['method']:<30} | {r['reward']:6.3f} | [{r['lo']:.3f}, {r['hi']:.3f}] | {r['near']:11.3f} | "
              f"{r['valid']:5.2f} | {r['n']:7d} |")
    for m in missing:
        print(f"  (missing) {m}  -> wait for the Brev queue, then `bash brev/run_on_brev.sh pull`")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    colors = {"Random": "0.6", "Lightning": "C0", "Qwen3": "C1"}
    fig, ax = plt.subplots(figsize=(1.9 * len(rows) + 2, 4.5))
    x = np.arange(len(rows))
    bars = ax.bar(x, [r["reward"] for r in rows],
                  yerr=[[r["reward"] - r["lo"] for r in rows], [r["hi"] - r["reward"] for r in rows]], capsize=5,
                  color=[colors[r["method"].split()[0].split("-")[0]] for r in rows])
    for b, r in zip(bars, rows):
        if "base" in r["method"]:
            b.set_alpha(0.5)   # untrained rows lighter than their fine-tuned counterpart
    ax.bar_label(bars, labels=[f"{r['reward']:.3f}" for r in rows], padding=4, fontsize=9)
    ax.set_xticks(x, [r["method"].replace(" (", "\n(") for r in rows], fontsize=9)
    ax.set(ylabel="mean test reward (oracle = 1.0)", ylim=(0, 1),
           title=f"Held-out materials (n={len(mats)}): random vs. fine-tuned models\nerror bars: 95% bootstrap CI over materials")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=160)
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()
