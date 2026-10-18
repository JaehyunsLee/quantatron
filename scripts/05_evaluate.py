"""Step 12: evaluate on held-out materials.

For each rollout file (NeMo Gym output of scripts/run_eval.sh) and for the classical baselines
(AVAS-style projection, MP2 natural-orbital threshold), report on one split:
  * mean reward (normalized score, 0..1)
  * near-oracle rate: fraction of answers with reward >= 1 - tol (default: within 5% of the oracle)
  * valid-answer rate (LLM runs only)
  * agreement with Phasecraft's Wannier orbital counts: Spearman rho and MAE between the chosen
    number of active orbitals per primitive cell and Phasecraft's n_wannier, per material.

    python scripts/05_evaluate.py results/base_test.jsonl results/grpo_test.jsonl [--split test]
"""

import argparse
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
SCORES = ROOT / "resources_servers" / "active_space" / "data" / "scores.json"


def spearman(x, y) -> float:
    if len(x) < 3:
        return float("nan")
    rx, ry = np.argsort(np.argsort(x)), np.argsort(np.argsort(y))
    return float(np.corrcoef(rx, ry)[0, 1])


def summarize(label: str, picks: dict[str, list[str | None]], mats: dict, tol: float) -> dict:
    """picks: material_id -> list of chosen candidate ids (None = invalid/no answer)."""
    rewards, near, valid = [], [], []
    per_cell, wannier = [], []
    for mid, choices in picks.items():
        m = mats[mid]
        rs = [m["candidates"][c]["reward"] if c in m["candidates"] else 0.0 for c in choices]
        rewards += rs
        near += [float(r >= 1 - tol) for r in rs]
        valid += [float(c in m["candidates"]) for c in choices]
        sizes = [int(c.split(":")[1]) for c in choices if c in m["candidates"]]
        if sizes and m["phasecraft"]["n_wannier"]:
            per_cell.append(float(np.mean(sizes)) / m["phasecraft"]["n_cells"])
            wannier.append(m["phasecraft"]["n_wannier"])
    return {
        "method": label,
        "n_materials": len(picks),
        "n_answers": len(rewards),
        "mean_reward": float(np.mean(rewards)) if rewards else float("nan"),
        "near_oracle_rate": float(np.mean(near)) if near else float("nan"),
        "valid_rate": float(np.mean(valid)) if valid else float("nan"),
        "wannier_spearman": spearman(per_cell, wannier),
        "wannier_mae": float(np.mean(np.abs(np.array(per_cell) - np.array(wannier)))) if per_cell else float("nan"),
    }


def load_rollouts(path: Path) -> dict[str, list[str | None]]:
    picks = defaultdict(list)
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        mid = row.get("material_id") or row.get("task_id")
        picks[mid].append(row.get("chosen"))
    return picks


def main():
    p = argparse.ArgumentParser()
    p.add_argument("rollouts", nargs="*", type=Path)
    p.add_argument("--split", default="test")
    p.add_argument("--json-out", type=Path, default=None)
    args = p.parse_args()

    scores = json.loads(SCORES.read_text())
    tol = scores["settings"]["near_oracle_tol"]
    mats = {mid: m for mid, m in scores["materials"].items() if m["split"] == args.split}
    if not mats:
        sys.exit(f"no materials in split {args.split!r}")

    rows = [summarize("oracle", {mid: [m["oracle"]] for mid, m in mats.items()}, mats, tol)]
    for name in ("avas", "no_threshold"):
        rows.append(summarize(name, {mid: [m["baselines"][name]] for mid, m in mats.items()}, mats, tol))
    # Best single fixed menu entry chosen on the *train* split (a strong "no-reasoning" baseline).
    train = [m for m in scores["materials"].values() if m["split"] == "train"]
    fixed = [c for c in {c for m in train for c in m["candidates"]} if c.startswith("frontier:")]
    if train and fixed:
        best_fixed = max(fixed, key=lambda c: np.mean([m["candidates"].get(c, {"reward": 0})["reward"] for m in train]))
        rows.append(summarize(f"fixed {best_fixed} (best on train)", {mid: [best_fixed] for mid in mats}, mats, tol))
    rng = random.Random(0)
    rows.append(summarize("random menu entry", {mid: [rng.choice(list(m["candidates"])) for _ in range(64)]
                                                 for mid, m in mats.items()}, mats, tol))
    for path in args.rollouts:
        picks = {mid: v for mid, v in load_rollouts(path).items() if mid in mats}
        rows.append(summarize(path.stem, picks, mats, tol))

    print(f"split={args.split}  materials={len(mats)}  metric={scores['settings']['metric']}  "
          f"beta={scores['settings']['beta']:.4g}  near-oracle tol={tol}")
    hdr = f"| {'method':<38} | {'reward':>6} | {'<=5% of oracle':>14} | {'valid':>5} | {'Wannier rho':>11} | {'Wannier MAE':>11} |"
    print(hdr + "\n|" + "|".join("-" * (len(c)) for c in hdr.split("|")[1:-1]) + "|")
    for r in rows:
        print(f"| {r['method']:<38} | {r['mean_reward']:6.3f} | {r['near_oracle_rate']:14.3f} | {r['valid_rate']:5.2f} | "
              f"{r['wannier_spearman']:11.3f} | {r['wannier_mae']:11.2f} |")
    if args.json_out:
        args.json_out.write_text(json.dumps(rows, indent=1))


if __name__ == "__main__":
    main()
