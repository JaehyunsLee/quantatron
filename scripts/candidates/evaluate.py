"""Evaluate active_space_candidates rollouts on the held-out test materials.

For each rollout file (output of scripts/candidates/run_eval.sh), its majority vote, and no-model baselines (random
entry and two fixed rules that only read the agent-visible menu), report:
  * mean reward (normalized score, 0..1)
  * near-oracle rate: fraction of answers with reward >= 1 - tol (default 0.05)
  * valid-answer rate

    python scripts/candidates/evaluate.py results/candidates_base_test.jsonl [results/candidates_grpo_test.jsonl] [--tol 0.05]
"""

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path

import numpy as np

DATA = Path(__file__).resolve().parents[2] / "resources_servers" / "active_space_candidates" / "data"
SCORES = DATA / "scores.json"
ORBITALS = DATA / "orbitals.json"


def rule_picks(orbitals: dict, mats: dict) -> dict[str, dict[str, list[str]]]:
    """No-model rules over the agent-visible menu. A model has to beat these to show it learned anything."""
    def width(c):
        lo, hi = c["window_ev"]
        return hi - lo

    rules = {
        # at beta = 0.2 the accuracy term usually outweighs the cost term, so this simple rule is strong
        "rule: largest candidate": lambda menu: max(menu, key=lambda k: (menu[k]["n_orbitals"], -width(menu[k]))),
        "rule: tightest window/orbital": lambda menu: min(menu, key=lambda k: width(menu[k]) / menu[k]["n_orbitals"]),
    }
    return {name: {mid: [pick(orbitals[mid]["menu"])] for mid in mats} for name, pick in rules.items()}


def majority(picks: dict[str, list[str | None]]) -> dict[str, list[str | None]]:
    """One answer per material: the most frequent valid choice over its samples (first seen wins ties)."""
    out = {}
    for mid, choices in picks.items():
        valid = [c for c in choices if c]
        out[mid] = [max(valid, key=lambda c: (valid.count(c), -valid.index(c)))] if valid else [None]
    return out


def summarize(label: str, picks: dict[str, list[str | None]], mats: dict, tol: float) -> dict:
    rewards, near, valid = [], [], []
    for mid, choices in picks.items():
        m = mats[mid]
        rs = [m["candidates"][c]["reward"] if c in m["candidates"] else 0.0 for c in choices]
        rewards += rs
        near += [float(r >= 1 - tol) for r in rs]
        valid += [float(c in m["candidates"]) for c in choices]
    return {
        "method": label,
        "n_materials": len(picks),
        "n_answers": len(rewards),
        "mean_reward": float(np.mean(rewards)) if rewards else float("nan"),
        "near_oracle_rate": float(np.mean(near)) if near else float("nan"),
        "valid_rate": float(np.mean(valid)) if valid else float("nan"),
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
    p.add_argument("--tol", type=float, default=0.05)
    args = p.parse_args()

    scores = json.loads(SCORES.read_text())
    mats = {mid: m for mid, m in scores["materials"].items() if m["split"] == "test"}
    if not mats:
        raise SystemExit("no materials in the test split")

    rows = [summarize("oracle", {mid: [m["oracle"]] for mid, m in mats.items()}, mats, args.tol)]
    rng = random.Random(0)
    rows.append(summarize(
        "random menu entry",
        {mid: [rng.choice(list(m["candidates"])) for _ in range(64)] for mid, m in mats.items()},
        mats, args.tol,
    ))
    orbitals = json.loads(ORBITALS.read_text())
    rows += [summarize(name, picks, mats, args.tol) for name, picks in rule_picks(orbitals, mats).items()]
    for path in args.rollouts:
        picks = {mid: v for mid, v in load_rollouts(path).items() if mid in mats}
        rows.append(summarize(path.stem, picks, mats, args.tol))
        rows.append(summarize(f"{path.stem} (majority)", majority(picks), mats, args.tol))

    print(f"test materials={len(mats)}  near-oracle tol={args.tol}")
    w = max(len(r["method"]) for r in rows)
    hdr = f"| {'method':<{w}} | {'reward':>6} | {'<=5% of oracle':>14} | {'valid':>5} |"
    print(hdr + "\n|" + "|".join("-" * len(c) for c in hdr.split("|")[1:-1]) + "|")
    for r in rows:
        print(f"| {r['method']:<{w}} | {r['mean_reward']:6.3f} | {r['near_oracle_rate']:14.3f} | {r['valid_rate']:5.2f} |")


if __name__ == "__main__":
    main()
