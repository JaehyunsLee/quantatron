"""Candidate-level counterpart of scripts/03_build_table.py: builds the active_space_candidates Gym environment data
directly from a flat per-candidate CSV (data/phasecraft_tables/dzvp.csv) instead of from data/work*/*.json.

Input columns used (one row per candidate):
    material, mp_id, formula, n_atoms, gap_ev, e_scf, ref_cas, use_for_training,
    candidate_id, n_electrons, n_orbitals, mo_indices, window_ev_low, window_ev_high,
    character, corr_recovered, toffoli_total, logical_qubits, lambda_ha, score

Only rows with use_for_training == True are kept. The table's own flag drops 120 of 450 rows: windows holding
~0 correlation (|E_corr| < 1 mHa, so corr_recovered is noise; this removes CsPbF3, CsPbI3, LaFeO3, LaNiO3,
LaTiO3, PbHfO3, SrTiO3 entirely), imported duplicates of re-run materials, and YFeO3's wrong imported structure.
Rows missing an agent-visible field (gap, energy window, orbital character) are dropped too: Al2O3 and LiNiO2 survive
only as imported rows from the older code, which recorded none of these, so their menus would be bare orbital
counts. That leaves 24 materials. The oracle is the best-scoring kept row (identical to the table's is_oracle there).

Confirmed from the data itself: score = corr_recovered - 0.2 * log10(toffoli_total) for
every row (beta = 0.2 exactly, not "auto" like configs/phase1.yaml).

Writes, mirroring the active_space environment's split between agent-visible and verify-only data:
    resources_servers/active_space_candidates/data/orbitals.json   agent-visible: structure + candidate menu with its
                                                                  log10 Toffoli cost (NO accuracy/score/reward)
    resources_servers/active_space_candidates/data/scores.json     verify()-only: oracle, best/worst score, per-candidate reward
    resources_servers/active_space_candidates/data/train.jsonl     16 materials
    resources_servers/active_space_candidates/data/test.jsonl      8 materials (whole, disjoint families)

    python scripts/candidates/build_env_data.py   # [--csv data/phasecraft_tables/dzvp.csv]
"""

import argparse
import json
import math
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from active_space.prompts_candidates import SYSTEM_PROMPT, tool_schemas, user_prompt  # noqa: E402

AGENT_VISIBLE = ["gap_ev", "window_ev_low", "window_ev_high", "character"]
BETA = 0.2  # verified: (corr_recovered - score) / log10(toffoli_total) == 0.2 for all 450 rows

# Whole chemical families held out for test. The SZV split held out halide perovskites, TM binary oxides and ionic
# salts; the DZVP table has no usable halide perovskite, so test keeps the other two families, now with every TM
# binary oxide in the table. NiO2 stays in train with the other nickelates (LiNiO2, BaNiO3), as in the SZV split.
TEST_MATERIALS = {
    "HfO2", "PdO", "NbO2", "TaO2", "Nb2O5", "CuO",  # tm_binary_oxide
    "BaS", "NaCl",                                  # ionic_salt
}


def parse_character(raw: str) -> dict[str, float]:
    return json.loads(raw)


def build(df: pd.DataFrame) -> tuple[dict, dict]:
    orbitals_out: dict[str, dict] = {}
    scores_out: dict[str, dict] = {}

    df = df[df["use_for_training"]].dropna(subset=AGENT_VISIBLE)
    for mid, g in df.groupby("mp_id"):
        name = g["material"].iloc[0]
        split = "test" if name in TEST_MATERIALS else "train"

        menu = {}
        candidates = {}
        for _, row in g.iterrows():
            cid = f"c{int(row['candidate_id'])}:{int(row['n_orbitals'])}"
            menu[cid] = {
                "n_electrons": int(row["n_electrons"]),
                "n_orbitals": int(row["n_orbitals"]),
                "mo_indices": json.loads(row["mo_indices"]),
                "window_ev": [round(float(row["window_ev_low"]), 3), round(float(row["window_ev_high"]), 3)],
                "character": {k: round(v, 3) for k, v in parse_character(row["character"]).items() if v >= 0.03},
                # The QPE cost estimate is agent-visible: it is cheap next to the correlated calculation behind
                # `accuracy`, which stays verify-only. With it the agent can weigh the cost term of the score exactly.
                "log10_toffolis": round(math.log10(int(row["toffoli_total"])), 3),
            }
            candidates[cid] = {
                "accuracy": float(row["corr_recovered"]),
                "toffolis": int(row["toffoli_total"]),
                "logical_qubits": int(row["logical_qubits"]) if pd.notna(row["logical_qubits"]) else None,
                "lambda": float(row["lambda_ha"]),
                "score": float(row["score"]),
            }

        best = max(candidates.values(), key=lambda c: c["score"])["score"]
        worst = min(candidates.values(), key=lambda c: c["score"])["score"]
        span = best - worst
        oracle_cid = max(candidates, key=lambda cid: candidates[cid]["score"])
        for c in candidates.values():
            c["reward"] = (c["score"] - worst) / span if span > 0 else 1.0

        r0 = g.iloc[0]
        orbitals_out[mid] = {
            "material_id": mid,
            "name": name,
            "formula": r0["formula"],
            "n_atoms": int(r0["n_atoms"]),
            "gap_eV": round(float(r0["gap_ev"]), 3),
            "e_scf_ha": round(float(r0["e_scf"]), 6),
            "ref_cas": r0["ref_cas"],  # classical reference CAS, for context only
            "menu": menu,
        }
        scores_out[mid] = {
            "name": name,
            "split": split,
            "oracle": oracle_cid,
            "best_score": best,
            "worst_score": worst,
            "candidates": candidates,
        }

    return orbitals_out, scores_out


def gym_row(entry: dict) -> dict:
    return {
        "responses_create_params": {
            "input": [
                {"role": "system", "content": SYSTEM_PROMPT.format(beta=f"{BETA:.2g}")},
                {"role": "user", "content": user_prompt(entry)},
            ],
            "tools": tool_schemas(),
        },
        "agent_ref": {"type": "responses_api_agents", "name": "active_space_candidates_simple_agent"},
        "task_id": entry["material_id"],
        "material_id": entry["material_id"],
        "name": entry["name"],
    }


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--csv", default=str(ROOT / "data" / "phasecraft_tables" / "dzvp.csv"))
    p.add_argument("--out", default=str(ROOT / "resources_servers" / "active_space_candidates" / "data"))
    args = p.parse_args()

    df = pd.read_csv(args.csv)
    orbitals, scores = build(df)
    absent = TEST_MATERIALS - {s["name"] for s in scores.values()}
    assert not absent, f"test materials with no usable rows in {args.csv}: {sorted(absent)}"
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "orbitals.json").write_text(json.dumps(orbitals, indent=1))
    (out / "scores.json").write_text(json.dumps({"settings": {"beta": BETA}, "materials": scores}, indent=1))

    rows = {"train": [], "test": []}
    for mid, s in scores.items():
        rows[s["split"]].append(gym_row(orbitals[mid]))
    for split, rs in rows.items():
        with open(out / f"{split}.jsonl", "w") as f:
            f.writelines(json.dumps(r) + "\n" for r in rs)

    print(f"materials: {len(orbitals)}  train: {len(rows['train'])}  test: {len(rows['test'])}")
    print("train:", sorted(s["name"] for s in scores.values() if s["split"] == "train"))
    print("test: ", sorted(s["name"] for s in scores.values() if s["split"] == "test"))
    print(f"wrote {out}/orbitals.json, scores.json, train.jsonl, test.jsonl")
