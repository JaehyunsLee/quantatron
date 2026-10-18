"""Steps 5-6: combine accuracy and cost into scores, mark the oracle, split by family and write
the two files the environment loads:

* orbitals.json  - what the agent may see (structure, orbitals, menu, localized summary)
* scores.json    - what only verify() and the evaluation read (scores, rewards, oracle, baselines)
"""

import json
import math
from collections import Counter
from pathlib import Path

import numpy as np
from pymatgen.core import Structure
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer

from .families import family_of, split_of

METRIC_KEY = {"mp2_pair": "acc_mp2_pair", "no_proxy": "acc_no_proxy", "casci": "acc_casci"}


def load_work(work_dir: Path) -> dict[str, dict]:
    return {p.stem: json.loads(p.read_text()) for p in sorted(Path(work_dir).glob("mp-*.json"))}


def candidate_scores(work: dict, metric: str, beta: float) -> dict[str, float]:
    key = METRIC_KEY[metric]
    return {
        cid: c[key] - beta * math.log10(c["toffolis"])
        for cid, c in work["candidates"].items()
        if c.get(key) is not None
    }


def oracle_of(scores: dict[str, float]) -> str:
    # Ties go to the cheaper (smaller) active space.
    return max(scores, key=lambda cid: (round(scores[cid], 12), -int(cid.split(":")[1])))


def normalized(scores: dict[str, float]) -> dict[str, float]:
    best, worst = max(scores.values()), min(scores.values())
    span = best - worst
    return {cid: (s - worst) / span if span > 0 else 1.0 for cid, s in scores.items()}


def auto_beta(works: list[dict], metric: str, grid=np.logspace(-3, 0, 61)) -> tuple[float, list]:
    """Pick beta so the oracle active-space size varies as much as possible across materials
    (max entropy of the oracle size distribution). A beta where every oracle is the smallest or the
    largest space makes the task trivial; this keeps the trade-off informative."""
    report = []
    for beta in grid:
        sizes = Counter(int(oracle_of(candidate_scores(w, metric, beta)).split(":")[1]) for w in works)
        p = np.array(list(sizes.values()), dtype=float) / len(works)
        report.append((float(beta), float(-(p * np.log(p)).sum()), dict(sorted(sizes.items()))))
    best = max(report, key=lambda r: (round(r[1], 6), -r[0]))
    return best[0], report


def structure_summary(cif: Path) -> dict:
    s = Structure.from_file(cif)
    try:
        sg = SpacegroupAnalyzer(s, symprec=0.1).get_space_group_symbol()
    except Exception:
        sg = "unknown"
    return {
        "formula": s.composition.reduced_formula,
        "spacegroup": sg,
        "natom": len(s),
        "abc": [round(x, 4) for x in s.lattice.abc],
        "angles": [round(x, 2) for x in s.lattice.angles],
        "lattice_vectors": np.round(s.lattice.matrix, 5).tolist(),
        "sites": [{"element": site.specie.symbol, "frac": np.round(site.frac_coords, 5).tolist()} for site in s],
        "density_g_cm3": round(float(s.density), 3),
    }


def build(cfg: dict, beta_override: float | None = None) -> dict:
    paths = cfg["paths"]
    phasecraft = {r["material_id"]: r for r in json.loads((paths["raw_dir"] / "phasecraft.json").read_text())}
    works = load_work(paths["work_dir"])
    metric = cfg["accuracy"]["metric"]
    split_cfg = cfg["split"]

    splits = {mid: split_of(w["name"], split_cfg["test_families"], split_cfg["validation_families"]) for mid, w in works.items()}
    beta_cfg = beta_override if beta_override is not None else cfg["score"]["beta"]
    beta_report = None
    if beta_cfg == "auto":
        train = [w for mid, w in works.items() if splits[mid] == "train"] or list(works.values())
        beta, beta_report = auto_beta(train, metric)
    else:
        beta = float(beta_cfg)

    orbitals_out, scores_out = {}, {}
    for mid, w in works.items():
        rec = phasecraft[mid]
        scores = candidate_scores(w, metric, beta)
        rewards = normalized(scores)
        oracle = oracle_of(scores)
        key = METRIC_KEY[metric]

        menu: dict[str, list[int]] = {}
        for cid in scores:
            sel, n = cid.split(":")
            menu.setdefault(sel, []).append(int(n))
        menu = {sel: sorted(ns) for sel, ns in menu.items()}

        meta = w["meta"]
        basis = sorted(set(meta["basis"].values()))
        orbitals_out[mid] = {
            "material_id": mid,
            "name": w["name"],
            "structure": structure_summary(paths["raw_dir"] / rec["cif"]),
            "description": rec.get("description"),
            "calc": {
                "basis_label": "/".join(basis) + f", {meta['pseudo']}",
                "supercell": meta["supercell"],
                "n_cells": meta["n_cells"],
                "natom_supercell": meta["natom_supercell"],
                "nelectron": meta["nelectron"],
                "gap_eV": round(meta["homo_lumo_gap_eV"], 3),
            },
            "orbitals": w["orbitals"],
            "localized_counts": w["localized_counts"],
            "localized_summary": ", ".join(
                f"{k} {v['occ']}/{v['vir']}" for k, v in sorted(w["localized_counts"].items(), key=lambda kv: -sum(kv[1].values()))
            ),
            "menu": menu,
            "members": {
                cid: {"occ": [i - meta["window_occ"] for i in w["menu"][cid]["occ"]],
                      "vir": [a - meta["window_occ"] for a in w["menu"][cid]["vir"]]}
                for cid in scores
            },
        }
        scores_out[mid] = {
            "name": w["name"],
            "family": family_of(w["name"]),
            "split": splits[mid],
            "oracle": oracle,
            "best_score": scores[oracle],
            "worst_score": min(scores.values()),
            "candidates": {
                cid: {
                    "accuracy": w["candidates"][cid][key],
                    "toffolis": w["candidates"][cid]["toffolis"],
                    "logical_qubits": w["candidates"][cid]["logical_qubits"],
                    "lambda": w["candidates"][cid]["lambda"],
                    "score": scores[cid],
                    "reward": rewards[cid],
                }
                for cid in scores
            },
            "baselines": {k: v["choice"] for k, v in w["baselines"].items()},
            "phasecraft": {
                "n_wannier": rec.get("n_wannier"),
                "n_cells": meta["n_cells"],
                "phasecraft_n_qubits": rec.get("phasecraft_n_qubits"),
                "id_note": rec.get("id_note"),
            },
        }

    settings = {"metric": metric, "beta": beta, "near_oracle_tol": cfg["score"]["near_oracle_tol"],
                "beta_mode": "auto" if beta_cfg == "auto" else "fixed"}
    return {"orbitals": orbitals_out, "scores": {"settings": settings, "materials": scores_out},
            "beta_report": beta_report}
