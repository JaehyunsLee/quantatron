"""Steps 2-4 for one material: mean field, localization, candidate menu, accuracy and cost.

All candidates are subsets of the canonical RHF orbitals inside a frontier window, so a single
window ERI tensor (<= 60 orbitals) serves every candidate:

* accuracy: MP2 pair energies restricted to the active occupied/virtual sets, divided by the
  full-system MP2 correlation energy (the "correlation energy recovered"); a natural-orbital
  proxy and optional CASCI are stored alongside.
* cost: the active-space Hamiltonian (frozen-core h1 + active ERIs) goes to cost.estimate().
"""

import json
import math
import time
from pathlib import Path

import numpy as np
from pymatgen.core import Structure
from pyscf import lo
from pyscf.pbc import gto as pgto
from pyscf.pbc import mp as pmp
from pyscf.pbc import scf as pscf
from pyscf.pbc import tools as ptools

from . import cost as costing

HARTREE_TO_EV = 27.211386


# --------------------------------------------------------------------------- step 2: mean field
def _load_cp2k(path: Path, element: str, name: str) -> list:
    """Minimal CP2K basis reader (PySCF's parser mishandles sets with several l, e.g. La in MOLOPT_UCL)."""
    lines = [l.split("#")[0].strip() for l in path.read_text().splitlines()]
    for i, l in enumerate(lines):
        tok = l.split()
        if len(tok) >= 2 and tok[0] == element and name.upper() in (t.upper() for t in tok[1:]):
            break
    else:
        raise KeyError(f"{element} {name} not in {path.name}")
    pos, shells = i + 2, []
    for _ in range(int(lines[i + 1])):
        head = [int(x) for x in lines[pos].split()]
        lmin, lmax, nexp, nshell = head[1], head[2], head[3], head[4:]
        rows = [[float(x) for x in lines[pos + 1 + k].split()] for k in range(nexp)]
        col = 1
        for l, n in zip(range(lmin, lmax + 1), nshell):
            for _ in range(n):
                shells.append([l] + [[r[0], r[col]] for r in rows])
                col += 1
        pos += 1 + nexp
    return shells


def _basis_for(element: str, basis: str, fallback: str):
    for name in (basis, fallback):
        try:
            pgto.basis.load(name, element)
            return name
        except Exception:
            continue
    # Lanthanides are only in the UCL MOLOPT file that ships with PySCF.
    ucl = Path(pgto.basis.__file__).parent / "BASIS_MOLOPT_UCL"
    for name in ("DZVP-MOLOPT-SR-GTH",):
        try:
            return _load_cp2k(ucl, element, name)
        except (KeyError, FileNotFoundError):
            continue
    raise RuntimeError(f"no GTH basis for {element}")


def build_cell(cif_path: Path, cfg: dict, min_occ: int) -> tuple[pgto.Cell, dict]:
    """Cell with the fixed basis/pseudo. The configured supercell is doubled along the shortest
    axis until the electron count is even (closed-shell RHF) and there are >= min_occ occupied
    orbitals (so every menu size fits). The rule is deterministic, so it is still a fixed protocol."""
    s = Structure.from_file(cif_path)
    elements = sorted({sp.symbol for sp in s.species})
    basis = {el: _basis_for(el, cfg["basis"], cfg["fallback_basis"]) for el in elements}

    cell = pgto.Cell()
    cell.atom = [(site.specie.symbol, site.coords.tolist()) for site in s]
    cell.a = s.lattice.matrix
    cell.unit = "Angstrom"
    cell.basis = basis
    cell.pseudo = cfg["pseudo"]
    cell.verbose = 0
    cell.build()

    mult = list(cfg["supercell"])
    lengths = np.array(s.lattice.abc)
    while True:
        sc = ptools.super_cell(cell, mult) if mult != [1, 1, 1] else cell
        if sc.nelectron % 2 == 0 and sc.nelectron // 2 >= min_occ:
            break
        axis = int(np.argmin(lengths * np.array(mult)))
        mult[axis] *= 2
    sc.verbose = 0
    meta = {
        "basis": {el: b if isinstance(b, str) else "dzvp-molopt-sr (UCL)" for el, b in basis.items()},
        "pseudo": cfg["pseudo"],
        "supercell": mult,
        "n_cells": int(np.prod(mult)),
        "natom_supercell": sc.natm,
        "nelectron": int(sc.nelectron),
        "nao": int(sc.nao_nr()),
    }
    return sc, meta


def run_scf(cell: pgto.Cell, cfg: dict):
    mf = pscf.RHF(cell)
    # Range-separated GDF builds the 3-center integrals several times faster than plain GDF.
    mf = mf.rs_density_fit() if cfg["density_fit"] == "rsgdf" else mf.density_fit()
    mf.conv_tol = cfg["conv_tol"]
    mf.max_cycle = cfg["max_cycle"]
    mf.level_shift = cfg["level_shift"]
    mf.damp = cfg["damp"]
    # PySCF's MINAO guess miscounts some GTH cores (e.g. Y); fall back to atomic, then core-Hamiltonian guesses.
    for guess in ("minao", "atom", "1e"):
        try:
            mf.init_guess = guess
            mf.kernel()
            break
        except (IndexError, ValueError, KeyError):
            continue
    if not mf.converged:
        mf = mf.newton()
        mf.kernel()
    return mf


# --------------------------------------------------------------------------- atomic character
def orth_ao(cell, S):
    """meta-Lowdin AOs; plain Lowdin when PySCF's ANO projection fails for a GTH core (e.g. Y)."""
    try:
        return lo.orth_ao(cell, "meta_lowdin", s=S)
    except ValueError:
        return lo.orth.lowdin(S)  # symmetric S^-1/2, no atomic pre-orthogonalization


def orbital_character(cell, mf, C: np.ndarray) -> tuple[np.ndarray, list[str]]:
    """Weights (n_shell_groups x n_orb) of each MO on '<El>-<nl>' groups, from meta-Lowdin AOs."""
    S = cell.pbc_intor("int1e_ovlp")
    X = orth_ao(cell, S).T @ S @ C  # MO coefficients in the orthonormal AO basis
    labels = cell.ao_labels(fmt=False)
    keys = sorted({f"{el}-{nl}" for _, el, nl, _ in labels})
    idx = {k: i for i, k in enumerate(keys)}
    W = np.zeros((len(keys), C.shape[1]))
    for mu, (_, el, nl, _) in enumerate(labels):
        W[idx[f"{el}-{nl}"]] += X[mu] ** 2
    W /= W.sum(axis=0, keepdims=True)
    return W, keys


def top_character(w: np.ndarray, keys: list[str], k: int = 3, floor: float = 0.05) -> dict[str, float]:
    order = np.argsort(-w)[:k]
    return {keys[i]: round(float(w[i]), 3) for i in order if w[i] >= floor}


# --------------------------------------------------------------------------- step 2b: localization
def localize(cell, C: np.ndarray, method: str) -> np.ndarray:
    if C.shape[1] == 0:
        return C
    if method == "pm":
        loc = lo.PM(cell, C)
        loc.pop_method = "meta_lowdin"
    elif method == "boys":
        loc = lo.Boys(cell, C)
    else:  # ibo needs IAOs from an all-electron minimal basis; with GTH cores PM is the safe equivalent
        loc = lo.PM(cell, C)
        loc.pop_method = "meta_lowdin"
    loc.verbose = 0
    try:
        return loc.kernel()
    except ValueError:  # meta-Lowdin populations unavailable for this cell (see orth_ao)
        loc = lo.PM(cell, C)
        loc.pop_method = "mulliken"
        loc.init_guess = None  # the atomic guess hits the same ANO projection; start from C itself
        loc.verbose = 0
        return loc.kernel()


# --------------------------------------------------------------------------- step 3: candidate menu
def is_open_shell(key: str) -> bool:
    """'<El>-<nl>' is a transition-metal d or lanthanide/actinide f shell."""
    from pymatgen.core import Element

    el, nl = key.split("-")
    e = Element(el)
    return (nl.endswith("d") and e.is_transition_metal) or (nl.endswith("f") and (e.is_lanthanoid or e.is_actinoid))


def make_menu(W_win: np.ndarray, keys: list[str], occ_idx, vir_idx, energies, sizes, n_targets):
    """Fixed menu: selection rule x size. Every candidate is a balanced CAS(n e, n o)."""
    n_occ, n_vir = len(occ_idx), len(vir_idx)
    # Character targets: the element shells carrying the most weight among the 10+10 frontier orbitals.
    frontier = list(range(max(0, n_occ - 10), n_occ)) + list(range(n_occ, min(n_occ + 10, n_occ + n_vir)))
    weight = W_win[:, frontier].sum(axis=1)
    targets = [keys[i] for i in np.argsort(-weight)[:n_targets]]
    # Open d/f shells are always on the menu, even when O-2p etc. dominate the frontier weight.
    targets += [k for k in keys if is_open_shell(k) and k not in targets]

    gap_dist = np.abs(energies - 0.5 * (energies[n_occ - 1] + energies[n_occ]))
    menu = {}
    for sel in ["frontier"] + targets:
        if sel == "frontier":
            occ_rank = list(range(n_occ - 1, -1, -1))
            vir_rank = list(range(n_occ, n_occ + n_vir))
        else:
            t = keys.index(sel)
            occ_rank = sorted(range(n_occ), key=lambda i: (-round(W_win[t, i], 3), gap_dist[i]))
            vir_rank = sorted(range(n_occ, n_occ + n_vir), key=lambda a: (-round(W_win[t, a], 3), gap_dist[a]))
        for n in sizes:
            if n // 2 > n_occ or n // 2 > n_vir:
                continue
            menu[f"{sel}:{n}"] = {
                "selection": sel,
                "n_orbitals": n,
                "n_electrons": n,
                "occ": sorted(occ_rank[: n // 2]),
                "vir": sorted(vir_rank[: n // 2]),
            }
    return menu, targets


# --------------------------------------------------------------------------- step 4a: accuracy
def window_mp2(eri_w: np.ndarray, e_w: np.ndarray, n_occ: int):
    """Pair-energy tensor E[i,j,a,b] (window indices) whose restricted sums give active-space MP2,
    plus the window MP2 one-particle density matrix for natural-orbital diagnostics."""
    o, v = slice(0, n_occ), slice(n_occ, len(e_w))
    ovov = eri_w[o, v, o, v]  # (ia|jb)
    eo, ev = e_w[o], e_w[v]
    denom = eo[:, None, None, None] + eo[None, None, :, None] - ev[None, :, None, None] - ev[None, None, None, :]
    t2 = ovov / denom  # t[i,a,j,b]
    pair = ovov * (2 * t2 - t2.transpose(0, 3, 2, 1))  # E[i,a,j,b]
    # MP2 unrelaxed 1-RDM blocks (spin-summed).
    tt = 2 * t2 - t2.transpose(0, 3, 2, 1)
    d_oo = -2 * np.einsum("iakb,jakb->ij", t2, tt)
    d_vv = 2 * np.einsum("iajc,ibjc->ab", t2, tt)
    dm = np.zeros((len(e_w), len(e_w)))
    dm[o, o] = 2 * np.eye(n_occ) + d_oo
    dm[v, v] = d_vv
    return pair, dm


def active_integrals(eri_w, fock_w, occ, vir):
    """Frozen-core active Hamiltonian in chemist notation: h1_eff = F_A - sum_{i in A_occ} [2(pq|ii) - (pi|iq)]."""
    A = list(occ) + list(vir)
    eri = eri_w[np.ix_(A, A, A, A)]
    o = list(range(len(occ)))
    j = np.einsum("pqii->pq", eri[:, :, o][:, :, :, o])
    k = np.einsum("piiq->pq", eri[:, o][:, :, o])
    h1 = fock_w[np.ix_(A, A)] - (2 * j - k)
    return h1, eri


def casci_corr(h1, eri, n_occ_act):
    from pyscf import fci

    n = h1.shape[0]
    e_cas, _ = fci.direct_spin1.kernel(h1, eri, n, (n_occ_act, n_occ_act))
    o = list(range(n_occ_act))
    e_hf = 2 * np.trace(h1[np.ix_(o, o)]) + 2 * np.einsum("iijj->", eri[np.ix_(o, o, o, o)]) - np.einsum(
        "ijji->", eri[np.ix_(o, o, o, o)]
    )
    return float(e_cas - e_hf)


# --------------------------------------------------------------------------- baselines (classical)
def avas_size(W_win: np.ndarray, X_win_target: np.ndarray, n_occ: int, threshold: float = 0.2) -> int:
    """AVAS-style count: eigenvalues > threshold of the occupied/virtual projectors onto the target AO shell."""
    occ_ev = np.linalg.eigvalsh(X_win_target[:, :n_occ].T @ X_win_target[:, :n_occ])
    vir_ev = np.linalg.eigvalsh(X_win_target[:, n_occ:].T @ X_win_target[:, n_occ:])
    return int((occ_ev > threshold).sum() + (vir_ev > threshold).sum())


def _avas_target(targets: list[str], keys: list[str]) -> str:
    """AVAS targets the open valence shell (transition-metal d or lanthanide f, e.g. 'V-3d') if present,
    otherwise the dominant frontier shell. Polarization d functions on main-group atoms are not targets."""
    return next((k for k in targets if is_open_shell(k)), targets[0])


def nearest_menu_entry(menu: dict, selection: str, n: int) -> str:
    sizes = sorted(c["n_orbitals"] for c in menu.values() if c["selection"] == selection)
    best = min(sizes, key=lambda s: (abs(s - n), s))
    return f"{selection}:{best}"


# --------------------------------------------------------------------------- driver
def run_material(rec: dict, cfg: dict, raw_dir: Path, out_dir: Path) -> dict:
    t0 = time.time()
    sizes = cfg["menu"]["sizes"]
    min_occ = max(sizes) // 2
    cell, meta = build_cell(raw_dir / rec["cif"], cfg["scf"], min_occ)
    if meta["natom_supercell"] > cfg["scf"]["max_atoms"]:
        raise RuntimeError(f"supercell has {meta['natom_supercell']} atoms > max_atoms")

    mf = run_scf(cell, cfg["scf"])
    e_mo, C, occ = mf.mo_energy, mf.mo_coeff, mf.mo_occ
    nocc = int((occ > 0).sum())
    nmo = C.shape[1]

    # Frontier window exposed to the agent.
    w_occ = min(cfg["orbitals"]["window_occ"], nocc)
    w_vir = min(cfg["orbitals"]["window_vir"], nmo - nocc)
    win = list(range(nocc - w_occ, nocc + w_vir))
    C_w, e_w = C[:, win], e_mo[win]

    W_all, keys = orbital_character(cell, mf, C_w)

    # Localized orbitals of the window (occupied and virtual separately) and their dominant character.
    loc_counts: dict[str, dict[str, int]] = {}
    for part, cols in (("occ", slice(0, w_occ)), ("vir", slice(w_occ, None))):
        C_loc = localize(cell, C_w[:, cols], cfg["localization"]["method"])
        W_loc, _ = orbital_character(cell, mf, C_loc)
        for i in range(W_loc.shape[1]):
            key = keys[int(np.argmax(W_loc[:, i]))]
            loc_counts.setdefault(key, {"occ": 0, "vir": 0})[part] += 1

    # Window integrals and Fock matrix in the MO basis.
    eri_w = mf.with_df.ao2mo(C_w, compact=False).reshape(len(win), len(win), len(win), len(win))
    fock_w = C_w.T @ mf.get_fock() @ C_w

    # Total correlation energy (denominator of the accuracy).
    mp2 = pmp.RMP2(mf)
    mp2.verbose = 0
    e_corr_total, _ = mp2.kernel(with_t2=False)
    pair, dm_w = window_mp2(eri_w, e_w, w_occ)
    no_occ = np.linalg.eigvalsh(dm_w)
    corr_weight = np.concatenate([2 - np.diag(dm_w)[:w_occ], np.diag(dm_w)[w_occ:]])

    menu, targets = make_menu(W_all, keys, list(range(w_occ)), list(range(w_occ, len(win))), e_w, sizes,
                              cfg["orbitals"]["n_character_targets"])

    candidates = {}
    for cid, cand in menu.items():
        o_idx, v_idx = cand["occ"], [a - w_occ for a in cand["vir"]]
        e_act = float(pair[np.ix_(o_idx, v_idx, o_idx, v_idx)].sum())
        h1, eri = active_integrals(eri_w, fock_w, cand["occ"], cand["vir"])
        row = {
            "acc_mp2_pair": e_act / e_corr_total,
            "acc_no_proxy": float(corr_weight[cand["occ"] + cand["vir"]].sum() / corr_weight.sum()),
            "acc_casci": None,
            **costing.estimate(h1, eri, cfg["cost"]),
        }
        if cand["n_orbitals"] <= cfg["accuracy"]["casci_max_orb"]:
            row["acc_casci"] = casci_corr(h1, eri, len(cand["occ"])) / e_corr_total
        candidates[cid] = row

    # Classical baselines, mapped onto the menu.
    S = cell.pbc_intor("int1e_ovlp")
    X = orth_ao(cell, S).T @ S @ C_w
    labels = [f"{el}-{nl}" for _, el, nl, _ in cell.ao_labels(fmt=False)]
    avas_target = _avas_target(targets, keys)
    rows = [i for i, l in enumerate(labels) if l == avas_target]
    n_avas = avas_size(W_all, X[rows], w_occ)
    n_no = int(((no_occ > 0.02) & (no_occ < 1.98)).sum())
    baselines = {
        "avas": {"target": avas_target, "n_raw": n_avas, "choice": nearest_menu_entry(menu, avas_target, n_avas)},
        "no_threshold": {"n_raw": n_no, "choice": nearest_menu_entry(menu, "frontier", n_no)},
    }

    homo = e_w[w_occ - 1]
    orbitals = [
        {
            "index": i - w_occ,  # -1 = HOMO, 0 = LUMO
            "energy_eV": round(float((e_w[i] - homo) * HARTREE_TO_EV), 3),
            "occupation": 2 if i < w_occ else 0,
            "mp2_occupation": round(float(np.diag(dm_w)[i]), 4),
            "character": top_character(W_all[:, i], keys),
        }
        for i in range(len(win))
    ]

    result = {
        "material_id": rec["material_id"],
        "name": rec["name"],
        "meta": {**meta, "scf_converged": bool(mf.converged), "e_hf": float(mf.e_tot),
                 "e_corr_mp2": float(e_corr_total), "homo_lumo_gap_eV": float((e_w[w_occ] - homo) * HARTREE_TO_EV),
                 "window_occ": w_occ, "window_vir": w_vir, "wall_time_s": round(time.time() - t0, 1)},
        "character_keys": keys,
        "targets": targets,
        "localized_counts": loc_counts,
        "orbitals": orbitals,
        "menu": {cid: {k: v for k, v in c.items()} for cid, c in menu.items()},
        "candidates": candidates,
        "baselines": baselines,
        "natural_occupations": [round(float(x), 5) for x in sorted(no_occ, reverse=True)],
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f"{rec['material_id']}.json").write_text(json.dumps(result, indent=1))
    return result
