"""Step 1: the 46 Phasecraft materials and their Materials Project structures.

Phasecraft's database (materials.phasecraft.io) is a Gatsby site, so every page has a
machine-readable ``page-data.json``. Structures come from the Materials Project: through
``mp-api`` when ``MP_API_KEY`` is set, otherwise through the keyless OPTIMADE endpoint.
"""

import json
import os
import re
from pathlib import Path

import requests

PHASECRAFT = "https://materials.phasecraft.io"
OPTIMADE = "https://optimade.materialsproject.org/v1/structures"


def _get_json(url: str) -> dict:
    r = requests.get(url, timeout=60)
    r.raise_for_status()
    return r.json()


def _slug(name: str) -> str:
    return re.sub(r"[^a-z0-9]", "", name.lower())


def fetch_phasecraft() -> list[dict]:
    """One record per material: MP id, Wannier count, applications and Phasecraft circuit costs."""
    index = _get_json(f"{PHASECRAFT}/page-data/index/page-data.json")
    names = [m["name"] for m in index["result"]["data"]["materialData"]["materials"]]
    records = []
    for name in names:
        page = _get_json(f"{PHASECRAFT}/page-data/materials/{_slug(name)}/page-data.json")
        m = page["result"]["data"]["material"]
        info, ham, wan = m["material_info"], m.get("hamiltonian") or {}, m.get("wannier_functions") or {}
        records.append(
            {
                "name": name,
                "material_id": info["materials_project_id"],
                "natom": info.get("natom"),
                "spacegroup": info.get("spacegroup"),
                "description": info.get("description"),
                "applications": m.get("applications") or [],
                "n_wannier": wan.get("n_wannier_fncs", ham.get("n_wannier_tight_binding")),
                "baseline_n_bands": ham.get("baseline_n_bands"),
                "phasecraft_n_qubits": m["highlights"].get("phasecraft_n_qubits"),
                "phasecraft_circuit_depth": m["highlights"].get("phasecraft_circuit_depth"),
                "ks_electrons": (m.get("density_functional_theory") or {}).get("ks_electrons"),
            }
        )
    return records


def _structure_from_optimade(attrs: dict):
    from pymatgen.core import Lattice, Structure

    species = {s["name"]: s["chemical_symbols"][0] for s in attrs["species"]}
    return Structure(
        Lattice(attrs["lattice_vectors"]),
        [species[s] for s in attrs["species_at_sites"]],
        attrs["cartesian_site_positions"],
        coords_are_cartesian=True,
    )


def fetch_structure(material_id: str):
    """pymatgen Structure for a Materials Project id, or None if MP does not serve it."""
    api_key = os.environ.get("MP_API_KEY")
    if api_key:
        from mp_api.client import MPRester

        with MPRester(api_key) as mpr:
            try:
                return mpr.get_structure_by_material_id(material_id)
            except Exception:
                return None
    r = requests.get(f"{OPTIMADE}/{material_id}", timeout=60)
    data = r.json().get("data") if r.ok else None
    return _structure_from_optimade(data["attributes"]) if data else None


def resolve_by_formula(formula: str, natom: int | None, spacegroup: str | None) -> tuple[str, object, str]:
    """Pick the MP entry for a formula that Phasecraft lists without a usable MP id.

    Preference: same space group and atom count, then same space group, then lowest energy above hull.
    """
    from pymatgen.core import Composition
    from pymatgen.symmetry.analyzer import SpacegroupAnalyzer

    reduced = "".join(
        f"{el}{'' if amt == 1 else int(amt)}"
        for el, amt in sorted(Composition(formula).reduced_composition.as_dict().items())
    )
    params = {
        "filter": f'chemical_formula_reduced="{reduced}"',
        "response_fields": "lattice_vectors,species,species_at_sites,cartesian_site_positions,nsites,_mp_stability",
        "page_limit": 50,
    }
    entries = requests.get(OPTIMADE, params=params, timeout=60).json().get("data", [])
    if not entries:
        raise RuntimeError(f"no Materials Project entry for {formula}")

    def e_hull(attrs):
        stab = attrs.get("_mp_stability") or {}
        values = [v.get("energy_above_hull") for v in stab.values() if v.get("energy_above_hull") is not None]
        return min(values) if values else 1e9

    ranked = []
    for e in entries:
        s = _structure_from_optimade(e["attributes"])
        try:
            sg = SpacegroupAnalyzer(s, symprec=0.1).get_space_group_symbol()
        except Exception:
            sg = None
        ranked.append((sg != spacegroup, s.num_sites != natom, e_hull(e["attributes"]), e["id"], s, sg))
    ranked.sort(key=lambda t: t[:3])
    sg_miss, n_miss, _, mp_id, structure, sg = ranked[0]
    note = f"resolved by formula: spacegroup {'match' if not sg_miss else f'mismatch ({sg})'}, natom {'match' if not n_miss else f'mismatch ({structure.num_sites})'}"
    return mp_id, structure, note


def fetch_all(raw_dir: Path) -> list[dict]:
    raw_dir = Path(raw_dir)
    (raw_dir / "cifs").mkdir(parents=True, exist_ok=True)
    records = fetch_phasecraft()
    for rec in records:
        rec["phasecraft_id"] = rec["material_id"]
        structure = fetch_structure(rec["material_id"]) if rec["material_id"].startswith("mp-") else None
        rec["id_note"] = "phasecraft id"
        if structure is None:
            rec["material_id"], structure, rec["id_note"] = resolve_by_formula(
                rec["name"], rec["natom"], rec["spacegroup"]
            )
        cif_path = raw_dir / "cifs" / f"{rec['material_id']}.cif"
        if not cif_path.exists():
            cif_path.write_text(structure.to(fmt="cif"))
        rec["cif"] = str(cif_path.relative_to(raw_dir))
    (raw_dir / "phasecraft.json").write_text(json.dumps(records, indent=1))
    return records
