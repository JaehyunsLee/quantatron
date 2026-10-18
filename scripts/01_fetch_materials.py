"""Step 1: download the Phasecraft material list (46 MP ids + Wannier counts) and the CIFs."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from active_space.config import load_config
from active_space.families import family_of
from active_space.sources import fetch_all

if __name__ == "__main__":
    cfg = load_config()
    records = fetch_all(cfg["paths"]["raw_dir"])
    for r in records:
        print(f"{r['material_id']:>12}  {r['name']:<9} natom={r['natom']:<4} wannier={r['n_wannier']:<4} {family_of(r['name'])}")
    print(f"{len(records)} materials -> {cfg['paths']['raw_dir']}")
