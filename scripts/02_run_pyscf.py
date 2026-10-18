"""Steps 2-4: PySCF mean field + localization + menu + accuracy + Qualtran cost, per material.

Resumable: a material whose data/work/<mp-id>.json exists is skipped (use --force to redo).

    python scripts/02_run_pyscf.py                     # all 46, sequential
    python scripts/02_run_pyscf.py --jobs 8            # parallel over materials
    python scripts/02_run_pyscf.py --only SrVO3 Si     # a subset (by name or mp-id)
    python scripts/02_run_pyscf.py --config configs/phase1_fast.yaml --jobs 4   # GTH-SZV first pass
"""

import argparse
import json
import os
import sys
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from active_space.config import load_config


def _run(rec, cfg):
    from active_space.pipeline import run_material

    try:
        r = run_material(rec, cfg, cfg["paths"]["raw_dir"], cfg["paths"]["work_dir"])
        return rec["name"], f"ok  {len(r['candidates'])} candidates, {r['meta']['wall_time_s']}s"
    except Exception as e:
        err_dir = cfg["paths"]["work_dir"] / "errors"
        err_dir.mkdir(parents=True, exist_ok=True)
        (err_dir / f"{rec['material_id']}.txt").write_text(traceback.format_exc())
        return rec["name"], f"FAILED {type(e).__name__}: {e}"


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--jobs", type=int, default=1)
    p.add_argument("--only", nargs="*")
    p.add_argument("--force", action="store_true")
    p.add_argument("--config", default=None, help="e.g. configs/phase1_fast.yaml")
    args = p.parse_args()

    cfg = load_config(args.config)
    records = json.loads((cfg["paths"]["raw_dir"] / "phasecraft.json").read_text())
    if args.only:
        records = [r for r in records if r["name"] in args.only or r["material_id"] in args.only]
    todo = [r for r in records if args.force or not (cfg["paths"]["work_dir"] / f"{r['material_id']}.json").exists()]
    print(f"{len(todo)} materials to run ({len(records) - len(todo)} cached)")

    if args.jobs > 1:
        # Split the cores between workers so BLAS threads do not oversubscribe.
        os.environ.setdefault("OMP_NUM_THREADS", str(max(1, (os.cpu_count() or 1) // args.jobs)))
        with ProcessPoolExecutor(args.jobs) as pool:
            futures = [pool.submit(_run, r, cfg) for r in todo]
            for f in as_completed(futures):
                print(*f.result(), flush=True)
    else:
        for r in todo:
            print(*_run(r, cfg), flush=True)
