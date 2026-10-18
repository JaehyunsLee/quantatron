"""Steps 5-7: scores + oracle, family split, environment files and NeMo Gym datasets.

Reads data/work*/ (output of 02_run_pyscf.py) and writes into resources_servers/active_space/data/:
    orbitals.json         agent-visible lookups (served by get_structure / get_orbitals)
    scores.json           verify()-only scores, rewards, oracle, classical baselines, Phasecraft reference
    {train,validation,test,example}.jsonl   NeMo Gym rows (prompt + tools + material_id)

    python scripts/03_build_table.py [--config configs/phase1_fast.yaml] [--beta 0.05]
"""

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from active_space.config import load_config
from active_space.prompts import SYSTEM_PROMPT, tool_schemas, user_prompt
from active_space.table import build


def gym_row(entry: dict, score: dict, beta: float) -> dict:
    return {
        "responses_create_params": {
            "input": [
                {"role": "system", "content": SYSTEM_PROMPT.format(beta=f"{beta:.4g}")},
                {"role": "user", "content": user_prompt(entry)},
            ],
            "tools": tool_schemas(),
        },
        # Routing for older Gym/NeMo RL readers; `gym dataset collate` replaces it with task_source.
        "agent_ref": {"type": "responses_api_agents", "name": "active_space_simple_agent"},
        "task_id": entry["material_id"],
        "material_id": entry["material_id"],
        "name": entry["name"],
        "family": score["family"],
    }


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--config", default=None)
    p.add_argument("--beta", default=None, help="override score.beta (a number or 'auto')")
    args = p.parse_args()

    cfg = load_config(args.config)
    beta = None if args.beta is None else (args.beta if args.beta == "auto" else float(args.beta))
    out = build(cfg, beta)
    scores, orbitals = out["scores"], out["orbitals"]
    settings = scores["settings"]

    env_dir = cfg["paths"]["env_data_dir"]
    env_dir.mkdir(parents=True, exist_ok=True)
    (env_dir / "orbitals.json").write_text(json.dumps(orbitals))
    (env_dir / "scores.json").write_text(json.dumps(scores, indent=1))

    rows = {"train": [], "validation": [], "test": []}
    for mid, s in scores["materials"].items():
        rows[s["split"]].append(gym_row(orbitals[mid], s, settings["beta"]))
    rows["example"] = rows["train"][:5]
    for split, rs in rows.items():
        with open(env_dir / f"{split}.jsonl", "w") as f:
            f.writelines(json.dumps(r) + "\n" for r in rs)

    # ---- diagnostics
    if out["beta_report"]:
        print("beta auto-selection (beta, oracle-size entropy, oracle sizes):")
        for b, h, sizes in out["beta_report"][::6]:
            print(f"  {b:8.4f}  H={h:.2f}  {sizes}")
    print(f"\nmetric={settings['metric']}  beta={settings['beta']:.4g} ({settings['beta_mode']})")
    print(f"{'material':<10}{'split':<11}{'family':<24}{'oracle':<14}{'avas':<14}{'r_avas':>7}  {'no_thr':<14}{'r_no':>6}")
    for mid, s in sorted(scores["materials"].items(), key=lambda kv: (kv[1]["split"], kv[1]["family"])):
        b = s["baselines"]
        r = lambda cid: s["candidates"].get(cid, {}).get("reward", 0.0)
        print(f"{s['name']:<10}{s['split']:<11}{s['family']:<24}{s['oracle']:<14}{b['avas']:<14}{r(b["avas"]):7.2f}  "
              f"{b['no_threshold']:<14}{r(b['no_threshold']):6.2f}")
    print("\nrows per split:", {k: len(v) for k, v in rows.items()})
    print("oracle sizes:", dict(Counter(int(s["oracle"].split(":")[1]) for s in scores["materials"].values())))
    print("oracle selections:", dict(Counter(s["oracle"].split(":")[0] for s in scores["materials"].values())))
    print(f"wrote {env_dir}")
