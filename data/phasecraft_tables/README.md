# Precomputed active-space tables

Flat per-candidate tables for the materials in `data/raw/phasecraft.json`, one row per (material, candidate active
space). These are inputs, not outputs of this repo's PySCF pipeline (`scripts/01`–`03`, which writes `data/work*/`).

| File | Rows | Used by | Notes |
| --- | --- | --- | --- |
| `dzvp.csv` | 450 | `scripts/candidates/build_env_data.py` → `resources_servers/active_space_candidates/data/` | GTH-DZVP-MOLOPT-SR basis. Accuracy `corr_recovered` against an FCI reference in a 12-orbital window; cost `toffoli_total` from Qualtran's double-factorized block encoding × QPE steps; `score = corr_recovered − 0.2·log10(toffoli_total)`. `use_for_training` flags the 330 usable rows |
| `dzvp.json` | 450 | — | Same rows as `dzvp.csv`, plus the calculation `settings` (basis, pseudopotential, density fitting, reference, cost model) |
| `szv.csv` | 171 (16 materials) | — | Earlier GTH-SZV version of the table; kept for reference (its family split is described in `build_env_data.py`) |
| `szv_single_turn_prompts.jsonl` | 14 | — | Single-turn (`ANSWER: <id>`) prompts from an early, non-agentic version of the task; kept for reference |

`source` in `dzvp.csv` marks rows computed for this table (`this_pc`) versus rows imported from an older run
(`imported`); see the docstring of `scripts/candidates/build_env_data.py` for which rows are dropped and why.
