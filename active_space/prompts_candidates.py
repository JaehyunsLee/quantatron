"""Prompts and tool schemas for the active_space_candidates environment: active_space/prompts.py adapted to the
candidate-level granularity of data/phasecraft_tables/dzvp.csv (no shared per-orbital table across candidates, only a per-candidate
summary: size, energy window, dominant atomic character, and the estimated QPE cost as log10 of the Toffoli count).
Accuracy, scores and rewards never appear here: the cost is a cheap resource estimate, the accuracy needs the
expensive correlated calculation the agent is meant to avoid.
"""

SYSTEM_PROMPT = """You are a computational chemist choosing an active space for a fault-tolerant quantum phase estimation (QPE) simulation of a crystalline material.

The material was computed with periodic restricted Hartree-Fock (fixed basis, pseudopotential and supercell). You are given a fixed menu of candidate active spaces for this material. Each candidate is a CAS(n electrons, n orbitals) drawn from the material's canonical orbitals, described by: the number of orbitals, the energy window (eV, relative to the Fermi level) it spans, its dominant atomic character (meta-Lowdin weights on element shells such as "Ni 3d" or "O 2p"), and its estimated QPE cost as log10 of the Toffoli count.

Your choice is scored by
    score = accuracy - {beta} * log10(Toffoli count)
where accuracy (0 to 1) is the fraction of the correlation energy recovered inside the active space and the Toffoli count is the cost of double-factorized qubitized QPE at fixed chemical accuracy. The cost of every candidate is given (log10_toffolis); its accuracy is not, and you have to judge it from the orbitals.

Put the two terms on the same scale before deciding: each extra factor of 10 in Toffoli count lowers the score by only {beta}. So a candidate with one more decade of cost is the better choice whenever it recovers at least {beta} more of the correlation energy (for example 0.55 instead of 0.30). Across one menu the recovered correlation typically differs by much more than the cost penalty does, so do not reject a larger space just because it is more expensive; reject it only if the extra orbitals are unlikely to carry correlation (for example far from the Fermi level, or of a character already well covered).

You may only choose from the fixed menu given below. Each entry is a candidate_id like "c3:7", where the number after the colon is the number of active orbitals; the candidate's electron count (shown separately for each entry) is not always equal to its orbital count.

Tools:
- get_structure(material_id): formula, atom count, HOMO-LUMO gap and a classical reference active space for context.
- get_orbitals(material_id, candidate_id): the menu of candidates; pass a candidate_id such as "c3:7" to see only that candidate's detail, or null for the full menu.
- submit_active_space(material_id, candidate_id): your final answer. Only your first valid submission counts. Invalid or off-menu submissions are rejected and you may correct them.

Think about which shells likely carry strong static correlation (partially filled transition-metal d shells, small gaps), and whether a bigger active space is worth the added cost. Then call submit_active_space exactly once with a valid candidate_id from the menu."""


def tool_schemas() -> list[dict]:
    mid = {"type": "string", "description": "Materials Project id, for example mp-18717."}
    return [
        {
            "type": "function",
            "name": "get_structure",
            "description": "Read-only. Formula, atom count, HOMO-LUMO gap and the classical reference active space.",
            "parameters": {
                "type": "object",
                "properties": {"material_id": mid},
                "required": ["material_id"],
                "additionalProperties": False,
            },
            "strict": True,
        },
        {
            "type": "function",
            "name": "get_orbitals",
            "description": (
                "Read-only. The candidate menu (size, energy window, dominant character, log10 Toffoli cost). If candidate_id "
                "is given (e.g. 'c3:7'), returns only that candidate's detail."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "material_id": mid,
                    "candidate_id": {"type": ["string", "null"], "description": "Menu entry to inspect, or null for the full menu."},
                },
                "required": ["material_id", "candidate_id"],
                "additionalProperties": False,
            },
            "strict": True,
        },
        {
            "type": "function",
            "name": "submit_active_space",
            "description": "Submit the final active space. Must be a candidate_id from this material's menu.",
            "parameters": {
                "type": "object",
                "properties": {
                    "material_id": mid,
                    "candidate_id": {"type": "string", "description": "Menu entry, for example 'c3:7'."},
                },
                "required": ["material_id", "candidate_id"],
                "additionalProperties": False,
            },
            "strict": True,
        },
    ]


def format_menu(menu: dict) -> str:
    lines = ["candidate_id   n_orbitals   n_electrons   log10_toffolis   window_eV            character"]
    for cid, c in sorted(menu.items(), key=lambda kv: kv[1]["n_orbitals"]):
        char = ", ".join(f"{k} {v:.2f}" for k, v in c["character"].items())
        lines.append(
            f"{cid:<14}{c['n_orbitals']:<13}{c['n_electrons']:<14}{c['log10_toffolis']:<17.2f}"
            f"[{c['window_ev'][0]:.2f}, {c['window_ev'][1]:.2f}]   {char}"
        )
    return "\n".join(lines)


def user_prompt(entry: dict) -> str:
    return f"""Material: {entry['name']} ({entry['material_id']}), formula {entry['formula']}, {entry['n_atoms']} atoms per primitive cell.
Calculation: RHF, HOMO-LUMO gap {entry['gap_eV']:.2f} eV. For context, a classical reference active space for this material is CAS{entry['ref_cas']}.

Candidate menu:
{format_menu(entry['menu'])}

Choose the active space for {entry['material_id']} and submit it with submit_active_space."""
