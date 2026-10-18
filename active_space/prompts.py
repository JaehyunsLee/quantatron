"""Step 7-8: the prompt and tool schemas the policy sees. Scores never appear here."""

SYSTEM_PROMPT = """You are a computational chemist choosing an active space for a fault-tolerant quantum phase estimation (QPE) simulation of a crystalline material.

The material was computed with periodic restricted Hartree-Fock (Gamma point, fixed supercell, GTH pseudopotentials). You are given its frontier canonical orbitals: energy relative to the HOMO, Hartree-Fock occupation, MP2 natural occupation (the diagonal of the MP2 one-particle density matrix), and atomic character (meta-Lowdin weights on element shells such as "V-3d" or "O-2p").

Your choice is scored by
    score = accuracy - {beta} * log10(Toffoli count)
where accuracy is the fraction of the MP2 correlation energy recovered inside the active space and the Toffoli count is the cost of double-factorized qubitized QPE at fixed chemical accuracy (it grows quickly with the number of orbitals and the size of the Hamiltonian's one-norm). A larger space recovers more correlation but costs more; the best choice balances the two.

You may only choose from a fixed menu. A menu entry is a selection rule plus an even size n; it means a CAS(n electrons, n orbitals) built from n/2 occupied and n/2 virtual orbitals:
- "frontier": the n/2 highest occupied and n/2 lowest virtual orbitals.
- "<El>-<nl>" (for example "V-3d"): the n/2 occupied and n/2 virtual orbitals with the largest weight on that element shell.
The menu for each material lists which selections and sizes are available.

Tools:
- get_structure(material_id): crystal structure and composition.
- get_orbitals(material_id, candidate): the orbital table and localized-orbital summary; pass a candidate such as "V-3d:6" to see exactly which orbitals it contains, or null for the full table.
- submit_active_space(material_id, selection, n_orbitals): your final answer. Only your first valid submission counts. Invalid or off-menu submissions are rejected and you may correct them.

Think about which orbitals carry strong (static) correlation, for example partially filled transition-metal d shells, small HOMO-LUMO gaps, and MP2 occupations far from 2 or 0, and whether adding more orbitals is worth the cost. Then call submit_active_space exactly once with a valid menu entry."""


def tool_schemas() -> list[dict]:
    mid = {"type": "string", "description": "Materials Project id, for example mp-18717."}
    return [
        {
            "type": "function",
            "name": "get_structure",
            "description": "Read-only. Crystal structure, composition and the supercell used for the calculation.",
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
                "Read-only. Frontier orbital table (energies, occupations, MP2 occupations, atomic character), "
                "localized-orbital character counts and the menu. If candidate is given (e.g. 'frontier:8'), "
                "returns only the orbitals inside that candidate."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "material_id": mid,
                    "candidate": {
                        "type": ["string", "null"],
                        "description": "Menu entry '<selection>:<n>' to inspect, or null for the full table.",
                    },
                },
                "required": ["material_id", "candidate"],
                "additionalProperties": False,
            },
            "strict": True,
        },
        {
            "type": "function",
            "name": "submit_active_space",
            "description": "Submit the final active space. Must be an entry of this material's menu.",
            "parameters": {
                "type": "object",
                "properties": {
                    "material_id": mid,
                    "selection": {"type": "string", "description": "'frontier' or an element shell like 'V-3d'."},
                    "n_orbitals": {"type": "integer", "description": "Even active-space size from the menu."},
                },
                "required": ["material_id", "selection", "n_orbitals"],
                "additionalProperties": False,
            },
            "strict": True,
        },
    ]


def format_orbital_table(orbitals: list[dict]) -> str:
    lines = ["idx   E-E_HOMO(eV)  occ  MP2_occ  character"]
    for o in orbitals:
        char = ", ".join(f"{k} {v:.2f}" for k, v in o["character"].items())
        lines.append(f"{o['index']:>4}  {o['energy_eV']:>12.3f}  {o['occupation']:>3}  {o['mp2_occupation']:>7.4f}  {char}")
    return "\n".join(lines)


def format_menu(menu: dict[str, list[int]]) -> str:
    return "\n".join(f"- {sel}: sizes {sizes}" for sel, sizes in menu.items())


def user_prompt(entry: dict) -> str:
    s = entry["structure"]
    return f"""Material: {entry['name']} ({entry['material_id']}), space group {s['spacegroup']}, {s['natom']} atoms per primitive cell.
Lattice (Angstrom, degrees): a={s['abc'][0]:.3f} b={s['abc'][1]:.3f} c={s['abc'][2]:.3f} alpha={s['angles'][0]:.1f} beta={s['angles'][1]:.1f} gamma={s['angles'][2]:.1f}
Calculation: RHF/{entry['calc']['basis_label']}, supercell {entry['calc']['supercell']} ({entry['calc']['natom_supercell']} atoms, {entry['calc']['nelectron']} valence electrons), HOMO-LUMO gap {entry['calc']['gap_eV']:.2f} eV.
Localized orbitals in the frontier window, by dominant character (occupied/virtual): {entry['localized_summary']}

Frontier orbitals (index -1 is the HOMO, 0 is the LUMO):
{format_orbital_table(entry['orbitals'])}

Menu:
{format_menu(entry['menu'])}

Choose the active space for {entry['material_id']} and submit it with submit_active_space."""
