"""Material families used for the held-out split.

Whole families go to train, validation or test, so the evaluation measures transfer to
chemistry the policy has never seen, not memorization of a sibling compound.
"""

FAMILIES: dict[str, list[str]] = {
    "perovskite_oxide": [
        "BaBiO3", "BaIrO3", "BaMnO3", "BaNiO3", "BaRuO3", "CaVO3", "LaFeO3", "LaNiO3", "LaTiO3",
        "PbHfO3", "PbTiO3", "SrMoO3", "SrTiO3", "SrVO3", "SrZrO3", "YFeO3", "YVO3",
    ],
    "halide_perovskite": ["CsPbF3", "CsPbI3"],
    "layered_oxide": ["CaCuO2", "Li2CuO2", "LiNiO2", "Sr2RuO4", "NiO2"],
    "tm_binary_oxide": ["CuO", "Fe2O3", "PdO", "NbO2", "TaO2", "TiO2", "HfO2", "Nb2O5"],
    "main_group_oxide": ["Al2O3", "MgO", "CaO"],
    "covalent_semiconductor": ["C", "Si", "SiC", "BN"],
    "compound_semiconductor": ["AlAs", "GaAs", "CdSe", "ZnSe"],
    "ionic_salt": ["NaCl", "BaS"],
    "hydride": ["H3S"],
}

FAMILY_OF: dict[str, str] = {name: fam for fam, names in FAMILIES.items() for name in names}


def family_of(name: str) -> str:
    return FAMILY_OF.get(name, "other")


def split_of(name: str, test_families: list[str], validation_families: list[str]) -> str:
    fam = family_of(name)
    if fam in test_families:
        return "test"
    if fam in validation_families:
        return "validation"
    return "train"
