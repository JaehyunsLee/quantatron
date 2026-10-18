from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG = ROOT / "configs" / "phase1.yaml"


def _merge(base: dict, over: dict) -> dict:
    out = dict(base)
    for k, v in over.items():
        out[k] = _merge(base[k], v) if isinstance(v, dict) and isinstance(base.get(k), dict) else v
    return out


def _load(path: Path) -> dict:
    cfg = yaml.safe_load(path.read_text())
    parent = cfg.pop("defaults", None)
    return _merge(_load(path.parent / parent), cfg) if parent else cfg


def load_config(path: str | Path | None = None) -> dict:
    """Load a phase-1 config; a `defaults: other.yaml` key inherits from another config."""
    cfg = _load(Path(path) if path else DEFAULT_CONFIG)
    cfg["paths"] = {k: ROOT / v for k, v in cfg["paths"].items()}
    return cfg
