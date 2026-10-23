"""
config.py
=========
Single entry point for reading config/config.yaml from anywhere in the
project (notebooks, scripts, src modules), regardless of the current
working directory the interpreter was launched from.

Usage
-----
from src.utils.config import load_config, get_project_root

cfg = load_config()
raw_dir = get_project_root() / cfg["paths"]["data_raw"]
"""
from pathlib import Path
import yaml


def get_project_root() -> Path:
    """
    Walk up from this file until a directory containing 'config/config.yaml'
    is found. This makes path resolution independent of whether the
    notebook/script is launched from the project root, from notebooks/,
    or from src/.
    """
    here = Path(__file__).resolve()
    for parent in [here.parent, *here.parents]:
        if (parent / "config" / "config.yaml").exists():
            return parent
    raise FileNotFoundError(
        "Could not locate project root (no config/config.yaml found in any parent directory)."
    )


def load_config() -> dict:
    root = get_project_root()
    config_path = root / "config" / "config.yaml"
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    return cfg


def resolve_path(relative_path: str) -> Path:
    """Resolve a path from config.yaml (relative to project root) into an absolute Path."""
    return get_project_root() / relative_path
