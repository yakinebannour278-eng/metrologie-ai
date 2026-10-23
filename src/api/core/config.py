"""
core/config.py
===============
Configuration for the FastAPI microservice. Reuses the project's single
source of truth (config/config.yaml) via src.utils.config, so nothing here
is hardcoded — model path, metadata path, and database path all come from
the [api] section of config.yaml.
"""
from functools import lru_cache
from pathlib import Path

from src.utils.config import get_project_root, load_config


class Settings:
    def __init__(self) -> None:
        self.project_root: Path = get_project_root()
        cfg = load_config()
        api_cfg = cfg.get("api", {})

        self.model_path: Path = self.project_root / api_cfg["model_file"]
        self.model_metadata_path: Path = self.project_root / api_cfg["model_metadata_file"]
        self.training_features_path: Path = self.project_root / api_cfg["training_features_file"]
        self.database_path: Path = self.project_root / api_cfg["database_path"]

        self.min_year: int = cfg["plausibility"]["min_year"]
        self.max_year: int = cfg["plausibility"]["max_year"]

        self.logs_dir: Path = self.project_root / cfg["paths"]["logs"]
        self.logs_dir.mkdir(parents=True, exist_ok=True)
        self.database_path.parent.mkdir(parents=True, exist_ok=True)


@lru_cache
def get_settings() -> Settings:
    return Settings()
