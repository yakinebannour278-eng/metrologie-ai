"""
utils/validation.py
====================
Validation that depends on runtime config (config.yaml plausibility
bounds), so it can't live as a static pydantic validator on the schema.
"""
from fastapi import HTTPException, status

from src.api.core.config import Settings


def validate_year(year: int, settings: Settings) -> None:
    if not (settings.min_year <= year <= settings.max_year):
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail=(
                f"year={year} is outside the plausible range "
                f"[{settings.min_year}, {settings.max_year}] defined in config/config.yaml."
            ),
        )
