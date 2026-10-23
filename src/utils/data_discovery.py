"""
src/utils/data_discovery.py
============================
Automatic, filename-agnostic dataset discovery for the metrology AI pipeline.

Extracted from the inline logic of 01_Data_Understanding.ipynb so every
downstream notebook (04, 05, ...) shares the exact same selection rule
instead of duplicating it.
"""
from pathlib import Path
import pandas as pd


def score_candidate(csv_path: Path, expected_columns: list, exclude_keywords: list,
                     split_keywords: list) -> dict:
    """Read only the header (+ row count) of a CSV and score it against the expected schema."""
    try:
        header_df = pd.read_csv(csv_path, nrows=0, encoding="utf-8-sig")
    except Exception as e:
        return {"path": csv_path, "readable": False, "error": str(e), "score": -1}

    columns = list(header_df.columns)
    overlap = set(c.strip() for c in columns) & set(expected_columns)
    name_lower = csv_path.stem.lower()

    is_excluded = any(kw in name_lower for kw in exclude_keywords)
    is_split = any(kw in name_lower for kw in split_keywords)

    try:
        n_rows = sum(1 for _ in open(csv_path, encoding="utf-8-sig")) - 1
    except Exception:
        n_rows = -1

    return {
        "path": csv_path,
        "readable": True,
        "n_columns_found": len(columns),
        "n_expected_columns_matched": len(overlap),
        "matched_columns": sorted(overlap),
        "is_excluded_by_keyword": is_excluded,
        "is_split_or_augmented": is_split,
        "n_rows": n_rows,
    }


def discover_dataset(search_dirs: list, expected_columns: list, exclude_keywords: list,
                      split_keywords: list, prefer_split: bool = False, logger=None):
    """
    Automatically select the most appropriate CSV dataset among all files found
    in `search_dirs`, without ever hardcoding a filename.

    Selection rule:
      1. Keep only readable candidates.
      2. Drop candidates excluded by keyword (log / tracabilite / audit / exclu files).
      3. If prefer_split=False (default): prefer the full (non split/augmented) dataset
         that matches the complete expected schema, with the most rows.
      4. If prefer_split=True: prefer the split/augmented dataset that matches the
         complete expected schema, with the most rows (useful for notebooks that
         explicitly want a train/augmented table).
      5. If the preferred category has no full-schema match, fall back to the best
         available candidate by (columns matched, row count).

    Returns
    -------
    (DATASET_PATH, candidates_df) : tuple[Path, pandas.DataFrame]
    """
    def log(level, msg, *args):
        if logger is not None:
            getattr(logger, level)(msg, *args)

    candidate_files = []
    for d in search_dirs:
        d = Path(d)
        if d.exists():
            candidate_files.extend(sorted(d.glob("*.csv")))

    log("info", "Found %d candidate CSV file(s) in %s", len(candidate_files), search_dirs)

    candidates = [score_candidate(p, expected_columns, exclude_keywords, split_keywords)
                  for p in candidate_files]
    candidates_df = pd.DataFrame(candidates)

    valid = [c for c in candidates if c.get("readable") and not c.get("is_excluded_by_keyword")]
    if not valid:
        raise RuntimeError(f"No valid metrology dataset candidate found in {search_dirs}.")

    def full_schema(c):
        return c["n_expected_columns_matched"] == len(expected_columns)

    preferred_split_flag = prefer_split
    primary_pool = [c for c in valid if full_schema(c) and c["is_split_or_augmented"] == preferred_split_flag]

    if primary_pool:
        chosen = max(primary_pool, key=lambda c: c["n_rows"])
        log("info", "Selected dataset (prefer_split=%s, full schema match): %s",
            prefer_split, chosen["path"].name)
    else:
        full_matches = [c for c in valid if full_schema(c)]
        if full_matches:
            chosen = max(full_matches, key=lambda c: c["n_rows"])
            log("warning", "No dataset matched prefer_split=%s with full schema — "
                            "falling back to best full-schema match: %s", prefer_split, chosen["path"].name)
        else:
            chosen = max(valid, key=lambda c: (c["n_expected_columns_matched"], c["n_rows"]))
            log("warning", "No full-schema dataset found at all — falling back to best partial match: %s",
                chosen["path"].name)

    DATASET_PATH = chosen["path"]
    log("info", "DATASET SELECTED: %s (%d rows, %d/%d expected columns matched)",
        DATASET_PATH, chosen["n_rows"], chosen["n_expected_columns_matched"], len(expected_columns))

    return DATASET_PATH, candidates_df
