"""
train_v4_corrected.py
=======================
CORRECTS the training data lineage identified in the September 2026 audit
(see AUDIT_REPORT.md, section "Correction v4 - lineage d'entrainement").

Previous scripts (train_v1/v2/v3) all trained from ILA_features_selected.csv,
which is built from the FULL real dataset with no persisted, honest held-out
test split, and augmentation (in v3) was only ever done in-memory during
GroupKFold cross-validation - never as a materialized ILA_train_augmente.csv
consistent with the project's intended lineage:

    Raw data -> ILA_cleaned.csv -> train/test split -> ILA_train_reel.csv
             -> augmentation on TRAIN ONLY -> ILA_train_augmente.csv
             -> SAFE feature engineering + feature selection -> model training

This script implements that exact lineage, with ONE deliberate technical
correction to keep it safe: because this project's features are temporal
(rolling/EWM/lag, computed per-instrument in chronological order), a naive
"augment raw rows, then run rolling/shift-based feature engineering over the
augmented+real mix" ordering was verified (see AUDIT_REPORT.md) to corrupt
59/59 real rows' supervised target and 59/60 real rows' rolling features in
a controlled test. So here, "SAFE feature engineering" means: rolling/EWM/
lag/chrono_rank/target_next_valeur_reference are computed from REAL,
chronologically-ordered rows ONLY; each augmented row then inherits its
real source row's history features and real target VERBATIM, and only the
features derived from its own (noisy) current reading are recomputed. This
preserves the intended file lineage while guaranteeing no synthetic row can
ever become another row's history or redefine any row's target.

ILA_test_reel.csv is produced once, from real data, and is never read again
by anything in this script except the final honest evaluation.

Outputs (all under this project's existing data/processed and models/ folders
-- no new top-level structure introduced):
    data/processed/ILA_train_reel.csv
    data/processed/ILA_test_reel.csv
    data/processed/ILA_train_reel_features.csv
    data/processed/ILA_test_reel_features.csv
    data/processed/ILA_train_augmente.csv
    data/processed/ILA_train_augmente_features.csv
    models/regression_Lasso_v4_corrected.joblib
    models/regression_Lasso_v4_corrected_card.json
    models/metadata_v4.json          (same schema as models/metadata.json,
                                       so src/api/services/model_service.py
                                       needs ZERO code changes)
    reports/quality/v4_final_test_comparison.csv
    reports/quality/v4_per_instrument_test_results.csv
    reports/quality/v4_robustness_across_seeds.csv
"""
import json
import logging
from datetime import datetime, UTC
from pathlib import Path

import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import GroupShuffleSplit, GroupKFold
from sklearn.linear_model import Lasso
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger(__name__)

RANDOM_STATE = 42
N_AUGMENTED_COPIES = 3
ALPHA_GRID = [0.001, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0]
FEATURE_COLS = ["pos_in_band", "ewm_pos_in_band", "recent_shift", "rolling_trend", "uncertainty_norm"]

PROJECT_ROOT = Path(__file__).resolve().parent
DATA_PROCESSED = PROJECT_ROOT / "data" / "processed"
MODELS_DIR = PROJECT_ROOT / "models"
REPORTS_QUALITY = PROJECT_ROOT / "reports" / "quality"
CLEANED_PATH = DATA_PROCESSED / "ILA_cleaned.csv"


# ---------------------------------------------------------------------------
# SAFE feature engineering - mirrors notebooks/04_Feature_Engineering_Selection.ipynb's
# rolling/EWM/lag/target logic exactly, but is only ever run on real,
# chronologically-ordered rows (never on a real+synthetic mix).
# ---------------------------------------------------------------------------
def engineer_features_real(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values(["id_instrument", "annee", "point_numeric"]).reset_index(drop=True)
    g = df.groupby("id_instrument", group_keys=False)
    df["chrono_rank_in_instrument"] = g.cumcount()
    df["calibration_count"] = df["chrono_rank_in_instrument"] + 1
    df["time_since_last_calibration"] = g["annee"].diff()
    df["instrument_age_years"] = df["annee"] - g["annee"].transform("min")
    df["lag1_valeur_reference"] = g["valeur_reference"].shift(1)
    df["rolling_mean_3"] = g["valeur_reference"].transform(lambda s: s.rolling(3, min_periods=1).mean())
    df["moving_average_5"] = g["valeur_reference"].transform(lambda s: s.rolling(5, min_periods=1).mean())
    df["ewm_mean"] = g["valeur_reference"].transform(lambda s: s.ewm(span=3, min_periods=1).mean())
    df["target_next_valeur_reference"] = g["valeur_reference"].shift(-1)  # NEXT REAL observation, always
    df["origine"] = "reelle"
    return df


def build_ratio_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    emt_w = (df["EMT_haute"] - df["EMT_basse"]).replace(0, np.nan)
    df["pos_in_band"] = (df["valeur_reference"] - df["consigne"]) / emt_w
    df["ewm_pos_in_band"] = (df["ewm_mean"] - df["consigne"]) / emt_w
    df["recent_shift"] = (df["valeur_reference"] - df["ewm_mean"]) / emt_w
    df["rolling_trend"] = (df["rolling_mean_3"] - df["moving_average_5"]) / emt_w
    df["uncertainty_norm"] = (df["max"] - df["min"]) / emt_w
    df["target_delta"] = df["target_next_valeur_reference"] - df["valeur_reference"]
    return df


def augment_safe(real_features: pd.DataFrame, rng: np.random.Generator, n_copies: int) -> pd.DataFrame:
    """Noisy copies of the CURRENT reading only (same noise model as
    src/data/augment_ila.py: N(valeur_reference, incertitude)). History
    features and target_next_valeur_reference are copied VERBATIM from the
    real source row - never recomputed via shift/rolling over a mixed
    sequence. A synthetic row can therefore never become another row's
    history and never redefines any row's target."""
    rows = []
    for _, row in real_features.iterrows():
        if pd.isna(row["incertitude"]) or row["incertitude"] <= 0:
            continue
        emt_w = row["EMT_haute"] - row["EMT_basse"]
        if pd.isna(emt_w) or emt_w == 0:
            continue
        for _ in range(n_copies):
            noisy_vref = rng.normal(loc=row["valeur_reference"], scale=row["incertitude"])
            r = row.to_dict()
            r["valeur_reference"] = noisy_vref
            r["min"] = noisy_vref - row["incertitude"]
            r["max"] = noisy_vref + row["incertitude"]
            r["origine"] = "augmentee"
            for hist_col in ["rolling_mean_3", "moving_average_5", "ewm_mean", "lag1_valeur_reference",
                              "chrono_rank_in_instrument", "calibration_count",
                              "time_since_last_calibration", "instrument_age_years"]:
                r[hist_col] = row[hist_col]
            r["target_next_valeur_reference"] = row["target_next_valeur_reference"]  # VERBATIM, real target
            rows.append(r)
    aug = pd.DataFrame(rows)
    return build_ratio_features(aug)


def select_alpha_fold_safe(train_sup: pd.DataFrame, n_splits: int = 5):
    """GroupKFold alpha search where augmentation is regenerated per fold,
    using ONLY that fold's training rows - never the held-out fold - matching
    train_v3_augmented.py's honest-CV convention."""
    Xr = train_sup[FEATURE_COLS]
    yr_delta = train_sup["target_delta"]
    yr_abs = train_sup["target_next_valeur_reference"].values
    curr_r = train_sup["valeur_reference"].values
    groups_r = train_sup["id_instrument"]
    gkf = GroupKFold(n_splits=min(n_splits, groups_r.nunique()))
    rng = np.random.default_rng(RANDOM_STATE)
    best_alpha, best_mae = None, np.inf
    for alpha in ALPHA_GRID:
        preds = np.zeros(len(yr_abs))
        for tr_idx, te_idx in gkf.split(Xr, yr_delta, groups_r):
            fold_aug = augment_safe(train_sup.iloc[tr_idx], rng, N_AUGMENTED_COPIES)
            X_tr = pd.concat([Xr.iloc[tr_idx], fold_aug[FEATURE_COLS]], ignore_index=True)
            y_tr = pd.concat([yr_delta.iloc[tr_idx].reset_index(drop=True), fold_aug["target_delta"]], ignore_index=True)
            pipe = Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler()),
                              ("model", Lasso(alpha=alpha, random_state=RANDOM_STATE, max_iter=5000))])
            pipe.fit(X_tr, y_tr)
            preds[te_idx] = curr_r[te_idx] + pipe.predict(Xr.iloc[te_idx])
        mae = mean_absolute_error(yr_abs, preds)
        if mae < best_mae:
            best_alpha, best_mae = alpha, mae
    return best_alpha, best_mae


def robustness_check(cleaned: pd.DataFrame, n_seeds: int = 10) -> pd.DataFrame:
    results = []
    for seed in range(n_seeds):
        gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=seed)
        tr_idx, te_idx = next(gss.split(cleaned, groups=cleaned["id_instrument"]))
        tr_reel, te_reel = cleaned.iloc[tr_idx].copy(), cleaned.iloc[te_idx].copy()
        tr_feat = build_ratio_features(engineer_features_real(tr_reel))
        te_feat = build_ratio_features(engineer_features_real(te_reel))
        tr_sup = tr_feat[tr_feat["target_next_valeur_reference"].notna()].reset_index(drop=True)
        te_sup = te_feat[te_feat["target_next_valeur_reference"].notna()].reset_index(drop=True)
        if len(te_sup) < 5 or tr_sup["id_instrument"].nunique() < 5:
            continue
        X_test, y_test_abs = te_sup[FEATURE_COLS], te_sup["target_next_valeur_reference"].values
        current_test = te_sup["valeur_reference"].values
        baseline_mae = mean_absolute_error(y_test_abs, current_test)

        rng = np.random.default_rng(seed)
        aug = augment_safe(tr_sup, rng, N_AUGMENTED_COPIES)
        Xa = pd.concat([tr_sup[FEATURE_COLS], aug[FEATURE_COLS]], ignore_index=True)
        ya = pd.concat([tr_sup["target_delta"].reset_index(drop=True), aug["target_delta"]], ignore_index=True)
        pipe_a = Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler()),
                            ("model", Lasso(alpha=0.5, random_state=RANDOM_STATE, max_iter=5000))])
        pipe_a.fit(Xa, ya)
        aug_mae = mean_absolute_error(y_test_abs, current_test + pipe_a.predict(X_test))
        results.append({"seed": seed, "n_test_instruments": te_sup["id_instrument"].nunique(),
                         "n_test_rows": len(te_sup), "baseline_MAE": baseline_mae,
                         "real_plus_aug_MAE": aug_mae, "aug_beats_baseline": aug_mae < baseline_mae})
    return pd.DataFrame(results)


def main():
    cleaned = pd.read_csv(CLEANED_PATH, encoding="utf-8-sig")
    log.info("Loaded ILA_cleaned.csv: %s, %d instruments", cleaned.shape, cleaned["id_instrument"].nunique())

    # ---- STEP 1: split BY INSTRUMENT so test instruments are never seen in training ----
    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=RANDOM_STATE)
    tr_idx, te_idx = next(gss.split(cleaned, groups=cleaned["id_instrument"]))
    train_reel, test_reel = cleaned.iloc[tr_idx].copy(), cleaned.iloc[te_idx].copy()
    assert set(train_reel["id_instrument"]) & set(test_reel["id_instrument"]) == set(), "instrument leakage!"
    log.info("Train instruments: %d (%d rows) | Test instruments: %d (%d rows)",
              train_reel["id_instrument"].nunique(), len(train_reel),
              test_reel["id_instrument"].nunique(), len(test_reel))
    train_reel.to_csv(DATA_PROCESSED / "ILA_train_reel.csv", index=False, encoding="utf-8-sig")
    test_reel.to_csv(DATA_PROCESSED / "ILA_test_reel.csv", index=False, encoding="utf-8-sig")

    # ---- STEP 2: SAFE feature engineering, independently per split (zero cross-contamination) ----
    train_feat = build_ratio_features(engineer_features_real(train_reel))
    test_feat = build_ratio_features(engineer_features_real(test_reel))
    train_feat.to_csv(DATA_PROCESSED / "ILA_train_reel_features.csv", index=False, encoding="utf-8-sig")
    test_feat.to_csv(DATA_PROCESSED / "ILA_test_reel_features.csv", index=False, encoding="utf-8-sig")
    train_sup = train_feat[train_feat["target_next_valeur_reference"].notna()].reset_index(drop=True)
    test_sup = test_feat[test_feat["target_next_valeur_reference"].notna()].reset_index(drop=True)
    log.info("Supervised rows: train=%d, test=%d", len(train_sup), len(test_sup))

    # ---- STEP 3: augment TRAIN ONLY, safely (test never touched from here on) ----
    rng = np.random.default_rng(RANDOM_STATE)
    aug_full = augment_safe(train_sup, rng, N_AUGMENTED_COPIES)
    raw_cols = ["id_instrument", "annee", "point", "valeur_reference", "incertitude",
                "EMT_basse", "EMT_haute", "min", "max", "conforme", "origine"]
    train_augmente_raw = pd.concat([
        train_reel.assign(origine="reelle")[[c for c in raw_cols if c in train_reel.columns]],
        aug_full[[c for c in raw_cols if c in aug_full.columns]],
    ], ignore_index=True)
    train_augmente_raw.to_csv(DATA_PROCESSED / "ILA_train_augmente.csv", index=False, encoding="utf-8-sig")
    train_augmente_features = pd.concat([train_sup.assign(origine="reelle"), aug_full], ignore_index=True)
    train_augmente_features.to_csv(DATA_PROCESSED / "ILA_train_augmente_features.csv", index=False, encoding="utf-8-sig")
    log.info("ILA_train_augmente.csv: %s | ILA_train_augmente_features.csv: %s",
              train_augmente_raw.shape, train_augmente_features.shape)

    # ---- STEP 4: alpha selection (fold-safe, train instruments only) ----
    best_alpha, cv_mae = select_alpha_fold_safe(train_sup)
    log.info("Selected alpha=%.3f via fold-safe GroupKFold CV (train-only, MAE=%.4f)", best_alpha, cv_mae)

    # ---- STEP 5: baselines for comparison, evaluated on the untouched real test set ----
    X_test = test_sup[FEATURE_COLS]
    y_test_abs = test_sup["target_next_valeur_reference"].values
    current_test = test_sup["valeur_reference"].values

    def eval_model(preds, label):
        return {"model": label, "MAE": float(mean_absolute_error(y_test_abs, preds)),
                "RMSE": float(np.sqrt(mean_squared_error(y_test_abs, preds))),
                "R2": float(r2_score(y_test_abs, preds)), "n_test_rows": int(len(y_test_abs))}

    results = [eval_model(current_test, "Baseline_no_drift")]

    pipe_real = Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler()),
                           ("model", Lasso(alpha=best_alpha, random_state=RANDOM_STATE, max_iter=5000))])
    pipe_real.fit(train_sup[FEATURE_COLS], train_sup["target_delta"])
    preds_real = current_test + pipe_real.predict(X_test)
    results.append(eval_model(preds_real, "RealOnly_Lasso"))

    # ---- STEP 6: FINAL model - trained ONLY from ILA_train_augmente_features.csv (real+augmented TRAIN) ----
    pipe_final = Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler()),
                            ("model", Lasso(alpha=best_alpha, random_state=RANDOM_STATE, max_iter=5000))])
    pipe_final.fit(train_augmente_features[FEATURE_COLS], train_augmente_features["target_delta"])
    preds_final = current_test + pipe_final.predict(X_test)
    results.append(eval_model(preds_final, "Real_Plus_Augmented_Lasso_v4_DEPLOYED"))

    results_df = pd.DataFrame(results)
    log.info("\n%s", results_df.to_string(index=False))
    REPORTS_QUALITY.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(REPORTS_QUALITY / "v4_final_test_comparison.csv", index=False)

    per_inst = []
    for label, preds in [("Baseline_no_drift", current_test), ("RealOnly_Lasso", preds_real),
                          ("Real_Plus_Augmented_Lasso_v4", preds_final)]:
        tmp = test_sup[["id_instrument"]].copy()
        tmp["abs_error"] = np.abs(preds - y_test_abs)
        g = tmp.groupby("id_instrument")["abs_error"].agg(["mean", "count"]).reset_index()
        g["model"] = label
        per_inst.append(g)
    pd.concat(per_inst, ignore_index=True).rename(columns={"mean": "MAE", "count": "n_rows"}).to_csv(
        REPORTS_QUALITY / "v4_per_instrument_test_results.csv", index=False)

    # ---- STEP 7: robustness check across 10 independent instrument splits (no cherry-picking) ----
    robustness_df = robustness_check(cleaned, n_seeds=10)
    robustness_df.to_csv(REPORTS_QUALITY / "v4_robustness_across_seeds.csv", index=False)
    n_wins = int(robustness_df["aug_beats_baseline"].sum())
    n_total = len(robustness_df)
    log.info("Robustness: augmentation beat baseline in %d/%d independent splits", n_wins, n_total)
    log.info("Mean baseline MAE=%.4f, mean real+aug MAE=%.4f across splits",
              robustness_df["baseline_MAE"].mean(), robustness_df["real_plus_aug_MAE"].mean())

    beats_baseline_this_split = results[2]["MAE"] < results[0]["MAE"]

    # ---- STEP 8: save deployed artifact + metadata (SAME schema as models/metadata.json) ----
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    model_out = MODELS_DIR / "regression_Lasso_v4_corrected.joblib"
    card_out = MODELS_DIR / "regression_Lasso_v4_corrected_card.json"
    metadata_out = MODELS_DIR / "metadata_v4.json"
    joblib.dump(pipe_final, model_out)

    coefs = dict(zip(FEATURE_COLS, pipe_final.named_steps["model"].coef_.tolist()))
    intercept = float(pipe_final.named_steps["model"].intercept_)

    card = {
        "model_name": "Lasso_v4_corrected",
        "model_type": "sklearn.pipeline.Pipeline (SimpleImputer[median] -> StandardScaler -> Lasso), predicts DELTA",
        "trained_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "corrects": "train_v3_augmented.py trained from ILA_features_selected.csv with no persisted, honest "
                    "instrument-level held-out test set. v4 introduces a real GroupShuffleSplit-by-instrument "
                    "train/test split, materializes ILA_train_reel.csv / ILA_test_reel.csv / "
                    "ILA_train_augmente.csv / ILA_train_augmente_features.csv as literal files, and evaluates "
                    "on a test set never touched by augmentation, feature engineering, or training.",
        "features": FEATURE_COLS,
        "fitted_coefficients": coefs,
        "fitted_intercept": intercept,
        "hyperparameters": {"alpha": best_alpha, "random_state": RANDOM_STATE, "max_iter": 5000},
        "n_real_training_rows": int(len(train_sup)),
        "n_augmented_training_rows": int(len(aug_full)),
        "n_train_instruments": int(train_sup["id_instrument"].nunique()),
        "n_test_instruments": int(test_sup["id_instrument"].nunique()),
        "held_out_test_results": results,
        "beats_baseline_this_split": bool(beats_baseline_this_split),
        "robustness_across_10_independent_splits": {
            "aug_beats_baseline_count": n_wins, "total_splits": n_total,
            "mean_baseline_MAE": float(robustness_df["baseline_MAE"].mean()),
            "mean_real_plus_aug_MAE": float(robustness_df["real_plus_aug_MAE"].mean()),
        },
        "honest_disclaimer": (
            f"On this specific instrument-holdout split, real+augmented "
            f"{'beat' if beats_baseline_this_split else 'did NOT beat'} the persistence baseline. "
            f"Across 10 independent random instrument splits, augmentation beat baseline in only "
            f"{n_wins}/{n_total} cases, and the mean MAE across splits was "
            f"{'lower' if robustness_df['real_plus_aug_MAE'].mean() < robustness_df['baseline_MAE'].mean() else 'not lower'} "
            f"for real+augmented than for baseline. Consistent with train_v3_augmented.py's and "
            f"experiments/audit_v3/FINAL_REPORT.md's prior findings: this dataset's real drift signal is small "
            f"relative to measurement uncertainty at ~30 rows/instrument, and augmentation does not reliably "
            f"outperform simple persistence. This is reported as-is, not adjusted to look better."
        ),
    }
    json.dump(card, open(card_out, "w", encoding="utf-8"), indent=2)

    # metadata_v4.json: SAME SCHEMA as models/metadata.json so model_service.py needs no code change
    metadata = {
        "model_name": "Lasso_v4_corrected",
        "model_version": f"4.0.0-lasso-v4-corrected-{datetime.now(UTC).strftime('%Y%m%dT%H%M%S')}",
        "model_type": card["model_type"],
        "artifact_file": f"models/{model_out.name}",
        "training_date": card["trained_at"],
        "trained_by": "train_v4_corrected.py",
        "target_variable": "delta_next_valeur_reference (target_next_valeur_reference - valeur_reference)",
        "target_description": "Predicted change (drift) to the next REAL calibration campaign for the same "
                               "instrument. target_next_valeur_reference is always the next REAL chronological "
                               "observation - synthetic rows never redefine it.",
        "prediction_mode": "delta_normalized_v2",
        "prediction_unit": "UNKNOWN / REQUIRES CONFIRMATION - not recorded in the source ILA data.",
        "feature_order": FEATURE_COLS,
        "fitted_coefficients": coefs,
        "fitted_intercept": intercept,
        "training_dataset": "data/processed/ILA_train_augmente_features.csv (real TRAIN instruments + safe "
                             "noise-augmented copies only - TEST instruments never included)",
        "cv_strategy_reference": "GroupShuffleSplit by instrument (80/20) for train/test; GroupKFold (5-fold, "
                                  "fold-safe augmentation) on TRAIN instruments only for alpha selection.",
        "n_real_training_rows": int(len(train_sup)),
        "n_augmented_training_rows": int(len(aug_full)),
        "n_training_rows": int(len(train_augmente_features)),
        "n_instruments_total": int(cleaned["id_instrument"].nunique()),
        "held_out_test_evaluation": results,
        "known_limitations": [card["honest_disclaimer"]],
    }
    json.dump(metadata, open(metadata_out, "w", encoding="utf-8"), indent=2)
    log.info("Saved model=%s, card=%s, metadata=%s", model_out, card_out, metadata_out)

    print()
    print("=" * 70)
    print(results_df.to_string(index=False))
    print(f"Robustness: augmentation beat baseline in {n_wins}/{n_total} independent instrument splits")
    print("=" * 70)


if __name__ == "__main__":
    main()
