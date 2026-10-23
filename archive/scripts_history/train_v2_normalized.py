"""
train_v2_normalized.py
=======================
Retrains the drift-prediction model using the fix identified in the audit:
the original 12 features were found to be ~collinear (r > 0.94 for every pair)
because they all just encode "which instrument/scale regime is this", drowning
out the real within-instrument drift signal. This version:

1. Replaces them with 5 features normalized by the instrument's own EMT band
   width (a known, non-leaking, instrument-level scale constant) — removing
   the between-instrument scale confound.
2. Predicts the DELTA (next value - current value), not the absolute next
   value — the final prediction is reconstructed as current + predicted_delta.

Same methodology as notebooks/05_Regression_Model.ipynb for direct
comparability: GroupKFold(5, groups=id_instrument), RANDOM_STATE=42, same
metrics, same baseline definition.

HONEST RESULT (do not misrepresent this): this v2 model has MAE ~2.8 vs the
old v1 model's MAE ~5.0 (a real, substantial improvement) but the trivial
"no drift" persistence baseline still has a lower MAE (~1.96). v2 is deployed
because it is a genuine, verified improvement over v1 — NOT because it has
been shown to beat the baseline. See AUDIT_REPORT.md section 7.
"""
import json
import logging
from datetime import datetime, UTC
from pathlib import Path

import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import GroupKFold
from sklearn.linear_model import Lasso
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, median_absolute_error, r2_score

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger(__name__)

RANDOM_STATE = 42
PROJECT_ROOT = Path(__file__).resolve().parent
FEATURES_PATH = PROJECT_ROOT / "data" / "processed" / "ILA_features_selected.csv"
MODEL_OUT = PROJECT_ROOT / "models" / "regression_Lasso_v2_normalized.joblib"
CARD_OUT = PROJECT_ROOT / "models" / "regression_Lasso_v2_normalized_card.json"

FEATURE_COLS = ["pos_in_band", "ewm_pos_in_band", "recent_shift", "rolling_trend", "uncertainty_norm"]
ALPHA_GRID = [0.001, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0]


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    emt_width = (df["EMT_haute"] - df["EMT_basse"]).replace(0, np.nan)
    df["pos_in_band"] = (df["valeur_reference"] - df["consigne"]) / emt_width
    df["ewm_pos_in_band"] = (df["ewm_mean"] - df["consigne"]) / emt_width
    df["recent_shift"] = (df["valeur_reference"] - df["ewm_mean"]) / emt_width
    df["rolling_trend"] = (df["rolling_mean_3"] - df["moving_average_5"]) / emt_width
    df["uncertainty_norm"] = (df["max"] - df["min"]) / emt_width
    return df


def main():
    log.info("Loading %s", FEATURES_PATH)
    df = pd.read_csv(FEATURES_PATH, encoding="utf-8-sig")
    supervised = df[df["target_next_valeur_reference"].notna()].copy()
    supervised = build_features(supervised)

    X = supervised[FEATURE_COLS]
    y_delta = supervised["target_next_valeur_reference"] - supervised["valeur_reference"]
    current = supervised["valeur_reference"].values
    y_abs = supervised["target_next_valeur_reference"].values
    groups = supervised["id_instrument"]
    gkf = GroupKFold(n_splits=5)

    log.info("Honest hyperparameter search (GroupKFold, alpha grid=%s)", ALPHA_GRID)
    best_alpha, best_mae = None, np.inf
    for alpha in ALPHA_GRID:
        preds = np.zeros(len(y_abs))
        for tr, te in gkf.split(X, y_delta, groups):
            pipe = Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                ("model", Lasso(alpha=alpha, random_state=RANDOM_STATE, max_iter=5000)),
            ])
            pipe.fit(X.iloc[tr], y_delta.iloc[tr])
            preds[te] = current[te] + pipe.predict(X.iloc[te])
        mae = mean_absolute_error(y_abs, preds)
        log.info("  alpha=%.3f  MAE=%.4f", alpha, mae)
        if mae < best_mae:
            best_alpha, best_mae = alpha, mae
    log.info("Best alpha=%.3f (MAE=%.4f)", best_alpha, best_mae)

    # Final honest out-of-fold evaluation at the chosen alpha (this IS the reported metric)
    preds_oof = np.zeros(len(y_abs))
    for tr, te in gkf.split(X, y_delta, groups):
        pipe = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("model", Lasso(alpha=best_alpha, random_state=RANDOM_STATE, max_iter=5000)),
        ])
        pipe.fit(X.iloc[tr], y_delta.iloc[tr])
        preds_oof[te] = current[te] + pipe.predict(X.iloc[te])

    metrics = {
        "MAE": float(mean_absolute_error(y_abs, preds_oof)),
        "RMSE": float(np.sqrt(mean_squared_error(y_abs, preds_oof))),
        "MedAE": float(median_absolute_error(y_abs, preds_oof)),
        "R2": float(r2_score(y_abs, preds_oof)),
    }
    baseline_mae = float(mean_absolute_error(y_abs, current))
    metrics["baseline_persistence_MAE_for_comparison"] = baseline_mae
    metrics["beats_baseline"] = metrics["MAE"] < baseline_mae

    log.info("Final GroupKFold metrics: %s", metrics)

    # Refit on ALL supervised data for the deployed artifact (standard: CV estimates
    # generalization, final artifact uses all available data)
    final_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("model", Lasso(alpha=best_alpha, random_state=RANDOM_STATE, max_iter=5000)),
    ])
    final_pipe.fit(X, y_delta)

    joblib.dump(final_pipe, MODEL_OUT)
    log.info("Saved model to %s", MODEL_OUT)

    card = {
        "model_name": "Lasso_v2_normalized",
        "model_type": "sklearn.pipeline.Pipeline (SimpleImputer[median] -> StandardScaler -> Lasso, predicts DELTA)",
        "trained_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "features": FEATURE_COLS,
        "target": "delta = target_next_valeur_reference - valeur_reference (final prediction = valeur_reference + predicted_delta)",
        "hyperparameters": {"alpha": best_alpha, "random_state": RANDOM_STATE, "max_iter": 5000},
        "n_training_rows": int(len(supervised)),
        "n_instruments": int(groups.nunique()),
        "metrics_groupkfold": metrics,
        "predecessor_model": "regression_Lasso.joblib (v1) - MAE ~5.0 on the same rows, using 12 collinear absolute-scale features",
        "improvement_over_v1": "Real, verified: MAE reduced from ~5.0 (v1) to %.3f (v2) - a %.0f%% error reduction" % (
            metrics["MAE"], 100 * (1 - metrics["MAE"] / 5.0)
        ),
        "honest_disclaimer": "This model does NOT beat the trivial persistence baseline (MAE %.3f). It is deployed because it is a genuine, substantial improvement over v1, not because it has been shown to add predictive value over doing nothing. See AUDIT_REPORT.md." % baseline_mae,
    }
    json.dump(card, open(CARD_OUT, "w", encoding="utf-8"), indent=2)
    log.info("Saved model card to %s", CARD_OUT)

    print()
    print("=" * 70)
    print(f"v2 MAE:       {metrics['MAE']:.4f}")
    print(f"v1 MAE (ref): ~5.0")
    print(f"baseline MAE: {baseline_mae:.4f}")
    print(f"v2 beats baseline: {metrics['beats_baseline']}")
    print("=" * 70)


if __name__ == "__main__":
    main()
