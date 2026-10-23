"""
Experiment: does removing the between-instrument scale confound (collinearity finding)
and predicting drift (delta) instead of the absolute next value close the gap with the
persistence baseline?

Methodology matches notebooks/05_Regression_Model.ipynb exactly: GroupKFold(5, groups=id_instrument),
same RANDOM_STATE, same metrics, same baseline definition — so results are directly comparable.
"""
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold
from sklearn.linear_model import Lasso, Ridge, ElasticNet
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.dummy import DummyRegressor

RANDOM_STATE = 42

df = pd.read_csv("data/processed/ILA_features_selected.csv", encoding="utf-8-sig")
supervised = df[df["target_next_valeur_reference"].notna()].copy()

# --- New, scale-normalized features (fix for the collinearity finding) ---
# EMT band width = a known, non-leaking, instrument-level scale constant
emt_width = (supervised["EMT_haute"] - supervised["EMT_basse"]).replace(0, np.nan)

supervised["pos_in_band"] = (supervised["valeur_reference"] - supervised["consigne"]) / emt_width
supervised["ewm_pos_in_band"] = (supervised["ewm_mean"] - supervised["consigne"]) / emt_width
supervised["recent_shift"] = (supervised["valeur_reference"] - supervised["ewm_mean"]) / emt_width
supervised["rolling_trend"] = (supervised["rolling_mean_3"] - supervised["moving_average_5"]) / emt_width
supervised["uncertainty_norm"] = (supervised["max"] - supervised["min"]) / emt_width  # normalized incertitude proxy

NEW_FEATURES = ["pos_in_band", "ewm_pos_in_band", "recent_shift", "rolling_trend", "uncertainty_norm"]

# --- New target: predict the DRIFT (delta), not the absolute next value ---
supervised["target_delta"] = supervised["target_next_valeur_reference"] - supervised["valeur_reference"]

X = supervised[NEW_FEATURES]
y_delta = supervised["target_delta"]
y_abs_true = supervised["target_next_valeur_reference"]
current = supervised["valeur_reference"]
groups = supervised["id_instrument"]

gkf = GroupKFold(n_splits=5)

def evaluate(model, X, y_delta, current, y_abs_true, groups, label):
    preds_abs = np.zeros(len(y_abs_true))
    for train_idx, test_idx in gkf.split(X, y_delta, groups):
        pipe = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("model", model),
        ])
        pipe.fit(X.iloc[train_idx], y_delta.iloc[train_idx])
        pred_delta = pipe.predict(X.iloc[test_idx])
        preds_abs[test_idx] = current.iloc[test_idx].values + pred_delta
    mae = mean_absolute_error(y_abs_true, preds_abs)
    r2 = r2_score(y_abs_true, preds_abs)
    print(f"{label:30s}  MAE={mae:6.3f}  R2={r2:6.3f}")
    return mae

print("=== NEW APPROACH: scale-normalized features + delta target ===")
print(f"(n={len(supervised)} rows, {groups.nunique()} instruments, features={NEW_FEATURES})")
print()
evaluate(Lasso(alpha=0.01, random_state=RANDOM_STATE), X, y_delta, current, y_abs_true, groups, "Lasso (new features+delta)")
evaluate(Ridge(alpha=1.0, random_state=RANDOM_STATE), X, y_delta, current, y_abs_true, groups, "Ridge (new features+delta)")
evaluate(ElasticNet(alpha=0.01, random_state=RANDOM_STATE), X, y_delta, current, y_abs_true, groups, "ElasticNet (new features+delta)")

print()
print("=== BASELINE (for direct comparison, same rows) ===")
baseline_pred = current.values  # persistence: predict no change
baseline_mae = mean_absolute_error(y_abs_true, baseline_pred)
baseline_r2 = r2_score(y_abs_true, baseline_pred)
print(f"{'Persistence baseline':30s}  MAE={baseline_mae:6.3f}  R2={baseline_r2:6.3f}")

print()
print("=== OLD APPROACH for reference (original 12 collinear features, absolute target) ===")
OLD_FEATURES = ["EMT_basse","EMT_haute","max","rolling_min_3","consigne","valeur_reference",
                 "ewm_mean","moving_average_5","min","rolling_median_3","rolling_mean_3","rolling_max_3"]
X_old = supervised[OLD_FEATURES]
preds_abs_old = np.zeros(len(y_abs_true))
for train_idx, test_idx in gkf.split(X_old, y_abs_true, groups):
    pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("model", Lasso(alpha=0.01, random_state=RANDOM_STATE)),
    ])
    pipe.fit(X_old.iloc[train_idx], y_abs_true.iloc[train_idx])
    preds_abs_old[test_idx] = pipe.predict(X_old.iloc[test_idx])
mae_old = mean_absolute_error(y_abs_true, preds_abs_old)
r2_old = r2_score(y_abs_true, preds_abs_old)
print(f"{'Lasso (OLD 12 features, absolute)':30s}  MAE={mae_old:6.3f}  R2={r2_old:6.3f}")
