"""
audit_v3/03_full_benchmark.py
===============================
Benchmarks LinearRegression, Ridge, Lasso, ElasticNet, HuberRegressor,
RandomForestRegressor, ExtraTreesRegressor, GradientBoostingRegressor,
HistGradientBoostingRegressor under TWO validation strategies:
  - Chronological holdout (PRIMARY - matches deployment: train on past years,
    predict the most recent year)
  - GroupKFold by instrument (secondary - tests unseen-instrument generalization)

Both target strategies (A: absolute next value, B: drift, reconstructed) are
tested. Every result is compared directly against Baseline1 (no drift) on
the SAME test rows.
"""
import warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet, HuberRegressor
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor, GradientBoostingRegressor, HistGradientBoostingRegressor
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

RANDOM_STATE = 42
df = pd.read_csv("experiments/audit_v3/features_v3.csv", encoding="utf-8-sig")

FEATURES = ["lag_1", "delta_1", "rolling_mean_3", "expanding_mean", "ewm_mean",
            "distance_to_EMT_center", "normalized_distance_to_tolerance",
            "uncertainty_ratio", "n_previous_observations", "historical_nonconformity_rate"]
X = df[FEATURES]
y_abs = df["target_A_next_value"].values
y_drift = df["target_B_drift"].values
current = df["valeur_reference"].values
groups = df["id_instrument"]

years_sorted = np.sort(df["annee"].unique())
cutoff_year = years_sorted[int(len(years_sorted) * 0.7)]
train_mask = (df["annee"] <= cutoff_year).values
test_mask = (df["annee"] > cutoff_year).values

MODELS = {
    "LinearRegression": LinearRegression(),
    "Ridge": Ridge(alpha=1.0, random_state=RANDOM_STATE),
    "Lasso": Lasso(alpha=0.01, random_state=RANDOM_STATE, max_iter=5000),
    "ElasticNet": ElasticNet(alpha=0.01, random_state=RANDOM_STATE, max_iter=5000),
    "Huber": HuberRegressor(max_iter=1000),
    "RandomForest": RandomForestRegressor(n_estimators=200, random_state=RANDOM_STATE, n_jobs=-1),
    "ExtraTrees": ExtraTreesRegressor(n_estimators=200, random_state=RANDOM_STATE, n_jobs=-1),
    "GradientBoosting": GradientBoostingRegressor(random_state=RANDOM_STATE),
    "HistGradientBoosting": HistGradientBoostingRegressor(random_state=RANDOM_STATE),
}

def eval_metrics(y_true, y_pred):
    return {
        "MAE": mean_absolute_error(y_true, y_pred),
        "RMSE": np.sqrt(mean_squared_error(y_true, y_pred)),
        "R2": r2_score(y_true, y_pred),
    }

rows = []

baseline_chrono = eval_metrics(y_abs[test_mask], current[test_mask])
baseline_gkf = eval_metrics(y_abs, current)  # baseline needs no fitting, evaluate on all rows
rows.append({"model": "Baseline1_no_drift", "target": "-", "validation": "chronological", **baseline_chrono})
rows.append({"model": "Baseline1_no_drift", "target": "-", "validation": "GroupKFold", **baseline_gkf})

for name, model in MODELS.items():
    for target_label, y in [("A_absolute", y_abs), ("B_drift", y_drift)]:
        # --- Chronological (primary) ---
        pipe = Pipeline([("imp", SimpleImputer(strategy="median")), ("sc", StandardScaler()), ("m", model)])
        try:
            pipe.fit(X[train_mask], y[train_mask])
            raw_pred = pipe.predict(X[test_mask])
            final_pred = raw_pred if target_label == "A_absolute" else current[test_mask] + raw_pred
            m = eval_metrics(y_abs[test_mask], final_pred)
            rows.append({"model": name, "target": target_label, "validation": "chronological", **m})
        except Exception as e:
            rows.append({"model": name, "target": target_label, "validation": "chronological", "MAE": None, "RMSE": None, "R2": None, "error": str(e)[:100]})

        # --- GroupKFold (secondary) ---
        gkf = GroupKFold(n_splits=5)
        preds = np.zeros(len(y_abs))
        try:
            for tr, te in gkf.split(X, y, groups):
                p = Pipeline([("imp", SimpleImputer(strategy="median")), ("sc", StandardScaler()), ("m", model)])
                p.fit(X.iloc[tr], y[tr])
                raw = p.predict(X.iloc[te])
                preds[te] = raw if target_label == "A_absolute" else current[te] + raw
            m = eval_metrics(y_abs, preds)
            rows.append({"model": name, "target": target_label, "validation": "GroupKFold", **m})
        except Exception as e:
            rows.append({"model": name, "target": target_label, "validation": "GroupKFold", "MAE": None, "RMSE": None, "R2": None, "error": str(e)[:100]})

results = pd.DataFrame(rows)
results.to_csv("experiments/audit_v3/model_comparison.csv", index=False)

pd.set_option("display.width", 160)
pd.set_option("display.max_rows", 100)
print("=== FULL BENCHMARK: chronological validation (PRIMARY) ===")
chrono = results[results["validation"] == "chronological"].sort_values("MAE")
print(chrono.to_string(index=False))
print()
print(f"Baseline1 (chronological) MAE = {baseline_chrono['MAE']:.4f}")
beats = chrono[(chrono["MAE"].notna()) & (chrono["MAE"] < baseline_chrono["MAE"])]
print(f"Models beating baseline under chronological validation: {len(beats)} / {len(chrono)-2}")  # -2 for the 2 baseline rows
if len(beats):
    print(beats[["model","target","MAE"]].to_string(index=False))

print()
print("=== FULL BENCHMARK: GroupKFold validation (secondary) ===")
gkf_res = results[results["validation"] == "GroupKFold"].sort_values("MAE")
print(gkf_res.to_string(index=False))
print()
print(f"Baseline1 (GroupKFold) MAE = {baseline_gkf['MAE']:.4f}")
beats_gkf = gkf_res[(gkf_res["MAE"].notna()) & (gkf_res["MAE"] < baseline_gkf["MAE"])]
print(f"Models beating baseline under GroupKFold: {len(beats_gkf)} / {len(gkf_res)-2}")
if len(beats_gkf):
    print(beats_gkf[["model","target","MAE"]].to_string(index=False))
