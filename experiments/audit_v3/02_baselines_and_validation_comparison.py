"""
audit_v3/02_baselines_and_validation_comparison.py
=====================================================
Builds all 4 requested baselines and evaluates them under TWO validation
strategies to show explicitly how they differ, as requested: chronological
holdout (matches actual deployment: train on past, predict future) vs
GroupKFold by instrument (tests generalization to unseen instruments, but
does not respect global time order).
"""
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold
from sklearn.metrics import mean_absolute_error, mean_squared_error, median_absolute_error, r2_score

df = pd.read_csv("experiments/audit_v3/features_v3.csv", encoding="utf-8-sig")
y_abs = df["target_A_next_value"].values
current = df["valeur_reference"].values

def metrics(y_true, y_pred, label):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    medae = median_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    bias = np.mean(y_pred - y_true)
    print(f"{label:38s} MAE={mae:7.4f} RMSE={rmse:7.4f} MedAE={medae:7.4f} R2={r2:7.4f} bias={bias:+7.4f}")
    return {"label": label, "MAE": mae, "RMSE": rmse, "MedAE": medae, "R2": r2, "bias": bias}

results = []

# ============================================================
# BASELINE 1: No drift (predict current value, i.e. delta=0)
# ============================================================
pred_b1 = current
results.append(metrics(y_abs, pred_b1, "Baseline1_no_drift"))

# ============================================================
# BASELINE 2: Last observed drift (delta_1 carried forward)
#   predicted_next = current + delta_1 (previous drift)
#   If delta_1 unavailable (first observation), fall back to no-drift.
# ============================================================
delta_1 = df["delta_1"].values
pred_b2 = np.where(np.isnan(delta_1), current, current + delta_1)
results.append(metrics(y_abs, pred_b2, "Baseline2_last_observed_drift"))

# ============================================================
# BASELINE 3: Historical instrument mean drift (expanding mean of past
#   deltas, using ONLY information available before this row)
# ============================================================
work = df.sort_values(["id_instrument", "annee", "chrono_rank_in_instrument"]).copy()
work["realized_drift"] = work.groupby("id_instrument")["valeur_reference"].diff()
work["hist_mean_drift"] = work.groupby("id_instrument")["realized_drift"].transform(
    lambda s: s.shift(1).expanding(min_periods=1).mean()
)
work = work.set_index(df.index[work.index]) if False else work  # keep alignment simple
work_aligned = work.reindex(df.index)  # noop safeguard, same index since no reindexing happened
hist_mean_drift = work["hist_mean_drift"].reindex(df.index).values
pred_b3 = np.where(np.isnan(hist_mean_drift), current, current + hist_mean_drift)
results.append(metrics(y_abs, pred_b3, "Baseline3_historical_mean_drift"))

# ============================================================
# BASELINE 4: Linear historical trend (only where >=3 prior points exist;
#   uses historical_slope computed strictly causally in step 01)
# ============================================================
slope = df["historical_slope"].values
years_since = df["years_since_previous"].values
pred_trend_delta = slope * 1.0  # predict 1 year ahead
pred_b4 = np.where(np.isnan(pred_trend_delta), current, current + pred_trend_delta)
results.append(metrics(y_abs, pred_b4, "Baseline4_linear_trend"))

print()
print("=== Baseline comparison table ===")
baseline_df = pd.DataFrame(results)
baseline_df.to_csv("experiments/audit_v3/baseline_comparison.csv", index=False)
print(baseline_df.to_string(index=False))
print()

# ============================================================
# Show explicitly how chronological vs GroupKFold validation differ,
# using Baseline1 (deterministic, no fitting needed) as a neutral probe,
# then repeat with a simple fitted model (Ridge) to show it matters for
# fitted models too.
# ============================================================
print("=" * 70)
print("VALIDATION STRATEGY COMPARISON (same data, same baseline/model)")
print("=" * 70)

# --- Chronological split: train = years <= cutoff, test = years > cutoff ---
years_sorted = np.sort(df["annee"].unique())
cutoff_idx = int(len(years_sorted) * 0.7)
cutoff_year = years_sorted[cutoff_idx]
train_mask = df["annee"] <= cutoff_year
test_mask = df["annee"] > cutoff_year
print(f"Chronological split: train years <= {cutoff_year} (n={train_mask.sum()}), "
      f"test years > {cutoff_year} (n={test_mask.sum()})")

metrics(y_abs[test_mask.values], pred_b1[test_mask.values], "  [Chrono test] Baseline1_no_drift")

from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

FEATURES = ["lag_1", "delta_1", "rolling_mean_3", "expanding_mean", "ewm_mean",
            "distance_to_EMT_center", "normalized_distance_to_tolerance",
            "uncertainty_ratio", "n_previous_observations", "historical_nonconformity_rate"]
X = df[FEATURES]

pipe = Pipeline([("imp", SimpleImputer(strategy="median")), ("sc", StandardScaler()), ("m", Ridge(alpha=1.0, random_state=42))])
pipe.fit(X[train_mask], y_abs[train_mask])
pred_chrono_ridge = pipe.predict(X[test_mask])
metrics(y_abs[test_mask.values], pred_chrono_ridge, "  [Chrono test] Ridge (10 causal feats)")

# --- GroupKFold by instrument, for comparison ---
groups = df["id_instrument"]
gkf = GroupKFold(n_splits=5)
preds_gkf = np.zeros(len(y_abs))
for tr, te in gkf.split(X, y_abs, groups):
    p = Pipeline([("imp", SimpleImputer(strategy="median")), ("sc", StandardScaler()), ("m", Ridge(alpha=1.0, random_state=42))])
    p.fit(X.iloc[tr], y_abs[tr])
    preds_gkf[te] = p.predict(X.iloc[te])
metrics(y_abs, preds_gkf, "  [GroupKFold, all rows] Ridge (10 causal feats)")

print()
print("Interpretation: GroupKFold evaluates 'does this generalize to an unseen")
print("INSTRUMENT' (mixing time freely). Chronological evaluates 'does this")
print("generalize to the FUTURE' (the actual deployment scenario). They can")
print("legitimately disagree - both are reported, chronological is treated as")
print("primary per the requested methodology.")
