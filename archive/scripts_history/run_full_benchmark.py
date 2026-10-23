"""
run_full_benchmark.py
======================
Executes the CORRECTED data lineage (ILA_cleaned.csv -> instrument split ->
safe feature engineering -> augmentation on TRAIN ONLY -> training) and
benchmarks every suitable sklearn regressor against the persistence baseline
on the real, untouched test set. This is the actual execution engine behind
notebooks/05_Regression_Model.ipynb section 16 - run standalone here (no
Jupyter/nbconvert available in this sandbox) and its real console + figure
outputs are then embedded into the notebook cells so the notebook reflects
genuine executed results, not just code.

xgboost / lightgbm / catboost are NOT installed in this environment and
there is no network access to install them - they are excluded and this is
reported explicitly rather than faked.
"""
import json
import time
import warnings
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import GroupShuffleSplit, GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet, HuberRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import (RandomForestRegressor, ExtraTreesRegressor,
                               GradientBoostingRegressor, HistGradientBoostingRegressor,
                               AdaBoostRegressor, BaggingRegressor)
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, median_absolute_error

warnings.filterwarnings("ignore")
sns.set_style("whitegrid")
plt.rcParams["figure.facecolor"] = "white"

RANDOM_STATE = 42
N_AUGMENTED_COPIES = 3
FEATURE_COLS = ["pos_in_band", "ewm_pos_in_band", "recent_shift", "rolling_trend", "uncertainty_norm"]

PROJECT_ROOT = Path("/home/claude/metrology-ai-pipeline/metrology-ai-pipeline")
DATA_PROCESSED = PROJECT_ROOT / "data" / "processed"
MODELS_DIR = PROJECT_ROOT / "models"
REPORTS_QUALITY = PROJECT_ROOT / "reports" / "quality"
FIG_DIR = PROJECT_ROOT / "reports" / "figures" / "05_regression_model"
for d in (MODELS_DIR, REPORTS_QUALITY, FIG_DIR):
    d.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Safe pipeline (identical methodology validated earlier in this project)
# ---------------------------------------------------------------------------
def engineer_features_real(df):
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
    df["target_next_valeur_reference"] = g["valeur_reference"].shift(-1)
    df["origine"] = "reelle"
    return df


def build_ratio_features(df):
    df = df.copy()
    emt_w = (df["EMT_haute"] - df["EMT_basse"]).replace(0, np.nan)
    df["pos_in_band"] = (df["valeur_reference"] - df["consigne"]) / emt_w
    df["ewm_pos_in_band"] = (df["ewm_mean"] - df["consigne"]) / emt_w
    df["recent_shift"] = (df["valeur_reference"] - df["ewm_mean"]) / emt_w
    df["rolling_trend"] = (df["rolling_mean_3"] - df["moving_average_5"]) / emt_w
    df["uncertainty_norm"] = (df["max"] - df["min"]) / emt_w
    df["target_delta"] = df["target_next_valeur_reference"] - df["valeur_reference"]
    return df


def augment_safe(real_feat, rng, n_copies):
    rows = []
    for _, row in real_feat.iterrows():
        if pd.isna(row["incertitude"]) or row["incertitude"] <= 0:
            continue
        emt_w = row["EMT_haute"] - row["EMT_basse"]
        if pd.isna(emt_w) or emt_w == 0:
            continue
        for _ in range(n_copies):
            noisy = rng.normal(loc=row["valeur_reference"], scale=row["incertitude"])
            r = row.to_dict()
            r["valeur_reference"] = noisy
            r["min"] = noisy - row["incertitude"]
            r["max"] = noisy + row["incertitude"]
            r["origine"] = "augmentee"
            for c in ["rolling_mean_3", "moving_average_5", "ewm_mean", "lag1_valeur_reference",
                      "chrono_rank_in_instrument", "calibration_count",
                      "time_since_last_calibration", "instrument_age_years"]:
                r[c] = row[c]
            r["target_next_valeur_reference"] = row["target_next_valeur_reference"]
            rows.append(r)
    aug = pd.DataFrame(rows)
    return build_ratio_features(aug)


log_lines = []
def log(*a):
    s = " ".join(str(x) for x in a)
    print(s)
    log_lines.append(s)


# ---- STEP 1-3: corrected lineage (same as prior session, re-run from scratch) ----
cleaned = pd.read_csv(DATA_PROCESSED / "ILA_cleaned.csv", encoding="utf-8-sig")
gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=RANDOM_STATE)
tr_idx, te_idx = next(gss.split(cleaned, groups=cleaned["id_instrument"]))
train_reel, test_reel = cleaned.iloc[tr_idx].copy(), cleaned.iloc[te_idx].copy()
assert set(train_reel["id_instrument"]) & set(test_reel["id_instrument"]) == set(), "instrument leakage!"
log(f"Train: {train_reel['id_instrument'].nunique()} instruments / {len(train_reel)} rows | "
    f"Test: {test_reel['id_instrument'].nunique()} instruments / {len(test_reel)} rows (held out)")

train_feat = build_ratio_features(engineer_features_real(train_reel))
test_feat = build_ratio_features(engineer_features_real(test_reel))
train_reel.to_csv(DATA_PROCESSED / "ILA_train_reel.csv", index=False, encoding="utf-8-sig")
test_reel.to_csv(DATA_PROCESSED / "ILA_test_reel.csv", index=False, encoding="utf-8-sig")
train_feat.to_csv(DATA_PROCESSED / "ILA_train_reel_features.csv", index=False, encoding="utf-8-sig")
test_feat.to_csv(DATA_PROCESSED / "ILA_test_reel_features.csv", index=False, encoding="utf-8-sig")

train_sup = train_feat[train_feat["target_next_valeur_reference"].notna()].reset_index(drop=True)
test_sup = test_feat[test_feat["target_next_valeur_reference"].notna()].reset_index(drop=True)
log(f"Supervised: train={len(train_sup)}, test={len(test_sup)}")

rng = np.random.default_rng(RANDOM_STATE)
aug = augment_safe(train_sup, rng, N_AUGMENTED_COPIES)
raw_cols = ["id_instrument", "annee", "point", "valeur_reference", "incertitude",
            "EMT_basse", "EMT_haute", "min", "max", "conforme", "origine"]
train_augmente_raw = pd.concat([
    train_reel.assign(origine="reelle")[[c for c in raw_cols if c in train_reel.columns]],
    aug[[c for c in raw_cols if c in aug.columns]]], ignore_index=True)
train_augmente_raw.to_csv(DATA_PROCESSED / "ILA_train_augmente.csv", index=False, encoding="utf-8-sig")
train_aug_feat = pd.concat([train_sup.assign(origine="reelle"), aug], ignore_index=True)
train_aug_feat.to_csv(DATA_PROCESSED / "ILA_train_augmente_features.csv", index=False, encoding="utf-8-sig")
log(f"ILA_train_augmente_features.csv (actual training table): {train_aug_feat.shape}")

X_train = train_aug_feat[FEATURE_COLS]
y_train_delta = train_aug_feat["target_delta"]
groups_train_full = train_aug_feat["id_instrument"]  # for GroupKFold CV on the augmented table
X_test = test_sup[FEATURE_COLS]
y_test_abs = test_sup["target_next_valeur_reference"].values
current_test = test_sup["valeur_reference"].values
current_train = train_aug_feat["valeur_reference"].values

# ---- chronological diagnostic split (within TRAIN instruments only, no shuffling): ----
# per instrument, first 80% chronological rows = chrono-train, last 20% = chrono-val
chrono_val_idx, chrono_train_idx = [], []
for inst, g in train_sup.groupby("id_instrument"):
    g_sorted = g.sort_values(["annee", "point_numeric"])
    cut = max(1, int(len(g_sorted) * 0.8))
    chrono_train_idx += list(g_sorted.index[:cut])
    chrono_val_idx += list(g_sorted.index[cut:])
chrono_train_idx = [i for i in chrono_train_idx if i in train_sup.index]
chrono_val_idx = [i for i in chrono_val_idx if i in train_sup.index]
log(f"Chronological diagnostic split (within TRAIN instruments, no shuffling): "
    f"{len(chrono_train_idx)} chrono-train rows / {len(chrono_val_idx)} chrono-val rows")


# ---------------------------------------------------------------------------
# Model zoo
# ---------------------------------------------------------------------------
def make_pipeline(estimator):
    return Pipeline([("imputer", SimpleImputer(strategy="median")),
                      ("scaler", StandardScaler()),
                      ("model", estimator)])


base_models = {
    "DummyRegressor_mean": DummyRegressor(strategy="mean"),
    "LinearRegression": LinearRegression(),
    "Ridge": Ridge(alpha=1.0, random_state=RANDOM_STATE),
    "Lasso": Lasso(alpha=0.5, random_state=RANDOM_STATE, max_iter=5000),
    "ElasticNet": ElasticNet(alpha=0.5, l1_ratio=0.5, random_state=RANDOM_STATE, max_iter=5000),
    "HuberRegressor": HuberRegressor(max_iter=1000),
    "KNeighborsRegressor": KNeighborsRegressor(n_neighbors=5),
    "SVR_rbf": SVR(kernel="rbf", C=1.0),
    "DecisionTreeRegressor": DecisionTreeRegressor(max_depth=5, random_state=RANDOM_STATE),
    "RandomForestRegressor": RandomForestRegressor(n_estimators=200, random_state=RANDOM_STATE),
    "ExtraTreesRegressor": ExtraTreesRegressor(n_estimators=200, random_state=RANDOM_STATE),
    "GradientBoostingRegressor": GradientBoostingRegressor(random_state=RANDOM_STATE),
    "HistGradientBoostingRegressor": HistGradientBoostingRegressor(random_state=RANDOM_STATE),
    "AdaBoostRegressor": AdaBoostRegressor(random_state=RANDOM_STATE),
    "BaggingRegressor": BaggingRegressor(random_state=RANDOM_STATE),
}

# tuned variants: small grids appropriate for this dataset's size (655 real + 1896 aug rows)
tuned_grids = {
    "RandomForestRegressor_tuned": (
        RandomForestRegressor(random_state=RANDOM_STATE),
        {"n_estimators": [100, 300], "max_depth": [3, 6, None], "min_samples_leaf": [1, 3, 5]}),
    "ExtraTreesRegressor_tuned": (
        ExtraTreesRegressor(random_state=RANDOM_STATE),
        {"n_estimators": [100, 300], "max_depth": [3, 6, None], "min_samples_leaf": [1, 3, 5]}),
    "GradientBoostingRegressor_tuned": (
        GradientBoostingRegressor(random_state=RANDOM_STATE),
        {"n_estimators": [50, 150], "max_depth": [2, 3], "learning_rate": [0.03, 0.1]}),
}

results = []
fitted_pipelines = {}
n_cv_splits = min(5, groups_train_full.nunique())
gkf = GroupKFold(n_splits=n_cv_splits)

baseline_test_mae = mean_absolute_error(y_test_abs, current_test)
log(f"\nBaseline (no-drift persistence) held-out test MAE = {baseline_test_mae:.4f}")


def cv_mae_groupkfold(estimator):
    """GroupKFold CV MAE on the augmented TRAIN table (fold-safe: augmented rows follow
    their real source row's fold since they share id_instrument groups)."""
    preds = np.zeros(len(y_train_delta))
    for tr_i, te_i in gkf.split(X_train, y_train_delta, groups_train_full):
        pipe = make_pipeline(estimator)
        pipe.fit(X_train.iloc[tr_i], y_train_delta.iloc[tr_i])
        preds[te_i] = current_train[te_i] + pipe.predict(X_train.iloc[te_i])
    return mean_absolute_error(train_aug_feat["target_next_valeur_reference"].values, preds)


def chrono_val_mae(estimator):
    pipe = make_pipeline(estimator)
    Xc_tr = train_sup.loc[chrono_train_idx, FEATURE_COLS]
    yc_tr = train_sup.loc[chrono_train_idx, "target_delta"]
    Xc_val = train_sup.loc[chrono_val_idx, FEATURE_COLS]
    yc_val_abs = train_sup.loc[chrono_val_idx, "target_next_valeur_reference"].values
    curr_val = train_sup.loc[chrono_val_idx, "valeur_reference"].values
    pipe.fit(Xc_tr, yc_tr)
    preds = curr_val + pipe.predict(Xc_val)
    return mean_absolute_error(yc_val_abs, preds)


def get_feature_importance(pipe):
    model = pipe.named_steps["model"]
    if hasattr(model, "coef_"):
        return dict(zip(FEATURE_COLS, np.atleast_1d(model.coef_).tolist()))
    if hasattr(model, "feature_importances_"):
        return dict(zip(FEATURE_COLS, model.feature_importances_.tolist()))
    return None


all_models = {}
for name, est in base_models.items():
    all_models[name] = est

for name, (est, grid) in tuned_grids.items():
    from sklearn.model_selection import GridSearchCV
    t0 = time.perf_counter()
    search_pipe = make_pipeline(est)
    param_grid = {f"model__{k}": v for k, v in grid.items()}
    gs = GridSearchCV(search_pipe, param_grid, cv=GroupKFold(n_splits=min(3, groups_train_full.nunique())),
                       scoring="neg_mean_absolute_error", n_jobs=-1)
    gs.fit(X_train, y_train_delta, groups=groups_train_full)
    tuning_time = time.perf_counter() - t0
    best_est = gs.best_estimator_.named_steps["model"]
    all_models[name] = best_est
    log(f"{name}: hyperparameter search done in {tuning_time:.1f}s, best_params={gs.best_params_}")

log(f"\nTotal algorithms to benchmark: {len(all_models)}")
log(f"NOTE: xgboost / lightgbm / catboost are NOT installed in this sandbox and there is no network "
    f"access to install them - they are excluded from this benchmark (not faked).")

for name, est in all_models.items():
    t0 = time.perf_counter()
    pipe = make_pipeline(est)
    pipe.fit(X_train, y_train_delta)
    train_time = time.perf_counter() - t0
    preds_test = current_test + pipe.predict(X_test)
    fitted_pipelines[name] = pipe

    mae = mean_absolute_error(y_test_abs, preds_test)
    rmse = float(np.sqrt(mean_squared_error(y_test_abs, preds_test)))
    r2 = r2_score(y_test_abs, preds_test)
    medae = median_absolute_error(y_test_abs, preds_test)
    bias = float(np.mean(preds_test - y_test_abs))
    pred_std = float(np.std(preds_test))
    uses_features = pred_std > 1e-6  # constant predictor would have ~0 std across varying inputs

    try:
        cv_mae = cv_mae_groupkfold(est)
    except Exception as e:
        cv_mae = np.nan
    try:
        chrono_mae = chrono_val_mae(est)
    except Exception as e:
        chrono_mae = np.nan

    results.append({
        "model": name, "MAE": mae, "RMSE": rmse, "R2": r2, "MedAE": medae, "Bias": bias,
        "pred_std": pred_std, "uses_features": uses_features,
        "beats_baseline": mae < baseline_test_mae,
        "CV_MAE_GroupKFold": cv_mae, "Chrono_val_MAE": chrono_mae,
        "train_time_sec": train_time, "hyperparameters": str(est.get_params()),
    })
    log(f"{name:35s} MAE={mae:8.4f}  RMSE={rmse:8.4f}  R2={r2:8.5f}  Bias={bias:8.4f}  "
        f"beats_baseline={mae < baseline_test_mae}  uses_features={uses_features}  time={train_time:.3f}s")

results_df = pd.DataFrame(results).sort_values("MAE").reset_index(drop=True)
results_df.insert(0, "rank", range(1, len(results_df) + 1))
baseline_row = pd.DataFrame([{
    "rank": 0, "model": "Baseline_no_drift_persistence", "MAE": baseline_test_mae,
    "RMSE": float(np.sqrt(mean_squared_error(y_test_abs, current_test))),
    "R2": r2_score(y_test_abs, current_test),
    "MedAE": median_absolute_error(y_test_abs, current_test),
    "Bias": float(np.mean(current_test - y_test_abs)), "pred_std": float(np.std(current_test)),
    "uses_features": False, "beats_baseline": False, "CV_MAE_GroupKFold": np.nan, "Chrono_val_MAE": np.nan,
    "train_time_sec": 0.0, "hyperparameters": "n/a (persistence: predicts current reading unchanged)",
}])
full_results_df = pd.concat([baseline_row, results_df], ignore_index=True).sort_values("MAE").reset_index(drop=True)
full_results_df.to_csv(REPORTS_QUALITY / "full_algorithm_benchmark.csv", index=False)

log("\n" + "=" * 130)
log("FULL BENCHMARK RESULTS (sorted by held-out test MAE, real untouched test set)")
log("=" * 130)
log(full_results_df[["rank", "model", "MAE", "RMSE", "R2", "MedAE", "Bias",
                      "beats_baseline", "CV_MAE_GroupKFold", "train_time_sec"]].to_string(index=False))

n_beating = int(full_results_df["beats_baseline"].sum())
log(f"\nModels beating baseline on held-out test: {n_beating} / {len(all_models)}")

# ---- robustness across 10 independent splits, for the single best learned model ----
best_model_name = results_df.iloc[0]["model"]
best_est_template = all_models[best_model_name]
log(f"\nBest learned model by held-out test MAE: {best_model_name}")

robustness_rows = []
for seed in range(10):
    gss_s = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=seed)
    tr_i2, te_i2 = next(gss_s.split(cleaned, groups=cleaned["id_instrument"]))
    tr_r2, te_r2 = cleaned.iloc[tr_i2].copy(), cleaned.iloc[te_i2].copy()
    tr_f2 = build_ratio_features(engineer_features_real(tr_r2))
    te_f2 = build_ratio_features(engineer_features_real(te_r2))
    tr_s2 = tr_f2[tr_f2["target_next_valeur_reference"].notna()].reset_index(drop=True)
    te_s2 = te_f2[te_f2["target_next_valeur_reference"].notna()].reset_index(drop=True)
    if len(te_s2) < 5 or tr_s2["id_instrument"].nunique() < 5:
        continue
    baseline_mae2 = mean_absolute_error(te_s2["target_next_valeur_reference"], te_s2["valeur_reference"])
    rng2 = np.random.default_rng(seed)
    aug2 = augment_safe(tr_s2, rng2, N_AUGMENTED_COPIES)
    train_aug2 = pd.concat([tr_s2.assign(origine="reelle"), aug2], ignore_index=True)
    from sklearn.base import clone
    pipe2 = make_pipeline(clone(best_est_template))
    pipe2.fit(train_aug2[FEATURE_COLS], train_aug2["target_delta"])
    preds2 = te_s2["valeur_reference"].values + pipe2.predict(te_s2[FEATURE_COLS])
    mae2 = mean_absolute_error(te_s2["target_next_valeur_reference"], preds2)
    robustness_rows.append({"seed": seed, "baseline_MAE": baseline_mae2, f"{best_model_name}_MAE": mae2,
                             "beats_baseline": mae2 < baseline_mae2})
robustness_df = pd.DataFrame(robustness_rows)
robustness_df.to_csv(REPORTS_QUALITY / "robustness_across_seeds.csv", index=False)
n_wins = int(robustness_df["beats_baseline"].sum())
n_total = len(robustness_df)
log(f"\nRobustness ({best_model_name} vs baseline, 10 independent instrument splits): "
    f"beats baseline in {n_wins}/{n_total} splits")
log(f"Mean baseline MAE={robustness_df['baseline_MAE'].mean():.4f} | "
    f"Mean {best_model_name} MAE={robustness_df[f'{best_model_name}_MAE'].mean():.4f}")

honest_winner = best_model_name if (n_wins > n_total / 2 and
    robustness_df[f"{best_model_name}_MAE"].mean() < robustness_df["baseline_MAE"].mean()) else "Baseline_no_drift_persistence"
log(f"\nOBJECTIVE FINAL DECISION (robustness-checked, not single-split): {honest_winner}")

with open(REPORTS_QUALITY / "benchmark_log.txt", "w") as f:
    f.write("\n".join(log_lines))

# =============================================================================
# VISUALIZATIONS
# =============================================================================
plot_df = full_results_df[full_results_df["model"] != "DummyRegressor_mean"].copy()

fig, ax = plt.subplots(figsize=(11, 7))
colors = ["#d62728" if m == "Baseline_no_drift_persistence" else ("#2ca02c" if b else "#1f77b4")
          for m, b in zip(plot_df["model"], plot_df["beats_baseline"])]
ax.barh(plot_df["model"], plot_df["MAE"], color=colors)
ax.axvline(baseline_test_mae, color="#d62728", linestyle="--", linewidth=1, label="Baseline MAE")
ax.set_xlabel("MAE (held-out real test set)")
ax.set_title("Model comparison - MAE (green = beats baseline, blue = does not, red = baseline itself)")
ax.legend()
plt.tight_layout()
plt.savefig(FIG_DIR / "01_model_mae_comparison.png", dpi=110)
plt.close()

fig, ax = plt.subplots(figsize=(11, 7))
ax.barh(plot_df["model"], plot_df["RMSE"], color="#9467bd")
ax.axvline(float(np.sqrt(mean_squared_error(y_test_abs, current_test))), color="#d62728", linestyle="--", label="Baseline RMSE")
ax.set_xlabel("RMSE (held-out real test set)")
ax.set_title("Model comparison - RMSE")
ax.legend()
plt.tight_layout()
plt.savefig(FIG_DIR / "02_model_rmse_comparison.png", dpi=110)
plt.close()

best_pipe = fitted_pipelines[best_model_name]
preds_best = current_test + best_pipe.predict(X_test)
fig, ax = plt.subplots(figsize=(7, 7))
ax.scatter(y_test_abs, preds_best, alpha=0.6, edgecolor="k", linewidth=0.3)
lims = [min(y_test_abs.min(), preds_best.min()), max(y_test_abs.max(), preds_best.max())]
ax.plot(lims, lims, "r--", label="Perfect prediction")
ax.set_xlabel("Actual next valeur_reference (real)")
ax.set_ylabel("Predicted")
ax.set_title(f"Actual vs Predicted - {best_model_name} (best by held-out MAE)")
ax.legend()
plt.tight_layout()
plt.savefig(FIG_DIR / "03_actual_vs_predicted_best_model.png", dpi=110)
plt.close()

residuals = preds_best - y_test_abs
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
axes[0].scatter(preds_best, residuals, alpha=0.6, edgecolor="k", linewidth=0.3)
axes[0].axhline(0, color="r", linestyle="--")
axes[0].set_xlabel("Predicted"); axes[0].set_ylabel("Residual (pred - actual)")
axes[0].set_title("Residuals vs Predicted")
axes[1].hist(residuals, bins=20, color="#1f77b4", edgecolor="k")
axes[1].set_xlabel("Residual"); axes[1].set_title("Residual distribution")
plt.tight_layout()
plt.savefig(FIG_DIR / "04_residual_analysis.png", dpi=110)
plt.close()

per_inst = test_sup[["id_instrument"]].copy()
per_inst["abs_error"] = np.abs(preds_best - y_test_abs)
per_inst_agg = per_inst.groupby("id_instrument")["abs_error"].agg(["mean", "count"]).reset_index()
per_inst_agg.columns = ["id_instrument", "MAE", "n_rows"]
per_inst_agg.to_csv(REPORTS_QUALITY / "per_instrument_test_results.csv", index=False)
fig, ax = plt.subplots(figsize=(9, 6))
ax.barh(per_inst_agg["id_instrument"], per_inst_agg["MAE"], color="#ff7f0e")
ax.axvline(baseline_test_mae, color="#d62728", linestyle="--", label="Overall baseline MAE")
ax.set_xlabel("MAE"); ax.set_title(f"Per-instrument MAE - {best_model_name} (held-out test)")
ax.legend()
plt.tight_layout()
plt.savefig(FIG_DIR / "05_per_instrument_performance.png", dpi=110)
plt.close()

importance = get_feature_importance(best_pipe)
if importance:
    fig, ax = plt.subplots(figsize=(8, 5))
    items = sorted(importance.items(), key=lambda kv: abs(kv[1]))
    ax.barh([k for k, v in items], [v for k, v in items], color="#2ca02c")
    ax.set_title(f"Feature importance / coefficients - {best_model_name}")
    ax.axvline(0, color="k", linewidth=0.8)
    plt.tight_layout()
    plt.savefig(FIG_DIR / "06_feature_importance_best_model.png", dpi=110)
    plt.close()
    log(f"Feature importance saved for {best_model_name}: {importance}")
else:
    log(f"{best_model_name} does not expose coef_/feature_importances_ - no importance plot generated")

log("\nAll figures saved under reports/figures/05_regression_model/")
print("\nBENCHMARK COMPLETE")
