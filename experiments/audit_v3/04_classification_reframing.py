"""
audit_v3/04_classification_reframing.py
==========================================
Reframes the problem as classification: will the instrument's NEXT
measurement be non-conforme? Uses the same causal features and the same
chronological validation methodology as the regression experiments, for
direct comparability.

Two baselines (not one) are required for a fair classification comparison:
  - Baseline A: majority class (always predict "conforme", the more common
    class - 78% of rows). This is the classification equivalent of a naive
    predictor and must be beaten for any model to be considered useful.
  - Baseline B: persistence (predict next conformity = current conformity).
    The direct classification analog of the regression "no drift" baseline.
"""
import warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

RANDOM_STATE = 42

df = pd.read_csv("data/processed/ILA_cleaned.csv", encoding="utf-8-sig")
df = df.sort_values(["id_instrument", "annee", "chrono_rank_in_instrument"]).reset_index(drop=True)
g = df.groupby("id_instrument", group_keys=False)

# --- Same causal features as the regression experiment (rebuilt here for independence) ---
df["lag_1"] = g["valeur_reference"].shift(1)
df["delta_1"] = df["valeur_reference"] - df["lag_1"]
past_val = g["valeur_reference"].shift(1)
df["rolling_mean_3"] = past_val.groupby(df["id_instrument"]).transform(lambda s: s.rolling(3, min_periods=1).mean())
df["expanding_mean"] = past_val.groupby(df["id_instrument"]).transform(lambda s: s.expanding(min_periods=1).mean())
df["ewm_mean"] = past_val.groupby(df["id_instrument"]).transform(lambda s: s.ewm(span=3, min_periods=1).mean())
emt_width = (df["EMT_haute"] - df["EMT_basse"]).replace(0, np.nan)
df["distance_to_EMT_center"] = df["valeur_reference"] - (df["EMT_basse"] + df["EMT_haute"]) / 2
df["normalized_distance_to_tolerance"] = df["distance_to_EMT_center"] / emt_width
df["uncertainty_ratio"] = df["incertitude"] / emt_width
df["n_previous_observations"] = df["chrono_rank_in_instrument"]
conforme_bin = (df["conforme"] == "non conforme").astype(float)
df["current_nonconforme"] = conforme_bin  # available at prediction time - it's the CURRENT status
past_conforme = conforme_bin.groupby(df["id_instrument"]).shift(1)
df["historical_nonconformity_rate"] = past_conforme.groupby(df["id_instrument"]).transform(
    lambda s: s.expanding(min_periods=1).mean()
)

# --- Classification target: will the NEXT measurement be non-conforme? ---
df["target_next_nonconforme"] = conforme_bin.groupby(df["id_instrument"]).shift(-1)

supervised = df[df["target_next_nonconforme"].notna()].copy()
print(f"Supervised rows: {len(supervised)}  Instruments: {supervised['id_instrument'].nunique()}")
print(f"Target class balance (next_nonconforme): {supervised['target_next_nonconforme'].value_counts(normalize=True).round(3).to_dict()}")
print()

FEATURES = ["lag_1", "delta_1", "rolling_mean_3", "expanding_mean", "ewm_mean",
            "distance_to_EMT_center", "normalized_distance_to_tolerance",
            "uncertainty_ratio", "n_previous_observations", "historical_nonconformity_rate",
            "current_nonconforme"]
X = supervised[FEATURES]
y = supervised["target_next_nonconforme"].values.astype(int)
current_status = supervised["current_nonconforme"].values.astype(int)

years_sorted = np.sort(supervised["annee"].unique())
cutoff_year = years_sorted[int(len(years_sorted) * 0.7)]
train_mask = (supervised["annee"] <= cutoff_year).values
test_mask = (supervised["annee"] > cutoff_year).values
print(f"Chronological split: train years <= {cutoff_year} (n={train_mask.sum()}), test years > {cutoff_year} (n={test_mask.sum()})")
print(f"Test set class balance: {pd.Series(y[test_mask]).value_counts(normalize=True).round(3).to_dict()}")
print()


def eval_clf(y_true, y_pred, y_proba, label):
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    try:
        auc = roc_auc_score(y_true, y_proba) if y_proba is not None else float("nan")
    except ValueError:
        auc = float("nan")
    print(f"{label:35s} acc={acc:.3f} precision={prec:.3f} recall={rec:.3f} f1={f1:.3f} AUC={auc:.3f}")
    return {"label": label, "accuracy": acc, "precision": prec, "recall": rec, "f1": f1, "AUC": auc}


results = []

# --- Baseline A: majority class ---
pred_majority = np.zeros(test_mask.sum(), dtype=int)  # always predict "conforme" (0 = not non-conforme)
results.append(eval_clf(y[test_mask], pred_majority, None, "BaselineA_majority_class"))

# --- Baseline B: persistence (next status = current status) ---
pred_persist = current_status[test_mask]
results.append(eval_clf(y[test_mask], pred_persist, pred_persist.astype(float), "BaselineB_persistence"))

print()
print("=== Real classifiers, chronological validation ===")
MODELS = {
    "LogisticRegression": LogisticRegression(max_iter=2000, random_state=RANDOM_STATE, class_weight="balanced"),
    "RandomForest": RandomForestClassifier(n_estimators=200, random_state=RANDOM_STATE, class_weight="balanced", n_jobs=-1),
    "GradientBoosting": GradientBoostingClassifier(random_state=RANDOM_STATE),
}
for name, model in MODELS.items():
    pipe = Pipeline([("imp", SimpleImputer(strategy="median")), ("sc", StandardScaler()), ("m", model)])
    pipe.fit(X[train_mask], y[train_mask])
    pred = pipe.predict(X[test_mask])
    proba = pipe.predict_proba(X[test_mask])[:, 1]
    results.append(eval_clf(y[test_mask], pred, proba, name))

print()
res_df = pd.DataFrame(results)
res_df.to_csv("experiments/audit_v3/classification_comparison.csv", index=False)

baseline_f1 = res_df[res_df["label"] == "BaselineB_persistence"]["f1"].iloc[0]
baseline_acc = res_df[res_df["label"] == "BaselineB_persistence"]["accuracy"].iloc[0]
print(f"Baseline B (persistence) F1={baseline_f1:.3f}, accuracy={baseline_acc:.3f}")
beats = res_df[(res_df["label"].isin(MODELS.keys())) & (res_df["f1"] > baseline_f1)]
print(f"Models beating persistence baseline on F1: {len(beats)} / {len(MODELS)}")
if len(beats):
    print(beats[["label", "f1", "accuracy", "AUC"]].to_string(index=False))
