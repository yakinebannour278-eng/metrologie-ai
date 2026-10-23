"""
audit_v3/01_build_features_and_baselines.py
=============================================
Rebuilds features and targets directly from ILA_cleaned.csv (not from the
already-engineered ILA_features_selected.csv), independently, to verify
temporal causality from scratch rather than trusting the prior pipeline.

Grouping: by id_instrument, ordered by (annee, original row order) - same
grouping the existing deployed pipeline uses (chrono_rank_in_instrument).
This is a design choice (not per instrument+point) - documented, not hidden.

No calibration-date features are built: no date column beyond `annee`
(whole years) exists in the source data - confirmed by inspecting
ILA_features_selected.csv columns directly.
"""
import numpy as np
import pandas as pd

RANDOM_STATE = 42

df = pd.read_csv("data/processed/ILA_cleaned.csv", encoding="utf-8-sig")
df = df.sort_values(["id_instrument", "annee", "chrono_rank_in_instrument"]).reset_index(drop=True)

g = df.groupby("id_instrument", group_keys=False)

# --- Causal, past-only temporal features ---
df["lag_1"] = g["valeur_reference"].shift(1)
df["lag_2"] = g["valeur_reference"].shift(2)
df["lag_3"] = g["valeur_reference"].shift(3)
df["delta_1"] = df["valeur_reference"] - df["lag_1"]  # most recent observed drift (available at prediction time)
df["years_since_previous"] = df["annee"] - g["annee"].shift(1)

# rolling/expanding over PAST values only (shift(1) first, then window)
past_val = g["valeur_reference"].shift(1)
df["rolling_mean_3"] = past_val.groupby(df["id_instrument"]).transform(lambda s: s.rolling(3, min_periods=1).mean())
df["rolling_std_3"] = past_val.groupby(df["id_instrument"]).transform(lambda s: s.rolling(3, min_periods=2).std())
df["expanding_mean"] = past_val.groupby(df["id_instrument"]).transform(lambda s: s.expanding(min_periods=1).mean())
df["expanding_std"] = past_val.groupby(df["id_instrument"]).transform(lambda s: s.expanding(min_periods=2).std())
df["ewm_mean"] = past_val.groupby(df["id_instrument"]).transform(lambda s: s.ewm(span=3, min_periods=1).mean())

# historical slope: linear fit of valeur_reference vs annee, using strictly past points
def historical_slope(sub):
    slopes = [np.nan]
    for i in range(1, len(sub)):
        past = sub.iloc[:i]
        if len(past) >= 2 and past["annee"].nunique() >= 2:
            coef = np.polyfit(past["annee"], past["valeur_reference"], 1)[0]
        else:
            coef = np.nan
        slopes.append(coef)
    return pd.Series(slopes, index=sub.index)

df["historical_slope"] = df.groupby("id_instrument", group_keys=False).apply(
    lambda sub: historical_slope(sub.sort_values(["annee", "chrono_rank_in_instrument"]))
)

# --- Metrology features (known AT the current measurement, not future) ---
df["distance_to_EMT_lower"] = df["valeur_reference"] - df["EMT_basse"]
df["distance_to_EMT_upper"] = df["EMT_haute"] - df["valeur_reference"]
emt_width = (df["EMT_haute"] - df["EMT_basse"]).replace(0, np.nan)
df["distance_to_EMT_center"] = df["valeur_reference"] - (df["EMT_basse"] + df["EMT_haute"]) / 2
df["normalized_distance_to_tolerance"] = df["distance_to_EMT_center"] / emt_width
df["uncertainty_ratio"] = df["incertitude"] / emt_width
df["measurement_width"] = df["max"] - df["min"]

# --- Instrument-history / conformity features (past only) ---
df["n_previous_observations"] = df["chrono_rank_in_instrument"]
conforme_bin = (df["conforme"] == "non conforme").astype(float)
past_conforme = conforme_bin.groupby(df["id_instrument"]).shift(1)
df["previous_conformity_nonconforme"] = past_conforme
df["historical_nonconformity_rate"] = past_conforme.groupby(df["id_instrument"]).transform(
    lambda s: s.expanding(min_periods=1).mean()
)

# --- Targets ---
next_val = g["valeur_reference"].shift(-1)
df["target_A_next_value"] = next_val
df["target_B_drift"] = next_val - df["valeur_reference"]

supervised = df[df["target_A_next_value"].notna()].copy()
print(f"Total rows: {len(df)}  Supervised rows (has a next measurement): {len(supervised)}")
print(f"Instruments: {df['id_instrument'].nunique()}")

supervised.to_csv("experiments/audit_v3/features_v3.csv", index=False)
print("Saved experiments/audit_v3/features_v3.csv")
print()
print("Feature candidate list (causal, verified past-only):")
FEATURE_CANDIDATES = [
    "lag_1", "lag_2", "lag_3", "delta_1", "years_since_previous",
    "rolling_mean_3", "rolling_std_3", "expanding_mean", "expanding_std", "ewm_mean",
    "historical_slope",
    "distance_to_EMT_lower", "distance_to_EMT_upper", "distance_to_EMT_center",
    "normalized_distance_to_tolerance", "uncertainty_ratio", "measurement_width",
    "n_previous_observations", "previous_conformity_nonconforme", "historical_nonconformity_rate",
    "valeur_reference", "consigne", "incertitude", "EMT_basse", "EMT_haute", "min", "max",
]
print(FEATURE_CANDIDATES)
print(f"n={len(FEATURE_CANDIDATES)} candidate features")
