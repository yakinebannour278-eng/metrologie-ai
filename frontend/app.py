"""
frontend/app.py
================
Metrologist-facing MVP interface. Talks to the FastAPI backend over HTTP
ONLY — no model loading, no preprocessing, no business logic here.

Run:
    streamlit run frontend/app.py

Requires the API to be running (see README.md):
    uvicorn src.api.main:app --reload
"""
import os
from datetime import datetime

import plotly.graph_objects as go
import requests
import streamlit as st

API_BASE_URL = os.environ.get("METROLOGY_API_URL", "http://127.0.0.1:8000")

st.set_page_config(page_title="Metrology Drift Prediction", page_icon="📐", layout="wide")

HISTORICAL_FIELDS = [
    "rolling_min_3", "rolling_max_3", "rolling_mean_3",
    "rolling_median_3", "moving_average_5", "ewm_mean",
]
OBSERVABLE_FIELDS = ["EMT_basse", "EMT_haute", "min", "max", "consigne", "valeur_reference"]


def api_get(path: str, **kwargs):
    try:
        r = requests.get(f"{API_BASE_URL}{path}", timeout=10, **kwargs)
        return r
    except requests.exceptions.ConnectionError:
        st.error(f"Cannot reach the API at {API_BASE_URL}. Is it running? (`uvicorn src.api.main:app --reload`)")
        st.stop()


def api_post(path: str, json: dict):
    try:
        r = requests.post(f"{API_BASE_URL}{path}", json=json, timeout=15)
        return r
    except requests.exceptions.ConnectionError:
        st.error(f"Cannot reach the API at {API_BASE_URL}. Is it running? (`uvicorn src.api.main:app --reload`)")
        st.stop()


def build_trend_chart(history: list[dict], predicted_year: int, predicted_value: float) -> go.Figure:
    """Full trend chart for a metrologist: observed reference values across
    past campaigns, the EMT tolerance band, the min/max interval, the
    consigne (setpoint), and the newly predicted point — clearly
    distinguished from observed measurements, per the project rule that a
    model prediction must never be visually confused with an observation."""
    years = [h["annee"] for h in history]
    ref = [h.get("valeur_reference") for h in history]
    emt_basse = [h.get("EMT_basse") for h in history]
    emt_haute = [h.get("EMT_haute") for h in history]
    mins = [h.get("min") for h in history]
    maxs = [h.get("max") for h in history]
    consigne = [h.get("consigne") for h in history]

    fig = go.Figure()

    # EMT tolerance band (wider, lighter fill)
    if any(v is not None for v in emt_haute) and any(v is not None for v in emt_basse):
        fig.add_trace(go.Scatter(x=years, y=emt_haute, mode="lines", line=dict(width=0),
                                  showlegend=False, hoverinfo="skip"))
        fig.add_trace(go.Scatter(x=years, y=emt_basse, mode="lines", line=dict(width=0),
                                  fill="tonexty", fillcolor="rgba(255,99,71,0.10)",
                                  name="EMT tolerance band", hoverinfo="skip"))

    # Min/max interval (narrower band)
    if any(v is not None for v in maxs) and any(v is not None for v in mins):
        fig.add_trace(go.Scatter(x=years, y=maxs, mode="lines", line=dict(width=0),
                                  showlegend=False, hoverinfo="skip"))
        fig.add_trace(go.Scatter(x=years, y=mins, mode="lines", line=dict(width=0),
                                  fill="tonexty", fillcolor="rgba(92,79,207,0.15)",
                                  name="Min/Max interval", hoverinfo="skip"))

    # Consigne (setpoint)
    if any(v is not None for v in consigne):
        fig.add_trace(go.Scatter(x=years, y=consigne, mode="lines", line=dict(dash="dot", color="gray"),
                                  name="Consigne (setpoint)"))

    # Observed reference values (real measurements)
    fig.add_trace(go.Scatter(x=years, y=ref, mode="lines+markers",
                              line=dict(color="#5C4FCF", width=2), marker=dict(size=8),
                              name="Observed valeur_reference"))

    # Predicted point — visually distinct (star marker, different color, dashed connector)
    if ref and ref[-1] is not None:
        fig.add_trace(go.Scatter(x=[years[-1], predicted_year], y=[ref[-1], predicted_value],
                                  mode="lines", line=dict(color="#E4572E", width=2, dash="dash"),
                                  showlegend=False, hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=[predicted_year], y=[predicted_value], mode="markers",
                              marker=dict(size=16, symbol="star", color="#E4572E",
                                          line=dict(color="white", width=1)),
                              name="Predicted (model) — not an observation"))

    fig.update_layout(
        title="Instrument trend: observed measurements vs. model prediction",
        xaxis_title="Year", yaxis_title="Value",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
        margin=dict(t=60, b=40),
        height=420,
    )
    return fig


# ---------------------------------------------------------------------------
# Sidebar: system status
# ---------------------------------------------------------------------------
with st.sidebar:
    st.header("System status")
    health = api_get("/health")
    if health.status_code == 200:
        h = health.json()
        if h["model_loaded"]:
            st.success("Model loaded")
        else:
            st.error("Model NOT loaded — predictions unavailable")
    else:
        st.error("API unreachable")

    st.caption(f"API: {API_BASE_URL}")
    st.divider()
    st.caption(
        "This tool is a decision-support aid for a metrologist. "
        "It does not automatically replace professional judgement and does not "
        "by itself constitute ISO/IEC 17025 evidence of conformity."
    )

st.title("📐 Metrology Drift Prediction")
st.caption("AI Pipeline for Drift Prediction — regression module (L'Épi d'Or, ISO/IEC 17025)")

info_resp = api_get("/model/info")
model_info = info_resp.json() if info_resp.status_code == 200 else None

if model_info is None:
    st.error("Model information unavailable. Cannot build the prediction form until the API reports a loaded model.")
    st.stop()

with st.expander("ℹ️ About this model — read before using", expanded=False):
    st.markdown(f"**Model:** `{model_info['model_name']}` — version `{model_info['model_version']}`")
    st.markdown(f"**Target predicted:** `{model_info['target_variable']}` — {model_info['target_description']}")
    st.markdown(f"**Prediction unit:** {model_info['prediction_unit']}")
    st.markdown("**Evaluation (GroupKFold, out-of-sample):**")
    st.json(model_info["evaluation_metrics_groupkfold"])
    st.markdown("**Known limitations:**")
    for lim in model_info["known_limitations"]:
        st.warning(lim)

tab_predict, tab_history = st.tabs(["🔮 New prediction", "🗂️ Prediction history"])

with tab_predict:
    st.subheader("1. Instrument information")
    instruments = api_get("/model/instruments").json()
    col1, col2, col3 = st.columns(3)
    with col1:
        if instruments:
            instrument_id = st.selectbox("Instrument ID *", options=instruments, index=None, placeholder="Select an instrument…")
            instrument_id = instrument_id or ""
        else:
            instrument_id = st.text_input("Instrument ID *", placeholder="e.g. INST-03B-CHAMBREDEFERMENTATION")
            st.caption("No instrument list available from the API — enter the exact ID manually.")
    with col2:
        year = st.number_input("Measurement year *", min_value=2000, max_value=2035, value=datetime.now().year, step=1)
    with col3:
        point = st.text_input("Measurement point", placeholder="optional")

    if instrument_id.strip():
        if st.button("📥 Load last known values for this instrument"):
            resp = api_get(
                f"/model/instruments/{instrument_id.strip()}/last-known-features",
                params={"year": int(year)},
            )
            if resp.status_code == 200:
                st.session_state["prefill"] = resp.json()
                st.success(f"Loaded last known values from before {int(year)} (campaign year {resp.json().get('source_year')}).")
            elif resp.status_code == 404:
                st.info(resp.json().get("detail", "No historical data available — enter values manually below."))
            else:
                st.error(f"Could not load history: {resp.text}")

    prefill = st.session_state.get("prefill", {})

    st.subheader("2. Metrology parameters (observed at this measurement)")
    c1, c2, c3 = st.columns(3)
    with c1:
        emt_basse = st.number_input("EMT lower bound *", value=float(prefill.get("EMT_basse") or 0.0), format="%.4f")
        emt_haute = st.number_input("EMT upper bound *", value=float(prefill.get("EMT_haute") or 0.0), format="%.4f")
    with c2:
        min_val = st.number_input("Min *", value=float(prefill.get("min") or 0.0), format="%.4f")
        max_val = st.number_input("Max *", value=float(prefill.get("max") or 0.0), format="%.4f")
    with c3:
        consigne = st.number_input("Consigne (nominal/setpoint, optional)", value=float(prefill.get("consigne") or 0.0), format="%.4f")
        valeur_reference = st.number_input("Reference value (current) *", value=float(prefill.get("valeur_reference") or 0.0), format="%.4f")

    st.subheader("3. Historical features")
    st.caption(
        "Computed from the instrument's measurement history (rolling/lag statistics). "
        "Use **Load last known values** above to pre-fill, then review before submitting."
    )
    h1, h2, h3 = st.columns(3)
    with h1:
        rolling_min_3 = st.number_input("Rolling min (last 3)", value=float(prefill.get("rolling_min_3") or 0.0), format="%.4f")
        rolling_max_3 = st.number_input("Rolling max (last 3)", value=float(prefill.get("rolling_max_3") or 0.0), format="%.4f")
    with h2:
        rolling_mean_3 = st.number_input("Rolling mean (last 3)", value=float(prefill.get("rolling_mean_3") or 0.0), format="%.4f")
        rolling_median_3 = st.number_input("Rolling median (last 3)", value=float(prefill.get("rolling_median_3") or 0.0), format="%.4f")
    with h3:
        moving_average_5 = st.number_input("Moving average (last 5)", value=float(prefill.get("moving_average_5") or 0.0), format="%.4f")
        ewm_mean = st.number_input("Exp. weighted mean", value=float(prefill.get("ewm_mean") or 0.0), format="%.4f")

    st.subheader("4. Prediction")
    if st.button("▶️ Generate prediction", type="primary"):
        if not instrument_id.strip():
            st.error("Instrument ID is required.")
        else:
            payload = {
                "instrument_id": instrument_id.strip(),
                "year": int(year),
                "point": point.strip() or None,
                "features": {
                    "EMT_basse": emt_basse, "EMT_haute": emt_haute,
                    "min": min_val, "max": max_val,
                    "consigne": consigne, "valeur_reference": valeur_reference,
                    "rolling_min_3": rolling_min_3, "rolling_max_3": rolling_max_3,
                    "rolling_mean_3": rolling_mean_3, "rolling_median_3": rolling_median_3,
                    "moving_average_5": moving_average_5, "ewm_mean": ewm_mean,
                },
            }
            resp = api_post("/predict", json=payload)
            if resp.status_code == 200:
                result = resp.json()

                if result["history_status"] == "INSUFFICIENT_HISTORY":
                    st.error(
                        f"⚠️ INSUFFICIENT HISTORY: this instrument has only "
                        f"{result['n_prior_observations']} prior recorded observation(s). "
                        "No confident prediction is provided — a metrologist's direct "
                        "judgement is required for this instrument."
                    )
                    st.info(result["interpretation"])
                else:
                    if result["recommended_source"] == "baseline_persistence":
                        st.success("Recommendation generated (baseline)")
                        m1, m2, m3 = st.columns(3)
                        m1.metric(
                            "✅ RECOMMENDED value",
                            f"{result['recommended_prediction']:.4f}",
                            help="Persistence baseline: assumes no change. Extensive testing found "
                                 "no ML model that reliably beats this on this dataset.",
                        )
                        m2.metric(
                            "ML model (reference only, not recommended)",
                            f"{result['prediction']:.4f}",
                        )
                        m3.metric("Prior observations", result["n_prior_observations"])
                        st.warning(
                            "The trained ML model has NOT been shown to beat this simple baseline "
                            "(see experiments/audit_v3/FINAL_REPORT.md — 14 algorithms tested, 0 beat it). "
                            "It is shown above for transparency only."
                        )
                    else:
                        st.success("Prediction generated (model verified to beat baseline)")
                        m1, m2, m3 = st.columns(3)
                        m1.metric("✅ RECOMMENDED value (ML model)", f"{result['recommended_prediction']:.4f}")
                        m2.metric("Baseline (for reference)", f"{result['baseline_prediction']:.4f}")
                        m3.metric("Prior observations", result["n_prior_observations"])
                    st.info(result["interpretation"])

                    hist_resp = api_get(f"/model/instruments/{instrument_id.strip()}/history")
                    if hist_resp.status_code == 200:
                        history = hist_resp.json()
                        fig = build_trend_chart(history, predicted_year=int(year), predicted_value=result["recommended_prediction"])
                        st.plotly_chart(fig, use_container_width=True)
                        st.caption(
                            "⭐ Star = the recommended value plotted above, not an observed measurement. "
                            "Historical points are averaged per year across the instrument's measurement points. "
                            "Shaded bands: EMT tolerance (red) and min/max interval (purple)."
                        )
                    else:
                        st.caption("No historical series available to chart for this instrument.")

                with st.expander("Full response"):
                    st.json(result)
            else:
                st.error(f"Prediction failed ({resp.status_code}): {resp.text}")

with tab_history:
    st.subheader("Local prediction history")
    filter_id = st.text_input("Filter by instrument ID", key="hist_filter")
    resp = api_get("/history", params={"instrument_id": filter_id or None, "limit": 100})
    if resp.status_code == 200:
        rows = resp.json()
        if rows:
            st.dataframe(rows, use_container_width=True)
        else:
            st.info("No predictions recorded yet.")
    else:
        st.error(f"Could not load history: {resp.text}")
