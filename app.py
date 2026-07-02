# =============================================================================
# Heat-Stress Neuroinflammation Simulator (Educational, Non-Clinical)
# =============================================================================
# Full documentation, honesty notes, parameter sources, and version history
# live in README.md in this repository -- kept there instead of a long
# docstring here so this file stays simple to copy/paste/upload without
# risk of a multi-line string block getting damaged in transfer.
#
# ONE-LINE SUMMARY: combines real historical reanalysis weather data with an
# illustrative ODE model of a proposed heat-stress -> BBB -> microglial-M1
# pathway. NOT a diagnostic or clinical tool. See README.md for details.
#
# Run with: streamlit run app.py
# =============================================================================

import streamlit as st
import requests
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.integrate import odeint

# ---------------------------------------------------------------------------
# GLOBAL CONFIGURATION
# ---------------------------------------------------------------------------
st.set_page_config(page_title="Heat-Stress Neuroinflammation Simulator (Educational)", layout="wide")

st.title("Heat-Stress Neuroinflammation Simulator")
st.markdown("""
**Author:** Tasnim | Independent student computational project (M.Sc. thesis-adjacent, not part of the official thesis)

This is an educational, hypothesis-generating simulation. It combines real historical
reanalysis weather data with an illustrative mathematical model of a proposed
heat-stress → blood-brain-barrier → microglial-activation pathway, based on the
general direction of published literature. The specific numeric rate constants
used are estimates chosen to produce plausible dynamics — they are **not** fitted
to patient data, biomarker measurements, or any specific study's reported values.
""")

st.warning(
    "**This tool is NOT a diagnostic, clinical, or predictive medical instrument.** "
    "It does not use real patient data, is not clinically validated, and should not be "
    "used to assess real health risk for any individual or location. It is a learning "
    "and portfolio project illustrating mechanistic simulation methods.",
    icon="⚠️",
)

with st.expander("Model parameter sources and honesty notes (click to expand)"):
    st.markdown("""
| Parameter | Value | Status | Source / derivation |
|---|---|---|---|
| BBB recovery rate (no active stressor) | 0.28 / day | **Literature-derived** | Yang et al., claudin-5/occludin/ZO-1 BBB permeability biphasic recovery over ~120h (5 days) post-injury → half-life ≈2.5 days → k = ln(2)/2.5 = 0.28/day |
| M1 microglial recovery rate (no active stressor) | 0.23 / day | **Literature-derived** | [18F]DPA-714 PET imaging, LPS-induced microglial activation peaking at 24h and returning to baseline by 72h → half-life ≈3 days → k = ln(2)/3 = 0.23/day |
| BBB / M1 recovery rate (under ongoing stress) | 0.05 / 0.08 per day | **Illustrative assumption** | No study reports a specific suppression factor for recovery *during* continued stress; hand-set to represent slowed repair. |
| BBB/M1 "gain" constants (per scenario) | 0.08–0.36 | **Illustrative, direction-only** | No paper reports a rate constant in this model's unit (permeability-fraction per °C-heat-index per day) — that unit doesn't exist outside this toy model. Cited papers (Montagne et al.; Perry & Holmes) support the *qualitative direction* only (e.g. aging reduces BBB integrity), not the numeric value. |
| Hemodynamic multiplier | 1.0 / 1.35 / 2.10 | **Illustrative** | Not derived from a specific source; represents a rough relative scaling assumption. |

**Why this matters:** a model that claims every number is "from the literature" when only some are is a bigger credibility risk than a model that's upfront about which parameters are grounded and which are estimates. This table exists so nobody has to take that on faith.
""")


# ---------------------------------------------------------------------------
# REVERSE GEOCODING UTILITY (unchanged logic, just a label fix)
# ---------------------------------------------------------------------------
def get_location_name(latitude, longitude):
    """Look up a human-readable place name for display purposes only."""
    try:
        url = f"https://api.open-meteo.com/v1/forecast?latitude={latitude}&longitude={longitude}&current=temperature_2m&timezone=auto"
        res = requests.get(url, timeout=10).json()
        timezone = res.get('timezone', 'Unknown Location')
        if "/" in timezone:
            return timezone.split("/")[-1].replace("_", " ")
        return timezone
    except Exception:
        return f"Coordinates ({latitude}, {longitude})"

# ---------------------------------------------------------------------------
# SIDEBAR CONTROLS
# ---------------------------------------------------------------------------
st.sidebar.header("Simulation Mode")
app_mode = st.sidebar.radio("Select view:", ["Single Location Deep-Dive", "Two-Location Comparison"])

st.sidebar.header("Timeline")
start_date = st.sidebar.date_input("Start Date", pd.to_datetime("2025-07-01"))
end_date = st.sidebar.date_input("End Date", pd.to_datetime("2025-07-20"))

st.sidebar.header("Hemodynamic Scenario")
blood_pressure_state = st.sidebar.selectbox(
    "Simulated blood-pressure scenario (illustrative multiplier, not measured):",
    ["Normotensive (~120 mmHg systolic)", "Elevated (~135 mmHg systolic)", "Severely Elevated (~180 mmHg systolic)"]
)
shear_stress_multiplier = {
    "Normotensive (~120 mmHg systolic)": 1.0,
    "Elevated (~135 mmHg systolic)": 1.35,
    "Severely Elevated (~180 mmHg systolic)": 2.10,
}[blood_pressure_state]

st.sidebar.header("Simulated Physiological Scenario")
st.sidebar.caption(
    "These are illustrative preset scenarios, not real patient records or measured cohort data."
)
cohort_profile = st.sidebar.selectbox(
    "Select a preset scenario:",
    [
        "Baseline (young, no known risk factors)",
        "Older-age scenario",
        "Chronic metabolic/vascular risk scenario (e.g. diabetes/hypertension pattern)",
        "Compounded high-risk scenario (older age + metabolic/vascular risk)",
    ]
)

# NOTE on rationale text below: these describe the *direction* of published
# findings (older age and chronic vascular/metabolic conditions are associated
# with reduced BBB integrity and more reactive microglia). The specific gain
# values (0.08-0.36 etc.) are this project's own illustrative estimates for
# producing plausible relative differences between scenarios -- they are NOT
# values reported in these papers. Citations describe the qualitative
# reasoning, not the source of the number.
if cohort_profile == "Baseline (young, no known risk factors)":
    bbb_gain, m1_gain = 0.08, 0.12
    rationale = "Modeled as intact tight junctions and low baseline microglial reactivity."
    lit_context = "General direction consistent with Montagne et al. (Neuron, 2017) on age-related BBB integrity."
elif cohort_profile == "Older-age scenario":
    bbb_gain, m1_gain = 0.16, 0.15
    rationale = "Modeled with reduced tight-junction integrity, reflecting reported age-related BBB vulnerability."
    lit_context = "General direction consistent with Montagne et al. (Nature Medicine, 2015) on age-related BBB breakdown."
elif cohort_profile == "Chronic metabolic/vascular risk scenario (e.g. diabetes/hypertension pattern)":
    bbb_gain, m1_gain = 0.12, 0.30
    rationale = "Modeled with a more reactive microglial baseline, reflecting reported chronic low-grade vascular inflammation."
    lit_context = "General direction consistent with Perry & Holmes (Nature Reviews Neurology, 2014) on primed microglia."
else:
    bbb_gain, m1_gain = 0.24, 0.36
    rationale = "Modeled as the combination of the two scenarios above (illustrative compounding, not a validated additive model)."
    lit_context = "Combined illustrative estimate; not itself drawn from a single source."

st.sidebar.info(f"**Modeling rationale (illustrative, not fitted):** {rationale}\n\n*Qualitative literature context:* {lit_context}")

# ---------------------------------------------------------------------------
# BACKEND: FETCH REAL REANALYSIS WEATHER DATA + RUN THE ODE MODEL
# ---------------------------------------------------------------------------
@st.cache_data
def fetch_and_model(latitude, longitude, s_date, e_date, b_gain, m_gain, shear_mult):
    """
    Fetch real historical daily weather data (Open-Meteo archive API, built on
    ECMWF ERA5 reanalysis) for the given coordinates and date range, compute a
    simple heat-stress index, and integrate the illustrative two-variable ODE
    model (BBB permeability, microglial M1 activation) forced by that index.
    """
    url = (
        f"https://archive-api.open-meteo.com/v1/archive?"
        f"latitude={latitude}&longitude={longitude}&start_date={s_date}&end_date={e_date}"
        f"&daily=temperature_2m_max,relative_humidity_2m_max&timezone=auto"
    )
    try:
        res = requests.get(url, timeout=15).json()
        if 'daily' not in res:
            return None
    except Exception:
        return None

    daily_data = res['daily']
    df = pd.DataFrame({
        'Date': pd.to_datetime(daily_data['time']),
        'Max_Temp': daily_data['temperature_2m_max'],
        'Max_Humidity': daily_data['relative_humidity_2m_max']
    })

    # Simplified heat index (Steadman-style approximation) -- a real, standard
    # meteorological formula, not something invented for this project.
    df['Heat_Stress_Index'] = df['Max_Temp'] + (0.55 * (df['Max_Humidity'] / 100) * (df['Max_Temp'] - 14.5))
    df['Anomaly'] = np.clip(df['Heat_Stress_Index'] - 25, 0, None)
    days_timeline = np.arange(len(df))

    def internal_ode(y, t, anomaly_data, bg, mg, sm):
        """
        y[0] = BBB permeability (0 = intact, 1 = fully permeable), illustrative
        y[1] = Microglial M1 activation fraction (0 to 1), illustrative

        BUGFIX vs. previous version: removed a constant "+0.5" offset that was
        previously added to the anomaly term. That offset meant the model kept
        driving BBB permeability upward even when anomaly = 0 (no heat stress),
        which is not the intended behavior -- the barrier should recover toward
        baseline when there is no stressor. Now it does.

        RECOVERY RATE CONSTANTS -- literature-derived, math shown:

        k_bbb_recovery (no ongoing anomaly) = 0.28 / day
            Source: Yang et al., "MMP-Mediated Disruption of Claudin-5 in the
            Blood-Brain Barrier of Rat Brain After Cerebral Ischemia" -- reports
            a biphasic BBB permeability pattern resolving over up to 120 hours
            (5 days) post-injury once the acute insult subsides. Approximate
            structural recovery half-life ~2.5 days.
            First-order rate constant: k = ln(2) / t_half = 0.693 / 2.5 = 0.28/day.

        k_m1_recovery (no ongoing anomaly) = 0.23 / day
            Source: Zhang et al. (PMC9392970), [18F]DPA-714 PET imaging of
            LPS-induced microglial activation in mice -- activation peaked at
            24h post-injection and had returned to baseline by 72h. Approximate
            recovery half-life ~3 days.
            First-order rate constant: k = ln(2) / t_half = 0.693 / 3.0 = 0.23/day.

        The SUPPRESSED recovery rates used while an anomaly is still present
        (0.05 and 0.08 below) are NOT literature-derived -- no study reports a
        specific suppression factor for "recovery rate while the stressor is
        still active." These remain hand-set illustrative assumptions, labeled
        as such rather than attributed to a source that doesn't support them.
        """
        idx = int(np.clip(t, 0, len(anomaly_data) - 1))
        dt_anomaly = anomaly_data[idx]
        BBB_perm, M1_activation = y[0], y[1]

        k_bbb_recovery = 0.28 if dt_anomaly == 0 else 0.05   # 0.28 = literature-derived; 0.05 = illustrative assumption
        k_m1_recovery = 0.23 if dt_anomaly == 0 else 0.08    # 0.23 = literature-derived; 0.08 = illustrative assumption

        d_BBB_dt = (bg * dt_anomaly * sm * (1.0 - BBB_perm)) - (k_bbb_recovery * BBB_perm)
        d_M1_dt = (mg * BBB_perm * (1.0 - M1_activation)) - (k_m1_recovery * M1_activation)
        return [d_BBB_dt, d_M1_dt]

    initial_states = [0.05, 0.01]

    sol_std = odeint(internal_ode, initial_states, days_timeline, args=(df['Anomaly'].values, b_gain, m_gain, shear_mult))
    sol_low = odeint(internal_ode, initial_states, days_timeline, args=(df['Anomaly'].values, b_gain * 0.8, m_gain * 0.8, shear_mult))
    sol_high = odeint(internal_ode, initial_states, days_timeline, args=(df['Anomaly'].values, b_gain * 1.2, m_gain * 1.2, shear_mult))

    df['BBB_Leakage'] = sol_std[:, 0]
    df['Microglia_M1'] = sol_std[:, 1]
    df['BBB_Low'], df['BBB_High'] = sol_low[:, 0], sol_high[:, 0]
    df['M1_Low'], df['M1_High'] = sol_low[:, 1], sol_high[:, 1]

    return df


# ---------------------------------------------------------------------------
# FRONTEND: SINGLE LOCATION VIEW
# ---------------------------------------------------------------------------
if app_mode == "Single Location Deep-Dive":
    st.header("Single Location Deep-Dive")
    col_input1, col_input2 = st.columns(2)
    with col_input1:
        lat = st.number_input("Latitude", value=36.8065, format="%.4f")
    with col_input2:
        lon = st.number_input("Longitude", value=10.1815, format="%.4f")

    city_name = get_location_name(lat, lon)

    if st.button("Run Simulation"):
        with st.spinner("Fetching reanalysis weather data and integrating model..."):
            data = fetch_and_model(lat, lon, start_date, end_date, bbb_gain, m1_gain, shear_stress_multiplier)
            if data is not None:
                c1, c2 = st.columns([2, 1])
                with c1:
                    st.subheader(f"Simulated Kinetic Curves for: {city_name}")
                    sns.set_theme(style="whitegrid")
                    fig, ax1 = plt.subplots(figsize=(10, 5))

                    line1 = ax1.plot(data['Date'], data['Anomaly'], color='#e74c3c', linewidth=2, label='Heat-Stress Index Anomaly (°C-equiv.)')
                    ax1.set_ylabel('Heat-Stress Anomaly (°C-equiv.)', color='#e74c3c')
                    ax1.tick_params(axis='x', rotation=45)

                    ax2 = ax1.twinx()
                    line2 = ax2.plot(data['Date'], data['BBB_Leakage'], color='#2980b9', linewidth=2.5, linestyle='--', label='Simulated BBB Permeability')
                    line3 = ax2.plot(data['Date'], data['Microglia_M1'], color='#2c3e50', linewidth=3, label='Simulated Microglial M1 Fraction')

                    ax2.fill_between(data['Date'], data['BBB_Low'], data['BBB_High'], color='#2980b9', alpha=0.15)
                    ax2.fill_between(data['Date'], data['M1_Low'], data['M1_High'], color='#2c3e50', alpha=0.15)
                    ax2.set_ylabel('Simulated Activation (0-1, unitless model output)', color='#2c3e50')

                    lns = line1 + line2 + line3
                    labs = [l.get_label() for l in lns]
                    ax1.legend(lns, labs, loc='upper center', bbox_to_anchor=(0.5, -0.25), ncol=3, frameon=True)

                    fig.tight_layout()
                    st.pyplot(fig)

                    st.markdown("""
                    ##### Chart Legend
                    * <span style='color:#e74c3c; font-weight:bold;'>Red line (left axis):</span> Heat-stress index anomaly, computed from real historical reanalysis temperature and humidity data.
                    * <span style='color:#2980b9; font-weight:bold;'>Blue dashed line (right axis):</span> Simulated BBB permeability (illustrative model output, unitless 0–1 scale).
                    * <span style='color:#2c3e50; font-weight:bold;'>Black line (right axis):</span> Simulated microglial M1 activation fraction (illustrative model output, unitless 0–1 scale).
                    * <span style='color:gray; font-weight:bold;'>Shaded bands:</span> ±20% one-parameter sensitivity sweep on the model's gain constants — **not** a genetic, demographic, or statistical confidence interval.
                    """, unsafe_allow_html=True)

                with c2:
                    st.subheader("Simulation Data Table")
                    st.dataframe(data[['Date', 'Anomaly', 'BBB_Leakage', 'Microglia_M1']].style.format(precision=3))

                # -----------------------------------------------------------
                # SUMMARY (renamed from "Certified Lab Report" and de-clinicalized)
                # -----------------------------------------------------------
                st.markdown("---")
                st.subheader("Simulation Summary (Non-Clinical)")

                max_stress = data['Anomaly'].max()
                max_bbb = data['BBB_Leakage'].max()
                max_m1 = data['Microglia_M1'].max()

                if max_bbb > 0.75:
                    bbb_interpretation = "Model predicts high simulated BBB permeability under this scenario."
                else:
                    bbb_interpretation = "Model predicts moderate-to-low simulated BBB permeability under this scenario."

                if max_m1 > 0.60:
                    m1_interpretation = "Model predicts a high simulated microglial M1 activation fraction under this scenario."
                else:
                    m1_interpretation = "Model predicts a moderate-to-low simulated microglial M1 activation fraction under this scenario."

                raw_download_text = f"""SIMULATION SUMMARY -- EDUCATIONAL / NON-CLINICAL
============================================================
This is output from an illustrative mechanistic simulation. It is NOT a
medical report, diagnosis, or clinically validated risk assessment.

Location: {city_name} (Lat: {lat}, Lon: {lon})
Scenario profile: {cohort_profile}
Hemodynamic scenario: {blood_pressure_state}
------------------------------------------------------------

Model outputs over the selected date range:
* Peak heat-stress index anomaly: {max_stress:.2f} (°C-equivalent, computed from real reanalysis weather data)
* Peak simulated BBB permeability: {(max_bbb*100):.1f}% of the model's 0-100% scale
* Peak simulated microglial M1 activation: {(max_m1*100):.1f}% of the model's 0-100% scale

Model interpretation (illustrative, not diagnostic):
* {bbb_interpretation}
* {m1_interpretation}

Notes:
* Rate constants are illustrative estimates chosen for plausible dynamics.
  They are not fitted to patient data or extracted from a specific numeric
  source in the cited literature; cited papers support the qualitative
  direction of the modeled effect only.
* Shaded uncertainty bands represent a +/-20% one-parameter sensitivity sweep,
  not a statistical or genetic confidence interval.
* This output must not be used to assess real individual or population health
  risk.
"""

                st.markdown(f"""
                * **Location:** {city_name} (Lat: {lat}, Lon: {lon})
                * **Scenario profile:** {cohort_profile}
                * **Hemodynamic scenario:** {blood_pressure_state}

                **Model outputs over the selected date range**
                * Peak heat-stress index anomaly: {max_stress:.2f} °C-equivalent (from real reanalysis weather data)
                * Peak simulated BBB permeability: {(max_bbb*100):.1f}% of model scale
                * Peak simulated microglial M1 activation: {(max_m1*100):.1f}% of model scale

                **Model interpretation (illustrative, not diagnostic)**
                * {bbb_interpretation}
                * {m1_interpretation}

                ---
                *Rate constants are illustrative, not fitted to patient data. Shaded bands are a ±20% sensitivity sweep, not a confidence interval. Not for real health risk assessment.*
                """)

                st.download_button(
                    label="Download Simulation Summary (.txt)",
                    data=raw_download_text,
                    file_name=f"simulation_summary_{city_name.replace(' ', '_')}.txt",
                    mime="text/plain"
                )
            else:
                st.error("Could not retrieve weather data. Check your coordinates and connection, then try again.")

# ---------------------------------------------------------------------------
# FRONTEND: TWO-LOCATION COMPARISON VIEW
# ---------------------------------------------------------------------------
else:
    st.header("Two-Location Comparison")
    col_a, col_b = st.columns(2)
    with col_a:
        st.subheader("Location A")
        lat_a = st.number_input("Latitude A", value=36.8065, format="%.4f")
        lon_a = st.number_input("Longitude A", value=10.1815, format="%.4f")
    with col_b:
        st.subheader("Location B")
        lat_b = st.number_input("Latitude B", value=50.8503, format="%.4f")
        lon_b = st.number_input("Longitude B", value=4.3517, format="%.4f")

    city_a_name = get_location_name(lat_a, lon_a)
    city_b_name = get_location_name(lat_b, lon_b)

    if st.button("Run Comparison"):
        with st.spinner("Fetching reanalysis weather data for both locations..."):
            df_a = fetch_and_model(lat_a, lon_a, start_date, end_date, bbb_gain, m1_gain, shear_stress_multiplier)
            df_b = fetch_and_model(lat_b, lon_b, start_date, end_date, bbb_gain, m1_gain, shear_stress_multiplier)

            if df_a is not None and df_b is not None:
                st.subheader(f"Comparison under scenario: {cohort_profile}")
                fig_comp, (ax_bbb, ax_m1) = plt.subplots(1, 2, figsize=(14, 5))

                ax_bbb.plot(df_a['Date'], df_a['BBB_Leakage'], label=f"{city_a_name}", color="#e67e22", linewidth=2.5)
                ax_bbb.plot(df_b['Date'], df_b['BBB_Leakage'], label=f"{city_b_name}", color="#9b59b6", linewidth=2.5, linestyle="--")
                ax_bbb.set_title("Simulated BBB Permeability")
                ax_bbb.set_ylabel("Model output (0-1 scale)")
                ax_bbb.legend()
                ax_bbb.tick_params(axis='x', rotation=45)

                ax_m1.plot(df_a['Date'], df_a['Microglia_M1'], label=f"{city_a_name}", color="#e67e22", linewidth=2.5)
                ax_m1.plot(df_b['Date'], df_b['Microglia_M1'], label=f"{city_b_name}", color="#9b59b6", linewidth=2.5, linestyle="--")
                ax_m1.set_title("Simulated Microglial M1 Activation")
                ax_m1.set_ylabel("Model output (0-1 scale)")
                ax_m1.legend()
                ax_m1.tick_params(axis='x', rotation=45)

                fig_comp.tight_layout()
                st.pyplot(fig_comp)

                st.markdown("""
                ##### Comparison Legend
                * **Solid orange line:** Location A.
                * **Dashed purple line:** Location B.
                * Both locations are simulated under the **same** scenario profile and hemodynamic setting, so any difference shown is driven only by the two locations' real weather data — not by any assumed demographic or genetic difference between them.
                """)
            else:
                st.error("Could not retrieve weather data for one or both locations. Check coordinates and connection.")
