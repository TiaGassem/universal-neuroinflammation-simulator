import streamlit as st
import requests
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.integrate import odeint

# Set page layout configuration
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
# REVERSE GEOCODING UTILITY
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
        "Chronic metabolic/vascular risk scenario ( diabetes/hypertension pattern)",
        "Compounded high-risk scenario (older age + metabolic/vascular risk)",
    ]
)

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

    df['Heat_Stress_Index'] = df['Max_Temp'] + (0.55 * (df['Max_Humidity'] / 100) * (df['Max_Temp'] - 14.5))
    df['Anomaly'] = np.clip(df['Heat_Stress_Index'] - 25, 0, None)
    days_timeline = np.arange(len(df))

    def internal_ode(y, t, anomaly_data, bg, mg, sm):
        idx = int(np.clip(t, 0, len(anomaly_data) - 1))
        dt_anomaly = anomaly_data[idx]
        BBB_perm, M1_activation = y[0], y[1]

        k_bbb_recovery = 0.28 if dt_anomaly == 0 else 0.05
        k_m1_recovery = 0.23 if dt_anomaly == 0 else 0.08

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
    st.header(" Single Location Deep-Dive")
    col_input1, col_input2 = st.columns(2)
    with col_input1:
        lat = st.number_input("Latitude", value=36.8065, format="%.4f")
    with col_input2:
        lon = st.number_input("Longitude", value=10.1815, format="%.4f")

    city_name = get_location_name(lat, lon)

    if st.button("Run Simulation", type="primary"):
        with st.spinner("Fetching data and running integration engine..."):
            data = fetch_and_model(lat, lon, start_date, end_date, bbb_gain, m1_gain, shear_stress_multiplier)
            if data is not None:
                plt.close('all')
                
                c1, c2 = st.columns([2, 1])
                with c1:
                    st.subheader(f"📈 Simulated Kinetic Projections: {city_name}")
                    sns.set_theme(style="whitegrid")
                    
                    fig, ax1 = plt.subplots(figsize=(11, 5.5))

                    line1 = ax1.plot(data['Date'], data['Anomaly'], color='#e74c3c', linewidth=2, label='Heat-Stress Index Anomaly (°C-equiv.)')
                    ax1.set_ylabel('Heat-Stress Anomaly (°C-equiv.)', color='#e74c3c', fontweight='bold')
                    ax1.tick_params(axis='x', rotation=45)

                    ax2 = ax1.twinx()
                    line2 = ax2.plot(data['Date'], data['BBB_Leakage'], color='#2980b9', linewidth=2.5, linestyle='--', label='Simulated BBB Permeability')
                    line3 = ax2.plot(data['Date'], data['Microglia_M1'], color='#2c3e50', linewidth=3, label='Simulated Microglial M1 Fraction')

                    ax2.fill_between(data['Date'], data['BBB_Low'], data['BBB_High'], color='#2980b9', alpha=0.12)
                    ax2.fill_between(data['Date'], data['M1_Low'], data['M1_High'], color='#2c3e50', alpha=0.12)
                    ax2.set_ylabel('Simulated Activation Scale (0.0 - 1.0)', color='#2c3e50', fontweight='bold')
                    ax2.set_ylim(-0.05, 1.05)

                    # FIXED: Correct string location type and layout mapping logic
                    lns = line1 + line2 + line3
                    labs = [l.get_label() for l in lns]
                    ax1.legend(lns, labs, loc='upper center', bbox_to_anchor=(0.5, -0.25), ncol=3, frameon=True)

                    fig.tight_layout()
                    st.pyplot(fig)
                    plt.close(fig)

                with c2:
                    st.subheader(" Automated Matrix Logs")
                    st.dataframe(
                        data[['Date', 'Anomaly', 'BBB_Leakage', 'Microglia_M1']].style.format({
                            'Anomaly': '{:.2f} °C',
                            'BBB_Leakage': '{:.3f}',
                            'Microglia_M1': '{:.3f}'
                        }), 
                        use_container_width=True,
                        height=400
                    )

                # -----------------------------------------------------------
                # TRANSLATIONAL BIOMEDICAL PREDICTIVE REPORT
                # -----------------------------------------------------------
                st.write("")
                st.markdown("---")
                st.markdown("##  TRANSLATIONAL BIOMEDICAL PREDICTIVE REPORT")
                
                # Metadata Grid Row
                m_col1, m_col2, m_col3 = st.columns(3)
                m_col1.metric("Target Location Context", f"{city_name} ({lat}, {lon})")
                m_col2.metric("Physiological Cohort Baseline", cohort_profile)
                m_col3.metric("Vascular Strain Model", blood_pressure_state)

                max_stress = data['Anomaly'].max()
                max_bbb = data['BBB_Leakage'].max()
                max_m1 = data['Microglia_M1'].max()

                # Operational Checkpoint Threshold Flags
                if max_bbb > 0.75:
                    bbb_status, bbb_level = "CRITICAL ENDOTHELIAL INTEGRITY LOSS", "error"
                else:
                    bbb_status, bbb_level = "MODERATE PERMEABILITY SHIFT DETECTED", "warning"

                if max_m1 > 0.60:
                    m1_status, m1_level = "ACCELERATED PHENOTYPIC TRANSGRESSION METRIC", "error"
                else:
                    m1_status, m1_level = "REGULATED HOMEOSYNAPTIC TRAJECTORY", "info"

                # Structural Split columns for structured clarity
                rep_left, rep_right = st.columns(2)
                
                with rep_left:
                    st.markdown("###  Simulated Quantitative Endpoints")
                    st.info(f"**Peak Atmospheric Displacement Factor:** {max_stress:.2f} °C over target threshold baselines.")
                    
                    if bbb_level == "error":
                        st.error(f"**Max Predicted Endothelial Disruption Profile:** {(max_bbb*100):.1f}% functional clearance variance.")
                    else:
                        st.warning(f"**Max Predicted Endothelial Disruption Profile:** {(max_bbb*100):.1f}% functional clearance variance.")
                        
                    if m1_level == "error":
                        st.error(f"**Peak Microglial M1 Pro-Inflammatory Transgression:** {(max_m1*100):.1f}% functional polarization.")
                    else:
                        st.info(f"**Peak Microglial M1 Pro-Inflammatory Transgression:** {(max_m1*100):.1f}% functional polarization.")

                with rep_right:
                    st.markdown("### Automated Pathokinetic Interpretation")
                    st.markdown(f"**Endothelial Barrier State:** `{bbb_status}`")
                    st.caption("The simulation engine maps accelerated junction disassembly rates mirroring structural vessel leak profiles during high sustained atmospheric workloads.")
                    
                    st.markdown(f"**Neuroimmune Signaling Response:** `{m1_status}`")
                    st.caption("Downstream pathway trajectories point to interactive cascading reactions heavily dependent on microvascular integrity breakdown values.")
                    
                    st.markdown("**Compounded Environmental Risk Analysis:**")
                    st.caption(f"An absolute environmental workload anomaly of {max_stress:.2f}°C serves as a functional kinetic catalyst when combined with your selected risk parameters.")

                st.write("")
                st.caption(" *Verification Disclosure: Core kinetic rates are illustrative estimates scaled qualitatively from parameters outlined in Montagne et al. and Perry & Holmes guidelines.*")

                # Reconstruct Plain Text Download Data Block
                raw_download_text = f"""TRANSLATIONAL BIOMEDICAL PREDICTIVE REPORT -- SIMULATION OUTPUT
================================================================================
Target Domain Location: {city_name} (Lat: {lat}, Lon: {lon})
Patient Stratification Profile: {cohort_profile}
Hemodynamic Loading Factor: {blood_pressure_state}
--------------------------------------------------------------------------------

SIMULATED QUANTITATIVE ENDPOINT RESULTS:
* Peak Atmospheric Heat-Stress Displacement Metric: {max_stress:.2f} °C above baseline parameters.
* Maximum Predicted Endothelial Disruption Index (BBB Leakage): {(max_bbb*100):.1f}% functional breakdown variance.
* Peak Simulated Microglial M1 Phenotypic Transgression Matrix: {(max_m1*100):.1f}% state cellular activation.

AUTOMATED PATHOKINETIC INTERPRETATION:
* Endothelial Barrier Integrity Status: {bbb_status}
* Neuroimmune Activation Pathway Response: {m1_status}
* Compounded Environmental Risk Analysis: Environmental heat workload of {max_stress:.2f}°C acts as an operational accelerator.

DISCLAIMER: Non-clinical, non-predictive portfolio simulation. Rate constants are illustrative estimates and are not fitted to real clinical patient matrices.
"""
                st.markdown("---")
                st.markdown("###  Export Structural Report Artifact")
                st.download_button(
                    label=" Download Academic Report Data (.txt)",
                    data=raw_download_text,
                    file_name=f"Biomedical_Predictive_Report_{city_name.replace(' ', '_')}.txt",
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

    if st.button("Run Comparison Analysis", type="primary"):
        with st.spinner("Fetching comparison metrics from API..."):
            df_a = fetch_and_model(lat_a, lon_a, start_date, end_date, bbb_gain, m1_gain, shear_stress_multiplier)
            df_b = fetch_and_model(lat_b, lon_b, start_date, end_date, bbb_gain, m1_gain, shear_stress_multiplier)

            if df_a is not None and df_b is not None:
                plt.close('all')
                st.subheader(f"Differential Comparison Pathokinetics: {cohort_profile}")
                fig_comp, (ax_bbb, ax_m1) = plt.subplots(1, 2, figsize=(14, 5.5))

                ax_bbb.plot(df_a['Date'], df_a['BBB_Leakage'], label=f"{city_a_name}", color="#e67e22", linewidth=2.5)
                ax_bbb.plot(df_b['Date'], df_b['BBB_Leakage'], label=f"{city_b_name}", color="#9b59b6", linewidth=2.5, linestyle="--")
                ax_bbb.set_title("Simulated BBB Permeability Matrix", fontweight='bold')
                ax_bbb.set_ylabel("Model Output Scale (0.0-1.0)")
                ax_bbb.legend(loc='lower left')
                ax_bbb.tick_params(axis='x', rotation=45)

                ax_m1.plot(df_a['Date'], df_a['Microglia_M1'], label=f"{city_a_name}", color="#e67e22", linewidth=2.5)
                ax_m1.plot(df_b['Date'], df_b['Microglia_M1'], label=f"{city_b_name}", color="#9b59b6", linewidth=2.5, linestyle="--")
                ax_m1.set_title("Simulated Microglial M1 Activation Profile", fontweight='bold')
                ax_m1.set_ylabel("Model Output Scale (0.0-1.0)")
                ax_m1.legend(loc='lower left')
                ax_m1.tick_params(axis='x', rotation=45)

                fig_comp.tight_layout()
                st.pyplot(fig_comp)
                plt.close(fig_comp)
            else:
                st.error("Could not retrieve comparison datasets. Please verify coordinates.")
