 Heat-Stress Neuroinflammation Simulator

Launch the live app
 https://universal-neuroinflammation-simulator-qebvnrydyxvbpxavfybq4u.streamlit.app/


What this is

An educational, hypothesis-generating simulation that combines real historical
weather data with an illustrative mechanistic model of a proposed heat-stress →
blood-brain-barrier → microglial-activation pathway. It is a student portfolio
project, not a validated scientific instrument.

This tool is NOT a diagnostic, clinical, or predictive medical instrument.
It does not use real patient data, has not been clinically validated, and
should not be used to assess real health risk for any individual or location.

Scientific overview

The underlying biological hypothesis — that heat stress can trigger peripheral
cytokine release, blood-brain barrier (BBB) tight-junction degradation, and
downstream microglial M1 polarization — is supported by a body of published
literature (see Parameter Sources below). This project
does not claim to have discovered that mechanism. What it does is
implement a small, transparent, two-variable Ordinary Differential Equation
(ODE) model of that pathway, driven by real historical weather data for a
location and date range you choose, so the resulting dynamics can be
inspected and questioned rather than taken on faith.

The app fetches daily historical temperature and humidity data from the
Open-Meteo Archive API, which is built on ECMWF ERA5 reanalysis — a
scientific dataset that blends satellite observations, ground station data,
and physical models. It computes a simplified heat-stress index from that
data and uses it to force a two-state ODE system representing simulated BBB
permeability and simulated microglial M1 activation.

Parameter sources

Not every constant in this model is literature-derived, and this table says
so explicitly rather than implying otherwise:

ParameterStatusSource / derivationBBB recovery rate (no active stressor): 0.28/dayLiterature-derivedYang et al., biphasic BBB permeability recovery over ~120h post-injury (half-life ≈2.5 days) → k = ln(2)/2.5M1 recovery rate (no active stressor): 0.23/dayLiterature-derived[18F]DPA-714 PET imaging, LPS-induced microglial activation peaking at 24h, baseline by 72h (half-life ≈3 days) → k = ln(2)/3Recovery rates under ongoing stress (0.05 / 0.08)Illustrative assumptionNo study reports a specific suppression factor for this conditionBBB/M1 production "gain" constants per scenarioIllustrative, direction-onlyNo paper reports a rate constant in this model's specific unit. Cited literature (Montagne et al.; Perry & Holmes) supports the qualitative direction of the effect, not the numeric valueHemodynamic scenario multiplierIllustrativeNot derived from a specific source

Full derivation math and in-app source table are visible in the running app
under "Model parameter sources and honesty notes."

Tech stack


Frontend: Streamlit Cloud
ODE solver: SciPy (scipy.integrate.odeint)
Weather data: Open-Meteo Archive API (ECMWF ERA5 reanalysis)
Data / plotting: Pandas, NumPy, Seaborn, Matplotlib


Limitations (read before citing or reusing)


The ODE model has not been fitted to any biomarker, patient, or clinical
outcome dataset — it produces plausible-looking dynamics, not validated
predictions.
The model operates at daily time resolution, while much of the underlying
cytokine/BBB kinetics literature describes hour-scale dynamics. This is a
real resolution mismatch, not a hidden assumption.
"Patient scenario" presets in the app are illustrative parameter sets, not
real patient data or a real clinical cohort.


Project status

This project was previously named "Universal In-Silico Pathokinetic
Platform." The name and framing were revised for accuracy — the model is a
proof-of-concept for one hypothesized pathway, not a universal or validated
clinical platform.

Author

Tasnim — independent student computational project, not part of an official
lab or affiliated institution unless stated otherwise.

License

This project is licensed under the GNU Affero General Public License v3.0
(AGPL-3.0).

AGPL-3.0 was chosen deliberately over more permissive options (MIT, Apache,
BSD) because this project is deployed as a hosted web app, not a downloaded
binary. Regular GPLv3's copyleft only triggers on distribution of the code;
AGPL-3.0 closes that gap by also requiring source disclosure when a modified
version is run as a network service that users interact with remotely — the
deployment shape this project actually uses. In short: anyone is free to use,
study, and build on this code, including commercially, but if they modify it
and offer it as a hosted service, their modified source must also be made
available under the same terms.
