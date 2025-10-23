"""
az_model_example.py
===================

Purpose
-------
Minimal example showing how to use `solaris_pointing.fitting.az_model`
to load previously fitted models and compute the azimuth-only offsets.
It then applies the offsets to "ideal" telescope coordinates.

Requirements
------------
- The models must have been generated beforehand (see scripts/az_model_cli.py)
  and saved as Joblib files:
  * output/models/az_model.joblib
  * output/models/el_model.joblib
- All angles in this example are in **degrees**.

What this example does
----------------------
1) Loads the azimuth and elevation models from Joblib files.
2) Computes (offset_az_deg, offset_el_deg) for a given azimuth.
3) Applies the offsets to ideal (az, el) to obtain corrected coordinates.

Usage
-----
Run the example:

    python examples/az_model_example.py

Adapt the variables `ideal_az_deg` and `ideal_el_deg` as needed.
"""

from solaris_pointing.fitting.az_model import (
    load_models,
    predict_offsets_deg,
)

# 1) Load pre-fitted models (produced by your fitting pipeline or CLI).
AZ_MODEL_PATH = "output/models/az_model.joblib"
EL_MODEL_PATH = "output/models/el_model.joblib"
az_model, el_model = load_models(AZ_MODEL_PATH, EL_MODEL_PATH)

# 2) Choose the *ideal* pointing coordinates (what the scheduler would command).
#    NOTE: Offsets depend on azimuth only. Units are degrees.
ideal_az_deg = 125.0000
ideal_el_deg = 40.0000

# Compute the offsets (in degrees) predicted at this azimuth.
off_az_deg, off_el_deg = predict_offsets_deg(
    az_model, el_model, azimuth_deg=ideal_az_deg
)

# 3) Apply offsets to obtain corrected (commanded) coordinates.
#    Sign convention: "corrected = ideal + offset"
corr_az_deg = ideal_az_deg + off_az_deg
corr_el_deg = ideal_el_deg + off_el_deg

# Print a concise summary for operators/integrators.
print("Input (ideal) az, el [deg]:", f"{ideal_az_deg:.6f}", f"{ideal_el_deg:.6f}")
print("Predicted offsets [deg]:   ", f"{off_az_deg:.6f}", f"{off_el_deg:.6f}")
print("Corrected (cmd) az, el [deg]:", f"{corr_az_deg:.6f}", f"{corr_el_deg:.6f}")

# Tip: Refit models every N days so that the "azimuth-only" approximation
# stays valid for your observing window.
