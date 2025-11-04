"""
predict_example.py
==================

Purpose
-------
Minimal example showing how to use `solaris_pointing.offsets.fitting.model_1d`
to load previously fitted models and compute the azimuth-only offsets.
It then applies the offsets to "ideal" telescope coordinates.

Requirements
------------
- The model must have been generated beforehand (see scripts/model_cli.py)
  and saved as Joblib files:
  * models/mymodel.joblib
- All angles in this example are in **degrees**.

What this example does
----------------------
1) Loads the azimuth and elevation models from Joblib file.
2) Computes (offset_az_deg, offset_el_deg) for a given azimuth.
3) Applies the offsets to ideal (az, el) to obtain corrected coordinates.

Usage
-----
Run the example:

    python examples/predict_example.py

Adapt the variables `ideal_az_deg` and `ideal_el_deg` as needed.
"""

from solaris_pointing.offsets.fitting.model_1d import (
    load_model,
    predict_offsets_deg,
)

# 1) Load pre-fitted model (produced by your fitting pipeline or CLI).
model = load_model("models/mymodel.joblib")

# 2) Choose the *ideal* pointing coordinates (what the scheduler would command).
#    NOTE: Offsets depend on azimuth only. Units are degrees.
ideal_az_deg = 80.0000
ideal_el_deg = 40.0000

# Compute the offsets (in degrees) predicted at this azimuth.
off_az_deg, off_el_deg = predict_offsets_deg(
    model, az_deg=ideal_az_deg
)

# 3) Apply offsets to obtain corrected (commanded) coordinates.
#    Sign convention: "corrected = ideal + offset"
corr_az_deg = ideal_az_deg + off_az_deg
corr_el_deg = ideal_el_deg + off_el_deg

# Print a concise summary for operators/integrators.
print(
    "Input (ideal) az, el:",
    f"{ideal_az_deg:.4f} deg",
    f"{ideal_el_deg:.4f} deg", 
)
print("Predicted offsets [deg]:   ", f"{off_az_deg:.4f}", f"{off_el_deg:.4f}")
print("Predicted offsets [arcmin]:   ", f"{off_az_deg*60:.4f}", f"{off_el_deg*60:.4f}")
print("Corrected (cmd) az, el:", f"{corr_az_deg:.4f}", f"{corr_el_deg:.4f}")

# Tip: Refit models every N days so that the "azimuth-only" approximation
# stays valid for your observing window.
