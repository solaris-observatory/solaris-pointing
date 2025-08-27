"""
az_model
========

Library to fit telescope pointing offset models where **both offsets depend
only on azimuth**. All quantities (azimuth, elevation, offsets) are handled
in **degrees**.

Rationale (why only azimuth)
----------------------------
The model must output the offset in azimuth and the offset in elevation,
each depending only on azimuth; that's because we observe only the Sun.
In fact, the elevation at a given azimuth changes only slowly from one day
to the next; over short time windows this lets us approximate both offsets
as functions of azimuth alone. To keep the approximation valid, the model
should be updated every N days so that, for any given azimuth, the
corresponding elevation does not drift too far from the value used when the
offset was originally computed.

Overview
--------
This module provides functions to:
- Read a TSV dataset of measurements (comment lines start with '#').
- Fit two independent polynomial models in the standard basis:
  offset_az(az) = P_az(az)        # degrees vs degrees
  offset_el(az) = P_el(az)        # degrees vs degrees
- Save / load models to/from disk.
- Predict offsets for a given azimuth (degrees).

Notes
-----
- Outlier rejection is based on a simple z-score on the *target* variable.
- Expected units:
  * azimuth [deg], elevation [deg] (elevation kept for QA; not used for fit)
  * offset_az [deg], offset_el [deg]
- If your input offsets are in arcseconds, pass input_offset_unit="arcsec" to
  the TSV reader and they will be converted to degrees (arcsec / 3600).

Example
-------
>>> from solaris_pointing.models.az_model import (
...     read_offsets_tsv, fit_models, save_models, load_models,
...     predict_offsets_deg,
... )
>>> df = read_offsets_tsv("offsets.tsv", input_offset_unit="deg")
>>> models = fit_models(df["azimuth"], df["offset_az"], df["offset_el"],
...                     degree=3, zscore=3.0)
>>> save_models(models, "az_model.joblib", "el_model.joblib")
>>> az_model, el_model = load_models("az_model.joblib", "el_model.joblib")
>>> predict_offsets_deg(az_model, el_model, azimuth_deg=125.0)
(0.0123, -0.0044)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Tuple

import joblib
import numpy as np
import pandas as pd
from numpy.polynomial import Polynomial

REQUIRED_COLUMNS = ["azimuth", "elevation", "offset_az", "offset_el"]


@dataclass
class FitResult:
    """Container for a single fitted polynomial and diagnostics."""

    model: Polynomial
    r2: float
    n_kept: int
    n_input: int


@dataclass
class AzOnlyModels:
    """Both fitted models (offset_az(az), offset_el(az)) and diagnostics."""

    az_model: Polynomial
    el_model: Polynomial
    az_r2: float
    el_r2: float
    n_kept_az: int
    n_kept_el: int
    n_input: int

    def as_tuple(self) -> Tuple[Polynomial, Polynomial]:
        return self.az_model, self.el_model


def _to_numpy_deg(values: Iterable[float]) -> np.ndarray:
    return np.asarray(values, dtype=float)


def _remove_outliers_z(
    x: np.ndarray, y: np.ndarray, z: float = 3.0
) -> Tuple[np.ndarray, np.ndarray]:
    """Return filtered (x, y) where |zscore(y)| < z; robust to zero/NaN std."""
    y = np.asarray(y, dtype=float)
    x = np.asarray(x, dtype=float)
    std = float(np.nanstd(y))
    if std == 0 or np.isnan(std):
        mask = np.ones_like(y, dtype=bool)
    else:
        zscores = np.abs((y - np.nanmean(y)) / std)
        mask = zscores < z
    return x[mask], y[mask]


def _fit_poly_standard(
    x: np.ndarray, y: np.ndarray, degree: int, zscore: float
) -> FitResult:
    """Outlier-reject on y then fit Polynomial; compute R² on filtered data."""
    x_clean, y_clean = _remove_outliers_z(x, y, z=zscore)
    n_input = len(x)
    n_kept = len(x_clean)
    if n_kept < max(2, degree + 1):
        raise ValueError(
            "Not enough points after outlier rejection: "
            f"{n_kept} (need >= {max(2, degree + 1)})"
        )
    model = Polynomial.fit(x_clean, y_clean, deg=degree).convert()
    y_pred = model(x_clean)
    ss_res = float(np.sum((y_clean - y_pred) ** 2))
    ss_tot = float(np.sum((y_clean - np.mean(y_clean)) ** 2))
    # Treat near-constant targets as perfect fits: R² = 1.0 when variance
    # is zero or numerically negligible.
    eps = max(1e-20, 1e-12 * (abs(float(np.mean(np.abs(y_clean)))) + 1.0))
    r2 = 1.0 if ss_tot <= eps else (1.0 - ss_res / ss_tot)
    return FitResult(model=model, r2=r2, n_kept=n_kept, n_input=n_input)


# ---------- Public API ----------


def read_offsets_tsv(path: str, input_offset_unit: str = "deg") -> pd.DataFrame:
    """
    Read a TSV (tab-separated) file with comment lines starting with '#'. Ensure
    the required columns exist and are numeric.

    Parameters
    ----------
    path : str
        Path to the TSV file.
    input_offset_unit : {"deg", "arcsec"}, default "deg"
        If "arcsec", offset_az and offset_el are converted to degrees
        (value / 3600).

    Returns
    -------
    pandas.DataFrame
        Data with required columns as float (degrees for all angles/offsets).
    """
    # IMPORTANT: use a single tab character as delimiter
    df = pd.read_csv(path, sep="\t", comment="#", engine="python")
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    # Cast to numeric and drop NaNs
    for c in REQUIRED_COLUMNS:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=REQUIRED_COLUMNS).reset_index(drop=True)

    unit = input_offset_unit.lower()
    if unit == "arcsec":
        df["offset_az"] = df["offset_az"] / 3600.0
        df["offset_el"] = df["offset_el"] / 3600.0
    elif unit != "deg":
        raise ValueError('input_offset_unit must be "deg" or "arcsec"')

    return df.copy()


def fit_models(
    azimuth_deg: Iterable[float],
    offset_az_deg: Iterable[float],
    offset_el_deg: Iterable[float],
    degree: int = 3,
    zscore: float = 3.0,
) -> AzOnlyModels:
    """
    Fit two polynomial models in the standard basis:
        offset_az(az) = P_az(az), offset_el(az) = P_el(az)

    Parameters
    ----------
    azimuth_deg, offset_az_deg, offset_el_deg : array-like of float
        Arrays in **degrees**.
    degree : int, default 3
        Use the smallest degree that captures the trend without overfitting.
    zscore : float, default 3.0
        |z| threshold for outlier rejection on the target variable.

    Returns
    -------
    AzOnlyModels
        Both models and diagnostics (R² and point counts).
    """
    az = _to_numpy_deg(azimuth_deg)
    off_az = _to_numpy_deg(offset_az_deg)
    off_el = _to_numpy_deg(offset_el_deg)

    az_res = _fit_poly_standard(az, off_az, degree=degree, zscore=zscore)
    el_res = _fit_poly_standard(az, off_el, degree=degree, zscore=zscore)

    return AzOnlyModels(
        az_model=az_res.model,
        el_model=el_res.model,
        az_r2=az_res.r2,
        el_r2=el_res.r2,
        n_kept_az=az_res.n_kept,
        n_kept_el=el_res.n_kept,
        n_input=max(az_res.n_input, el_res.n_input),
    )


def fit_models_from_tsv(
    tsv_path: str,
    degree: int = 3,
    zscore: float = 3.0,
    input_offset_unit: str = "deg",
) -> AzOnlyModels:
    """
    Read data from TSV and fit models with the same semantics as `fit_models`.

    Parameters
    ----------
    tsv_path : str
        Path to the TSV file (tab-separated, '#' comments).
    degree : int, default 3
        Polynomial degree.
    zscore : float, default 3.0
        Outlier rejection threshold on |z| of target.
    input_offset_unit : {"deg", "arcsec"}, default "deg"
        If the file stores offsets in arcseconds, pass "arcsec".

    Returns
    -------
    AzOnlyModels
    """
    df = read_offsets_tsv(tsv_path, input_offset_unit=input_offset_unit)
    return fit_models(
        df["azimuth"], df["offset_az"], df["offset_el"], degree=degree, zscore=zscore
    )


def predict_offsets_deg(
    az_model: Polynomial, el_model: Polynomial, azimuth_deg: float
) -> Tuple[float, float]:
    """
    Predict (offset_az_deg, offset_el_deg) at the given azimuth in degrees.

    Returns
    -------
    (float, float)
        Offsets in **degrees**.
    """
    az = float(azimuth_deg)
    return float(az_model(az)), float(el_model(az))


def save_models(models: AzOnlyModels, az_model_path: str, el_model_path: str) -> None:
    """Save both Polynomial models to disk with joblib."""
    joblib.dump(models.az_model, az_model_path)
    joblib.dump(models.el_model, el_model_path)


def load_models(
    az_model_path: str, el_model_path: str
) -> Tuple[Polynomial, Polynomial]:
    """Load both Polynomial models from disk with joblib."""
    az_model = joblib.load(az_model_path)
    el_model = joblib.load(el_model_path)
    if not isinstance(az_model, Polynomial) or not isinstance(el_model, Polynomial):
        raise TypeError("Loaded objects are not numpy.polynomial.Polynomial instances.")
    return az_model, el_model


def model_summary(models: AzOnlyModels) -> str:
    """Return a summary with polynomial forms and R² values."""
    lines = []
    lines.append(
        "Units: degrees for azimuth, elevation (not used for fit), and both offsets."
    )
    lines.append("Model 1: offset_az = P_az(azimuth)")
    lines.append(str(models.az_model))
    lines.append(
        f"R^2 = {models.az_r2:.6f} (kept {models.n_kept_az}/{models.n_input} points)"
    )
    lines.append("")
    lines.append("Model 2: offset_el = P_el(azimuth)")
    lines.append(str(models.el_model))
    lines.append(
        f"R^2 = {models.el_r2:.6f} (kept {models.n_kept_el}/{models.n_input} points)"
    )
    return "\n".join(lines)
