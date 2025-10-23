"""Azimuth-only pointing-offset modeling with a linearized azimuth.

This module fits two 1D models to telescope pointing offsets measured while tracking
the Sun:
  - offset_az(az_lin): azimuth correction vs linearized azimuth
  - offset_el(az_lin): elevation correction vs linearized azimuth

The models are linear in parameters and may be pure polynomials or extended
linear-basis models with optional Fourier terms.

Units & conventions
-------------------
- Input offsets: expected in degrees by default, unless specified via
  `input_offset_unit` ('deg', 'arcmin', or 'arcsec'); internally, all offsets
  are converted to degrees.
- Internal math: offsets are handled in degrees; trigonometric arguments use
  radians where required.
- Angles shown in outputs: degrees unless noted.
- Azimuth wrapping: azimuths are kept modulo 360° and unwrapped around the
  "cut" angle to build a continuous predictor az_lin in degrees, suitable for
  polynomial regression.

Linearized azimuth and cut angle
--------------------------------
Raw azimuth lives on a circle (0..360°) and exhibits a jump at 0/360°. Solar
campaigns often cover only one continuous arc (e.g. 250..360 and 0..70°).
We select a "cut" angle and unwrap around the largest data gap so that azimuth
becomes a single continuous interval; this prevents spurious edges in the
regression design. The linearized azimuth az_lin is then used as the predictor.

We store a "cut" azimuth (deg) to linearize azimuth values around the largest
gap in sampling. Given a raw azimuth az_deg ∈ [0, 360), we compute az_lin by:
  1) shifting az_deg by -cut,
  2) mapping to (-180, 180] (wrap-around),
  3) adding cut back so that the interval is contiguous across the gap.

Robust two-pass fitting
-----------------------
We adopt a simple and effective two-pass scheme.

Pass 1 (coarse fit):
  - Fit a linear-basis model to all points via least squares. The basis includes
    the chosen polynomial degree d and any enabled optional terms (Fourier
    1/rev harmonics, custom-period sin/cos, sector steps).
  - Apply ridge regularization (Tikhonov) by default with alpha=0.1
    (configurable). This stabilizes the solution when the design matrix has
    near-collinear columns or the sampling is uneven.

Residuals and robust scale:
  - Residual r = y - y_hat, where y is the measured offset and y_hat is the
    model prediction from the coarse fit.

Robust scale via MAD:
  - Compute the Median Absolute Deviation (MAD) of residuals and convert it to
    a robust standard-deviation estimate: sigma ≈ 1.4826 * MAD. This estimator
    is resilient to outliers by construction.

Inlier mask by robust z-score:
  - A point is an inlier if |r| / sigma ≤ z_thr, where z_thr is the threshold
    (default 2.5). Larger z_thr keeps more points; smaller z_thr is stricter.
    If sigma == 0 (all residuals identical), we keep all points.

Pass 2 (refit on inliers only):
  - Refit the same basis (same polynomial degree and enabled optional terms) on
    inliers. This final model is less affected by outliers than a single-pass
    fit.

We perform the two-pass procedure independently for offset_az and offset_el,
because their noise properties and outliers can differ.

Model stabilization and tuning
------------------------------
Regularization (ridge) adds a small diagonal term alpha * I to XᵀX before
solving the normal equations. This shrinks coefficients toward zero and
mitigates noise amplification when columns in X are nearly collinear or when
high-degree polynomials are used on sparse or unevenly sampled azimuths.
By default alpha=0.1; set alpha=0.0 to disable ridge.

Choosing degree and thresholds:
  - Robust threshold z_thr (default 2.5):
    If you see too many points classified as outlier (e.g. poor weather
    segments), increase z_thr; to be stricter, decrease it.
  - Polynomial degree d:
    Start with d=2 or d=3; if residuals show smooth curvature not captured by
    the current polynomial, consider increasing d. Prefer adding physics-based
    basis terms (Fourier/custom periods/sectors) before pushing d too high.

Optional linear-basis extensions
--------------------------------
The linear model can be extended with additional basis functions that capture
mechanical or structural periodicities.

1) Fourier harmonics
   - What: add sin(k*A) and cos(k*A) for k = 1..K, with A the wrapped azimuth
     in degrees converted internally to radians (A = deg2rad(A_deg)).
   - Why: capture periodic effects (tilt projection, encoder eccentricity,
     mechanical symmetries).
   - CLI flag: --fourier-k <int>
   - Amplitudes and phases are reported only for these 1/rev harmonics, because
     arbitrary-period terms (below) do not have a unique phase reference.

2) Custom periods (gear/encoder lines)
   - What: add sin/cos at user-specified periods P (degrees), not constrained
     to k cycles per 360°. Each P contributes cos(2π * A_deg / P) and
     sin(2π * A_deg / P), where A_deg is in degrees.
   - Why: capture tones from encoders, gear ratios, or residual mechanical
     periodicities not aligned to 1/rev harmonics.
   - How to choose --periods-deg:
     * Physics-first:
       · Encoder with N lines  -> P = 360 / N.
       · Worm gear ratio r:s    -> expect r/s rev tones; convert to degrees.
       · Support symmetry (m)   -> P = 360 / m.
     * Data-first (residual periodogram):
       · Fit poly + k/rev Fourier, compute residuals.
       · Compute periodogram over A ∈ [0, 360).
       · Pick dominant peaks not matching 360/k harmonics.
     * Typical starting set: one or two strong residual peaks or mechanical
       frequencies.

3) Sector / step features (cable-wrap states)
   - What: add piecewise-constant offsets by sector on wrapped azimuth. Provide
     sector edges (degrees), e.g. --sector-edges-deg 60,210 creates sectors
     [60, 210) and [210, 60) around the circle.
   - Why: model discontinuous changes when cable-wrap flips or harness tension
     changes; smooth bases alone would oscillate around those steps.
   - Keep the number of sectors small (2–3). Too many sectors fracture the fit.
   - Diagnostics: residuals near edges should flatten with no ringing.

Prediction range and soft margin
--------------------------------
The model is trained on a finite az_lin interval determined by the data. We
define a soft margin of ±5° beyond that interval for cautious use.
- If the requested azimuth is within ±5° of the training range, evaluation is
  allowed but flagged.
- If beyond and `allow_extrapolation=False`, an exception is raised.

Outside the soft margin, predictions should be treated with caution since the
polynomial is unconstrained there.

Metadata & diagnostics
----------------------
We store by-model and by-file diagnostics:
- counts of total points, kept in az fit, kept in el fit
- goodness-of-fit on inliers: R², RMSE (deg), MAE (deg)
- source file path and SHA256 hash (comments stripped) for traceability
- UTC timestamp and optional free-form notes
- library_version string (e.g. "az_model 1.0.0") for provenance

These fields let you judge model quality and reproducibility at a glance.

Usage and CLI
-------------
API:
- fit_models_from_tsv(
      path: str,
      degree: int = 3,
      zscore_az: float = 2.5,
      zscore_el: float = 2.5,
      ridge_alpha: float = 0.1,
      notes: str | None = None,
      fourier_k: int = 0,
      periods_deg: list[float] | None = None,
      sector_edges_deg: list[float] | None = None,
      input_offset_unit: str = "deg"
  ) -> ModelBundle:
    Read TSV, unwrap with an auto-detected cut, robust two-pass fit with MAD,
    ridge regularization, then return bundle (models + metadata + diagnostics).
    Offsets are converted to degrees internally.

- save_model_bundle(bundle: ModelBundle, path: str) -> None:
    Save bundle to a single joblib file.

- load_model_bundle(path: str) -> ModelBundle:
    Load a previously saved bundle.

- predict_offsets_deg(
      bundle: ModelBundle,
      az_deg: float,
      allow_extrapolation: bool = False
  ) -> tuple[float, float]:
    Normalize az_deg modulo 360°, unwrap, apply the soft ±5° margin, then
    evaluate az/el models in degrees. If outside the margin and
    allow_extrapolation=False, raise.

CLI (Command Line Interface):
  - Fit models from a TSV and save a .joblib bundle.
  - Print a human-readable summary (Fourier amplitudes/phases are shown only if
    --fourier-k > 0; custom-period terms are listed but not expanded into
    amplitude/phase pairs).
  - Predict at a given azimuth with range checks and soft margin.
  - Plot data vs fit using the linearized azimuth.

Future extensions
-----------------
- Periodic local bases (von Mises) or periodic cubic splines.
- Direction/backlash terms via a sweep-sign indicator.
- Model selection aids (AIC/BIC) and angular block cross-validation.
"""

from dataclasses import dataclass, asdict
from typing import Any, Tuple, Optional
import datetime
import hashlib
import numpy as np
import pandas as pd
from numpy.polynomial import Polynomial
import joblib


# ----- Angle helpers -----
def _wrap_deg(x):
    x = np.asarray(x, dtype=float)
    return np.mod(x, 360.0)

def _sector_dummy_columns(A_deg: np.ndarray, edges_deg: list) -> np.ndarray:
    """Return sector dummy matrix with (S-1) columns to avoid collinearity.
    Sectors are defined on wrapped azimuth A ∈ [0, 360) by sorted edges.
    Example: edges [60, 210] -> sectors [60,210) and [210,60).
    The last sector is dropped.
    """
    if not edges_deg:
        return np.zeros((A_deg.size, 0), float)
    edges = np.sort(np.mod(edges_deg, 360.0))
    # Build sector indices
    S = edges.size + 1
    # Define sector boundaries including 0 and 360
    bounds = np.concatenate(([0.0], edges, [360.0]))
    # For each angle, find sector index
    A = _wrap_deg(A_deg)
    idx = np.searchsorted(bounds, A, side='right') - 1
    idx[idx == S] = 0  # wrap-around safeguard
    # Build one-hot and drop last column
    M = np.zeros((A.size, S), float)
    M[np.arange(A.size), idx] = 1.0
    return M[:, :max(S-1, 0)]


# Linear-in-parameters model: polynomial + optional Fourier
class LinearBasisModel:
    """Callable linear-basis model over linearized azimuth (degrees).

    Columns layout:
      [1, x, x^2, ..., x^p, cos(1A), sin(1A), ..., cos(KA), sin(KA),
       cos(2πA/P1), sin(2πA/P1), ..., sector dummies ...]
    where x=az_lin (deg), A=wrap(x) in radians. Custom periods P are in degrees.
    """
    def __init__(
        self,
        coef,
        poly_degree: int,
        fourier_k: int = 0,
        periods_deg: Optional[list] = None,
        sector_edges_deg: Optional[list] = None,
    ) -> None:
        self.coef = np.asarray(coef, dtype=float)
        self.poly_degree = int(poly_degree)
        self.fourier_k = int(fourier_k)
        # State must be stored for prediction to match the fit basis.
        self.periods_deg = list(periods_deg) if periods_deg else None
        self.sector_edges_deg = list(sector_edges_deg) if sector_edges_deg else None

    def __call__(self, az_lin) -> np.ndarray:
        # Use the *full* design to match how the coefficients were estimated.
        x = np.asarray(az_lin, dtype=float)
        X = _design_matrix_full(
            x,
            degree=self.poly_degree,
            fourier_k=self.fourier_k,
            periods_deg=self.periods_deg,
            sector_edges_deg=self.sector_edges_deg,
        )
        return X @ self.coef

    def describe(self) -> dict:
        return {
            "type": "linear_basis",
            "poly_degree": self.poly_degree,
            "fourier_k": self.fourier_k,
            "periods_deg": self.periods_deg,
            "sector_edges_deg": self.sector_edges_deg,
        }

# -------------------------
# I/O utilities
# -------------------------

def _sha256_of_file_strip_comments(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for raw in f:
            try:
                line = raw.decode("utf-8", "ignore")
            except Exception:
                line = ""
            if line.lstrip().startswith("#"):
                continue
            h.update(line.encode("utf-8"))
    return h.hexdigest()


# --- Unit conversion helpers -------------------------------------------------

def _offset_unit_to_deg_factor(unit: str) -> float:
    """
    Return the conversion factor from the given angular unit to degrees.

    Parameters
    ----------
    unit : str
        Input angular unit. Accepted values:
        - "deg", "degree", "degrees"
        - "arcmin", "arcminute", "arcminutes"
        - "arcsec", "arcsecond", "arcseconds"

    Returns
    -------
    float
        Multiplicative factor to convert the value to degrees.
        For example:
        - deg → 1.0
        - arcmin → 1/60
        - arcsec → 1/3600
    """
    if unit is None:
        return 1.0
    u = unit.strip().lower()
    if u in ("deg", "degree", "degrees"):
        return 1.0
    if u in ("arcmin", "arcminute", "arcminutes"):
        return 1.0 / 60.0
    if u in ("arcsec", "arcsecond", "arcseconds"):
        return 1.0 / 3600.0
    raise ValueError(
        f"Unsupported input_offset_unit='{unit}'. "
        "Use 'deg', 'arcmin', or 'arcsec'."
    )



def read_offsets_tsv(
    path: str,
    *,
    input_offset_unit: str = "deg",
) -> "pd.DataFrame":
    """
    Read a delimited text file containing telescope offsets.

    The file must include at least:
      - azimuth (in degrees)
      - offset_az, offset_el (in `input_offset_unit`)

    Notes
    -----
    - The delimiter is auto-detected (tab, comma, semicolon, or whitespace).
    - Lines starting with '#' are treated as comments and ignored.
    - All offsets are converted to degrees for downstream processing.
    """
    import pandas as pd

    # Try strict TSV first for speed. If it fails, fall back to auto-detect.
    try:
        df = pd.read_csv(path, sep="\t", comment="#")
    except Exception:
        # Auto-detect delimiter (Python engine supports 'sep=None').
        # This handles commas/semicolons/spaces and mixed whitespace.
        df = pd.read_csv(path, sep=None, engine="python", comment="#")

    # Normalize column names (strip spaces)
    df.columns = [c.strip() for c in df.columns]

    required = {"azimuth", "offset_az", "offset_el"}
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(
            f"Missing required columns {missing} in '{path}'. "
            f"Found columns: {list(df.columns)}"
        )

    # Unit conversion → degrees
    fac = _offset_unit_to_deg_factor(input_offset_unit)
    if fac != 1.0:
        df["offset_az"] = df["offset_az"].astype(float) * fac
        df["offset_el"] = df["offset_el"].astype(float) * fac

    return df


# -------------------------
# Angle unwrapping on the circle
# -------------------------

def _circular_gaps_deg(az_deg: np.ndarray) -> Tuple[int, float, float]:
    az_sorted = np.sort(az_deg % 360.0)
    n = az_sorted.size
    if n < 2:
        return 0, float(az_sorted[0]), float(az_sorted[0])
    diffs = np.empty(n)
    diffs[:-1] = np.diff(az_sorted)
    diffs[-1] = (az_sorted[0] + 360.0) - az_sorted[-1]
    idx = int(np.argmax(diffs))
    start = float(az_sorted[idx])
    end = float(az_sorted[(idx + 1) % n])
    return idx, start, end


def unwrap_azimuth(az_deg: np.ndarray) -> Tuple[np.ndarray, float, float, float]:
    az = np.asarray(az_deg, dtype=float) % 360.0
    _, gstart, gend = _circular_gaps_deg(az)
    gap = (gend - gstart) % 360.0
    cut = (gstart + gap / 2.0) % 360.0
    az_lin = az.copy()
    mask = az_lin < cut
    az_lin[mask] = az_lin[mask] + 360.0
    return az_lin, float(cut), float(np.min(az_lin)), float(np.max(az_lin))

# -------------------------
# Robust utilities
# -------------------------

def _mad(x: np.ndarray) -> float:
    med = np.median(x)
    mad = np.median(np.abs(x - med))
    return 1.4826 * mad if mad > 0 else 0.0


def _robust_mask_from_residuals(res: np.ndarray, zthr: float) -> np.ndarray:
    sigma = _mad(res)
    if sigma <= 0:
        return np.ones_like(res, dtype=bool)
    z = np.abs(res) / sigma
    return z <= zthr

# -------------------------
# Design matrices and ridge solvers
# -------------------------

def _design_matrix_poly(x: np.ndarray, degree: int) -> np.ndarray:
    return np.vander(x, N=degree + 1, increasing=True)

def _design_matrix_linear(x_deg: np.ndarray, degree: int, fourier_k: int) -> np.ndarray:
    Xp = _design_matrix_poly(x_deg, degree)
    K = int(fourier_k)
    if K <= 0:
        return Xp
    A = np.deg2rad(_wrap_deg(x_deg))
    cols = [Xp]
    for k in range(1, K + 1):
        cols.append(np.cos(k * A)[:, None])
        cols.append(np.sin(k * A)[:, None])
    return np.concatenate(cols, axis=1)


def _design_matrix_full(
    x_deg: np.ndarray,
    degree: int,
    fourier_k: int,
    periods_deg: Optional[list],
    sector_edges_deg: Optional[list],
) -> np.ndarray:
    """
    Build the full linear design matrix for azimuth-linearized models.

    Columns (in order)
    ------------------
    1) Polynomial basis in az_lin (deg): [1, x, x^2, ..., x^degree]
    2) Optional 1/rev Fourier harmonics on wrapped azimuth A (deg):
       [cos(kA), sin(kA)] for k = 1..fourier_k, where A = wrap(x_deg) in [0,360)
    3) Optional custom periods in degrees:
       For each P in periods_deg: [cos(2πA/P), sin(2πA/P)]
    4) Optional sector dummy variables
       Piecewise-constant indicators for sectors defined by sector_edges_deg.
       We drop the last sector to avoid collinearity with the intercept.

    Returns
    -------
    np.ndarray
        Design matrix with shape (N, n_features).
    """
    # Base: polynomial + optional 1/rev Fourier
    X = _design_matrix_linear(x_deg, degree, fourier_k)

    cols = [X]
    A_deg = _wrap_deg(x_deg)

    # Custom periods
    if periods_deg:
        for P in periods_deg:
            if P is None:
                continue
            P = float(P)
            if P <= 0.0:
                continue
            w = 2.0 * np.pi * (A_deg / P)
            cols.append(np.cos(w)[:, None])
            cols.append(np.sin(w)[:, None])

    # Sector dummy columns
    if sector_edges_deg:
        D = _sector_dummy_columns(A_deg, list(sector_edges_deg))
        if D.size:
            cols.append(D)

    return np.concatenate(cols, axis=1)


def _fit_linear_ridge(
    x: np.ndarray,
    y: np.ndarray,
    degree: int,
    fourier_k: int,
    alpha: float,
    periods_deg: Optional[list] = None,
    sector_edges_deg: Optional[list] = None,
):
    """Solve ridge for poly + (optional) Fourier + (optional) terms.
    Returns Polynomial only if there are no non-polynomial bases enabled.
    """
    X = _design_matrix_full(
        x, degree, fourier_k,
        periods_deg=periods_deg,
        sector_edges_deg=sector_edges_deg,
    )
    XT_X = X.T @ X
    if alpha > 0:
        XT_X = XT_X + float(alpha) * np.eye(XT_X.shape[0])
    coef = np.linalg.solve(XT_X, X.T @ y)

    # Back-compat: return Polynomial ONLY when basis == pure polynomial
    if (fourier_k == 0) and not periods_deg and not sector_edges_deg:
        return Polynomial(coef)

    # Otherwise return LinearBasisModel with full Fourier
    return LinearBasisModel(
        coef=coef,
        poly_degree=degree,
        fourier_k=fourier_k,
        periods_deg=periods_deg,
        sector_edges_deg=sector_edges_deg,
    )

# -------------------------
# Two-pass fit and bundle
# -------------------------

@dataclass
class FitDiagnostics:
    n_input: int
    n_kept_az: int
    n_kept_el: int
    r2_az: float
    r2_el: float
    rmse_az_deg: float
    rmse_el_deg: float
    mae_az_deg: float
    mae_el_deg: float

@dataclass
class ModelMetadata:
    degree: int
    zscore_az: float
    zscore_el: float
    ridge_alpha: float
    cut_deg: float
    az_lin_min_deg: float
    az_lin_max_deg: float
    timestamp_utc: str
    data_hash: str
    source_path: str
    library_version: str
    notes: Optional[str] = None
    fourier_k: int = 0
    periods_deg: Optional[list] = None
    sector_edges_deg: Optional[list] = None
    input_offset_unit: str = "deg"


@dataclass
class ModelBundle:
    az_model: Any
    el_model: Any
    meta: ModelMetadata
    diag: FitDiagnostics

def _two_pass_fit(
    x: np.ndarray,
    y: np.ndarray,
    degree: int,
    zthr: float,
    ridge_alpha: float,
    fourier_k: int = 0,
    periods_deg: Optional[list] = None,
    sector_edges_deg: Optional[list] = None,
) -> Tuple[Any, np.ndarray, float, float, float]:
    m0 = _fit_linear_ridge(x, y, degree=degree, fourier_k=fourier_k,
                           alpha=ridge_alpha,
                           periods_deg=periods_deg,
                           sector_edges_deg=sector_edges_deg)
    res0 = y - m0(x)
    mask = _robust_mask_from_residuals(res0, zthr)
    if np.sum(mask) < max(3, degree + 1):
        mask = np.ones_like(mask, dtype=bool)
    m = _fit_linear_ridge(x[mask], y[mask], degree=degree,
                          fourier_k=fourier_k, alpha=ridge_alpha,
                          periods_deg=periods_deg,
                          sector_edges_deg=sector_edges_deg)
    yhat = m(x[mask])
    res = y[mask] - yhat
    ss_res = float(np.dot(res, res))
    y_mean = float(np.mean(y[mask]))
    dy = y[mask] - y_mean
    ss_tot = float(np.dot(dy, dy))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 1.0
    rmse = float(np.sqrt(np.mean(res * res)))
    mae = float(np.mean(np.abs(res)))
    return m, mask, r2, rmse, mae

# -------------------------
# Public API
# -------------------------

def fit_models_from_tsv(
    path: str,
    degree: int = 3,
    zscore_az: float = 2.5,
    zscore_el: float = 2.5,
    ridge_alpha: float = 0.1,
    notes: Optional[str] = None,
    fourier_k: int = 0,
    periods_deg: Optional[list] = None,
    sector_edges_deg: Optional[list] = None,
    input_offset_unit: str = "deg",
) -> ModelBundle:
    """
    Fit azimuth/elevation offset models from a single TSV file.

    Parameters
    ----------
    path : str
        Path to the TSV file containing azimuth, offset_az, offset_el columns.
    degree : int, optional
        Polynomial degree of the base model.
    zscore_az, zscore_el : float, optional
        Robust z-score thresholds for azimuth/elevation fits.
    ridge_alpha : float, optional
        Ridge regularization factor (Tikhonov).
    notes : str, optional
        Free-form metadata notes.
    fourier_k : int, optional
        Number of 1/rev Fourier harmonics
    periods_deg : list[float], optional
        Custom periods in degrees
    sector_edges_deg : list[float], optional
        Sector boundaries in degrees
    input_offset_unit : str, optional
        Unit of input offsets ("deg", "arcmin", or "arcsec"). Defaults to "deg".

    Returns
    -------
    ModelBundle
        Bundle containing azimuth/elevation models, metadata, and diagnostics.
    """

    # Read and convert input offsets to degrees
    df = read_offsets_tsv(path, input_offset_unit=input_offset_unit)

    az = df["azimuth"].to_numpy(float) % 360.0
    off_az = df["offset_az"].to_numpy(float)
    off_el = df["offset_el"].to_numpy(float)

    # Unwrap azimuth and perform robust two-pass fits
    az_lin, cut, az_lin_min, az_lin_max = unwrap_azimuth(az)
    m_az, mask_az, r2_az, rmse_az, mae_az = _two_pass_fit(
        az_lin, off_az, degree=degree, zthr=zscore_az,
        ridge_alpha=ridge_alpha, fourier_k=fourier_k,
        periods_deg=periods_deg, sector_edges_deg=sector_edges_deg
    )
    m_el, mask_el, r2_el, rmse_el, mae_el = _two_pass_fit(
        az_lin, off_el, degree=degree, zthr=zscore_el,
        ridge_alpha=ridge_alpha, fourier_k=fourier_k,
        periods_deg=periods_deg, sector_edges_deg=sector_edges_deg
    )

    # Metadata and diagnostics
    meta = ModelMetadata(
        degree=degree,
        zscore_az=zscore_az,
        zscore_el=zscore_el,
        ridge_alpha=ridge_alpha,
        cut_deg=cut,
        az_lin_min_deg=az_lin_min,
        az_lin_max_deg=az_lin_max,
        timestamp_utc=datetime.datetime.utcnow().isoformat(timespec="seconds") + "Z",
        data_hash=_sha256_of_file_strip_comments(path),
        source_path=str(path),
        library_version="az_model 1.0.0",
        notes=notes,
        fourier_k=int(fourier_k),
        periods_deg=periods_deg,
        sector_edges_deg=sector_edges_deg,
        input_offset_unit=input_offset_unit,
    )

    diag = FitDiagnostics(
        n_input=int(len(df)),
        n_kept_az=int(np.sum(mask_az)),
        n_kept_el=int(np.sum(mask_el)),
        r2_az=r2_az,
        r2_el=r2_el,
        rmse_az_deg=rmse_az,
        rmse_el_deg=rmse_el,
        mae_az_deg=mae_az,
        mae_el_deg=mae_el,
    )

    return ModelBundle(az_model=m_az, el_model=m_el, meta=meta, diag=diag)


def _unwrap_single(az_deg: float, cut_deg: float) -> float:
    az = float(az_deg) % 360.0
    return az if az >= cut_deg else az + 360.0

def predict_offsets_deg(bundle: ModelBundle, az_deg: float,
                        allow_extrapolation: bool=False) -> Tuple[float, float]:
    az_lin = _unwrap_single(az_deg, bundle.meta.cut_deg)
    margin = 5.0
    lo = bundle.meta.az_lin_min_deg - margin
    hi = bundle.meta.az_lin_max_deg + margin
    if (az_lin < lo or az_lin > hi) and not allow_extrapolation:
        raise ValueError(
            f"Requested azimuth {az_deg:.3f}° unwraps to {az_lin:.3f}°, "
            f"which is outside the observed linear range "
            f"[{lo:.3f}, {hi:.3f}]°. "
            "Set allow_extrapolation=True if intentional."
        )

    off_az = float(bundle.az_model(np.array([az_lin]))[0])
    off_el = float(bundle.el_model(np.array([az_lin]))[0])
    return off_az, off_el


def model_summary(bundle: ModelBundle) -> str:
    def poly_to_str(p: Polynomial) -> str:
        terms = []
        for i, c in enumerate(p.coef):
            if abs(c) < 1e-15:
                continue
            terms.append(f"{c:+.6e} x^{i}")
        return " ".join(terms) if terms else "0"

    lines = []
    md = bundle.meta
    dg = bundle.diag
    lines.append("Azimuth-only pointing model (linearized azimuth)")
    lines.append(f"  degree:              {md.degree}")
    lines.append(f"  zscore_az / zscore_el:  {md.zscore_az} / {md.zscore_el}")
    lines.append(f"  ridge_alpha:         {md.ridge_alpha}")
    lines.append(f"  cut (deg):           {md.cut_deg:.6f}")
    lines.append(
        f"  az_lin range (deg):  [{md.az_lin_min_deg:.6f}, {md.az_lin_max_deg:.6f}]"
    )
    lines.append(f"  fitted at (UTC):     {md.timestamp_utc}")
    lines.append(f"  data hash:           {md.data_hash[:16]}...")
    lines.append(f"  source TSV:          {md.source_path}")
    if md.notes:
        lines.append(f"  notes:               {md.notes}")
    lines.append("Diagnostics (inliers/refit):")
    lines.append(
        f"  n_input / kept_az / kept_el: "
        f"{dg.n_input} / {dg.n_kept_az} / {dg.n_kept_el}"
    )
    lines.append(f"  R2 (az / el):        {dg.r2_az:.5f} / {dg.r2_el:.5f}")
    lines.append(
        f"  RMSE (deg) az/el:    {dg.rmse_az_deg:.6e} / {dg.rmse_el_deg:.6e}"
    )
    lines.append(
        f"  MAE  (deg) az/el:    {dg.mae_az_deg:.6e} / {dg.mae_el_deg:.6e}"
    )

    if isinstance(bundle.az_model, Polynomial) and isinstance(bundle.el_model, Polynomial):
        lines.append("Polynomials (power basis):")
        lines.append(f"  offset_az(az_lin) = {poly_to_str(bundle.az_model)}")
        lines.append(f"  offset_el(az_lin) = {poly_to_str(bundle.el_model)}")
    else:
        lines.append("Linear-basis models: poly + optional Fourier")
        if hasattr(bundle.az_model, "describe"):
            lines.append(f"  az basis: {bundle.az_model.describe()}")
        if hasattr(bundle.el_model, "describe"):
            lines.append(f"  el basis: {bundle.el_model.describe()}")

        def flines(m):
            if not isinstance(m, LinearBasisModel) or m.fourier_k <= 0:
                return []
            p = m.poly_degree + 1
            coef = np.asarray(m.coef, dtype=float)
            out = []
            for k in range(1, m.fourier_k + 1):
                a = float(coef[p + 2*(k-1) + 0])
                b = float(coef[p + 2*(k-1) + 1])
                R = float(np.hypot(a, b))
                phi = float(np.degrees(np.arctan2(-b, a)))
                out.append(f"    k={k}: amp={R:.6e} deg, phase={phi:.2f} deg")
            return out

        fa = flines(bundle.az_model)
        fe = flines(bundle.el_model)
        if fa:
            lines.append("  Fourier amplitudes/phases (az):")
            lines.extend(fa)
        if fe:
            lines.append("  Fourier amplitudes/phases (el):")
            lines.extend(fe)
        # metadata
        if getattr(md, 'periods_deg', None):
            lines.append(
                "  Custom periods (deg): " +
                ", ".join(f"{p:g}" for p in md.periods_deg)
            )
        if getattr(md, 'sector_edges_deg', None):
            lines.append(
                "  Sector edges (deg): " +
                ", ".join(f"{e:g}" for e in md.sector_edges_deg)
            )
    return "\n".join(lines)

# -------------------------
# Persistence
# -------------------------

def save_model_bundle(bundle: ModelBundle, path: str) -> None:
    """Save bundle to a .joblib file (backward-compatible).

    Format 1 (legacy): both models are pure Polynomial -> store coef arrays.
    Format 2 (extended): at least one model is LinearBasisModel or any
    non-polynomial basis -> store a small descriptor next to coef.
    """

    def pack(m):
        # Pure Polynomial (legacy)
        if isinstance(m, Polynomial):
            return {
                "type": "poly",
                "coef": np.asarray(m.coef, dtype=float).tolist(),
            }
        # LinearBasisModel with Fourier fields
        return {
            "type": "linear_basis",
            "coef": np.asarray(m.coef, dtype=float).tolist(),
            "poly_degree": int(getattr(m, "poly_degree", 0)),
            "fourier_k": int(getattr(m, "fourier_k", 0)),
            "periods_deg": list(getattr(m, "periods_deg", []) or []),
            "sector_edges_deg": list(
                getattr(m, "sector_edges_deg", []) or []
            ),
        }

    both_poly = (
        isinstance(bundle.az_model, Polynomial) and
        isinstance(bundle.el_model, Polynomial)
    )

    if both_poly:
        # Keep the historical format (format=1) for pure polynomials
        to_save = {
            "format": 1,
            "az_coef": np.asarray(bundle.az_model.coef, float).tolist(),
            "el_coef": np.asarray(bundle.el_model.coef, float).tolist(),
            "meta": asdict(bundle.meta),
            "diag": asdict(bundle.diag),
        }
    else:
        # Extended format (format=2) for linear-basis models
        to_save = {
            "format": 2,
            "az_model": pack(bundle.az_model),
            "el_model": pack(bundle.el_model),
            "meta": asdict(bundle.meta),
            "diag": asdict(bundle.diag),
        }

    joblib.dump(to_save, path)

def load_model_bundle(path: str) -> ModelBundle:
    d = joblib.load(path)
    fmt = int(d.get("format", 1))
    if fmt == 1 and "az_coef" in d:
        az_model = Polynomial(np.asarray(d["az_coef"], dtype=float))
        el_model = Polynomial(np.asarray(d["el_coef"], dtype=float))
    else:
        def unpack(m):
            if m.get("type") == "poly":
                return Polynomial(np.asarray(m["coef"], dtype=float))
            return LinearBasisModel(
                coef=np.asarray(m["coef"], dtype=float),
                poly_degree=int(m.get("poly_degree", 0)),
                fourier_k=int(m.get("fourier_k", 0)),
                periods_deg=m.get("periods_deg", None),
                sector_edges_deg=m.get("sector_edges_deg", None),
            )

        if "az_model" in d:
            az_model = unpack(d["az_model"])
        else:
            az_model = Polynomial(np.asarray(d["az_coef"], dtype=float))

        if "el_model" in d:
            el_model = unpack(d["el_model"])
        else:
            el_model = Polynomial(np.asarray(d["el_coef"], dtype=float))
    meta = ModelMetadata(**d["meta"])
    diag = FitDiagnostics(**d["diag"])
    return ModelBundle(az_model=az_model, el_model=el_model, meta=meta, diag=diag)
