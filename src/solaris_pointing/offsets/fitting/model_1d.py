"""One-dimensional (azimuth-only predictor) pointing-offset modeling for AZ/EL.

This module fits *two* 1D models of telescope pointing offsets measured while
tracking the Sun, using the **linearized azimuth** (``az_lin``) as predictor:

  - ``offset_az(az_lin)``: azimuth correction vs. linearized azimuth
  - ``offset_el(az_lin)``: elevation correction vs. linearized azimuth

Models are linear in parameters. They can be:
  1) **Polynomial** in ``az_lin`` up to a chosen degree.
  2) **Polynomial + Fourier** terms at k/rev harmonics (k = 1..K).
  3) **Custom-period** sinusoidal terms with user-provided periods (deg).
  4) **Sector dummies** (piecewise constants) on wrapped azimuth sectors.

All fits are **robust** via a two-pass procedure with MAD-based z-score
thresholding. A small **ridge** (L2) regularization stabilizes the solution.

----------------------------------------------------------------------------
Units and conventions
----------------------------------------------------------------------------
- Input offsets are read from TSV and converted to **degrees** internally.
  The reader accepts ``input_offset_unit`` in ``{"deg","arcmin","arcsec"}``.
- Angles in trigonometric terms use **radians** internally, but periods for
  custom sinusoids are specified in **degrees**.
- The azimuth predictor is **linearized** by unwrapping the circle at the
  midpoint of the **largest angular gap** present in the data. See
  ``unwrap_azimuth``.

----------------------------------------------------------------------------
Design matrix (for linear-basis models)
----------------------------------------------------------------------------
Let ``x = az_lin (deg)``, ``A = wrap(x) (rad)`` and custom periods ``P_i (deg)``.
The column layout is:

  [ 1, x, x^2, ..., x^p,  cos(1·A), sin(1·A), ..., cos(K·A), sin(K·A),
    cos(2π·x/P1), sin(2π·x/P1), ..., sector dummies ]

- Polynomial degree ``p = degree`` (p ≥ 0).
- k/rev block: harmonics k = 1..K (K = ``fourier_k``).
- Custom periods: one cos/sin pair per each ``P_i`` in ``periods_deg``.
- Sector dummies: one-hot encoding of sectors given by ``sector_edges_deg``,
  with the **last sector dropped** to avoid collinearity with the intercept.

----------------------------------------------------------------------------
Robust two-pass fit
----------------------------------------------------------------------------
1) Fit a provisional model on all points.
2) Compute residuals and a robust scale ``S = 1.4826 * median(|r − median(r)|)``.
3) Keep inliers where ``|r| ≤ zscore * S``. If too few inliers remain, fall
   back to all points.
4) Refit on inliers only. Diagnostics (R², RMSE, MAE, kept counts) are saved.

The **same** ``zscore`` applies to both AZ and EL fits (unified parameter).

----------------------------------------------------------------------------
Prediction range and soft margin
----------------------------------------------------------------------------
The observed linearized range is stored as ``[az_lin_min_deg, az_lin_max_deg]``.
A **±5° soft margin** is used at prediction time:

- If the requested azimuth unwraps within the observed range ±5°, evaluation
  is allowed (the caller may still choose to disallow it).
- If outside and ``allow_extrapolation=False``, a ``ValueError`` is raised.

----------------------------------------------------------------------------
Saved bundle format (joblib)
----------------------------------------------------------------------------
``save_model_bundle`` writes a dict with:

- ``format``: 2 (new) or 1 (legacy).
- ``az_model`` / ``el_model``:
    - ``{"type":"poly", "coef":[...]}`` for Polynomial
    - ``{"type":"linear_basis", "coef":[...], "poly_degree", "fourier_k",
       "periods_deg", "sector_edges_deg"}`` for linear-basis models
- ``meta``: ``ModelMetadata`` with fields:
    - ``degree``, ``zscore``, ``ridge_alpha``, ``notes``
    - ``fourier_k``, ``periods_deg``, ``sector_edges_deg``
    - ``az_lin_min_deg``, ``az_lin_max_deg``, ``timestamp_utc``,
      ``data_hash``, ``source_path``, ``library_version``
- ``diag``: ``FitDiagnostics`` with counts and metrics (per axis)
- Optionally, arrays ``az_lin``, ``offset_az``, ``offset_el`` (for inspection)

``load_model_bundle`` accepts both format 2 and legacy format 1.

----------------------------------------------------------------------------
Public API (used by the CLI)
----------------------------------------------------------------------------
- ``read_offsets_tsv(path, *, input_offset_unit="deg") -> pd.DataFrame``:
    Read offsets (required cols: ``azimuth``, ``offset_az``, ``offset_el``),
    auto-detect delimiter, ignore ``#`` comments. Convert offsets to degrees.

- ``unwrap_azimuth(az_deg) -> (az_lin, cut_deg, az_lin_min, az_lin_max)``:
    Linearize azimuth modulo 360°, cutting at the midpoint of the largest gap.

- ``fit_models_from_tsv(path, degree=3, zscore=2.5, ridge_alpha=0.1,
                        notes=None, fourier_k=0, periods_deg=None,
                        sector_edges_deg=None, input_offset_unit="deg")
                        -> ModelBundle``:
    Unified parameters are applied to **both** axes.

- ``predict_offsets_deg(bundle, az_deg, allow_extrapolation=False)
                        -> (off_az_deg, off_el_deg)``:
    Unwrap the query azimuth using the stored cut; apply soft-margin policy.

- ``model_summary(bundle) -> str`` /
  ``model_summary_axis(bundle, axis) -> str``:
    Human-readable summaries (per-axis variant prints only the requested axis),
    including the explicit analytic equation for polynomial / linear-basis.

- ``save_model_bundle(bundle, path)`` / ``load_model_bundle(path) -> ModelBundle``

- ``_mad(x) -> float``: robust scale (1.4826 * MAD).

----------------------------------------------------------------------------
CLI usage (see ``model_cli.py``)
----------------------------------------------------------------------------
- **Fit** (both axes by default; per-axis selection via ``--az`` / ``--el``):
    ``python scripts/model_cli.py fit data.tsv --degree 3 --zscore 2.5
       --fourier-k 2 --periods-deg 90,45 --sector-edges-deg 60,210``

- **Predict** (auto-select backend from bundle metadata):
    ``python scripts/model_cli.py predict models/mystem --azimuth 12.0 --unit arcsec``

- **Summary**:
    ``python scripts/model_cli.py summary models/mystem_az.joblib``

----------------------------------------------------------------------------
Notes and guidance
----------------------------------------------------------------------------
- Start with ``degree=2 or 3``. Add ``fourier_k`` only if residuals show
  clear periodic structure aligned with rotation; otherwise consider one or
  two entries in ``periods_deg`` from a residual periodogram.
- Use a small ``ridge_alpha`` (e.g. 0.01–0.1) to stabilize fits with correlated
  bases. Set to 0.0 to disable ridge.
- Keep ``sector_edges_deg`` short (2–3 sectors). Too many sectors fragment
  the model and inflate variance.
"""

from dataclasses import dataclass, asdict, field
from typing import Any, Tuple, List, Optional

import joblib
import datetime
import hashlib
import numpy as np
import pandas as pd
from numpy.polynomial import Polynomial


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
    idx = np.searchsorted(bounds, A, side="right") - 1
    idx[idx == S] = 0  # wrap-around safeguard
    # Build one-hot and drop last column
    M = np.zeros((A.size, S), float)
    M[np.arange(A.size), idx] = 1.0
    return M[:, : max(S - 1, 0)]


# Linear-in-parameters model: polynomial + optional Fourier
class LinearBasisModel:
    """Callable linear-basis model over linearized azimuth (degrees).

    Columns layout:
      [1, x, x^2, ..., x^p, cos(1A), sin(1A), ..., cos(KA), sin(KA),
       cos(2πA/P1), sin(2πA/P1), ..., sector dummies ...]
    where x = az_lin (deg), A = wrap(x) in radians. Custom periods P are in degrees.
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
        # State needed at predict-time to rebuild the same design:
        self.periods_deg = list(periods_deg) if periods_deg else []
        self.sector_edges_deg = list(sector_edges_deg) if sector_edges_deg else []

    def __call__(self, az_lin) -> np.ndarray:
        x = np.asarray(az_lin, dtype=float)
        X = _design_matrix_full(
            x_deg=x,
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
        f"Unsupported input_offset_unit='{unit}'. Use 'deg', 'arcmin', or 'arcsec'."
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
    return mad if mad > 0 else 0.0


def _robust_mask_from_residuals(res: np.ndarray, zthr: float) -> np.ndarray:
    sigma = 1.4826 * _mad(res)
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
        x,
        degree,
        fourier_k,
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
    # core fit settings
    degree: int
    zscore: float
    ridge_alpha: float
    cut_deg: float

    # units & notes
    input_offset_unit: str = "arcmin"
    notes: str = ""

    # unified basis controls
    fourier_k: int = 0
    periods_deg: List[float] = field(default_factory=list)
    sector_edges_deg: List[float] = field(default_factory=list)

    # dataset provenance / range (NEW)
    az_lin_min_deg: float = float("nan")
    az_lin_max_deg: float = float("nan")
    timestamp_utc: str = ""  # e.g. "2025-10-30T14:55:02Z"
    data_hash: str = ""  # optional fingerprint of input
    source_path: str = ""  # original TSV path(s)
    library_version: str = ""  # package/version string


@dataclass
class ModelBundle:
    az_model: Any
    el_model: Any
    meta: ModelMetadata
    diag: FitDiagnostics
    az_lin: Optional[np.ndarray] = field(default=None, repr=False)
    offset_az: Optional[np.ndarray] = field(default=None, repr=False)
    offset_el: Optional[np.ndarray] = field(default=None, repr=False)


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
    m0 = _fit_linear_ridge(
        x,
        y,
        degree=degree,
        fourier_k=fourier_k,
        alpha=ridge_alpha,
        periods_deg=periods_deg,
        sector_edges_deg=sector_edges_deg,
    )
    res0 = y - m0(x)
    mask = _robust_mask_from_residuals(res0, zthr)
    if np.sum(mask) < max(3, degree + 1):
        mask = np.ones_like(mask, dtype=bool)
    m = _fit_linear_ridge(
        x[mask],
        y[mask],
        degree=degree,
        fourier_k=fourier_k,
        alpha=ridge_alpha,
        periods_deg=periods_deg,
        sector_edges_deg=sector_edges_deg,
    )
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
    zscore: float = 2.5,
    ridge_alpha: float = 0.1,
    notes: Optional[str] = None,
    fourier_k: int = 0,
    periods_deg: Optional[list] = None,
    sector_edges_deg: Optional[list] = None,
    input_offset_unit: str = "deg",
    **kwargs,
) -> ModelBundle:
    """
    Fit azimuth/elevation offset models from a single TSV file.

    - Azimuth model uses (degree, fourier_k, periods_deg, sector_edges_deg)
    - Elevation model uses (degree, fourier_k, periods_deg, sector_edges_deg)
    """
    # Read + unit-convert offsets to degrees
    df = read_offsets_tsv(path, input_offset_unit=input_offset_unit)

    az = df["azimuth"].to_numpy(float) % 360.0
    off_az = df["offset_az"].to_numpy(float)
    off_el = df["offset_el"].to_numpy(float)

    # Unwrap azimuth
    az_lin, cut, az_lin_min, az_lin_max = unwrap_azimuth(az)

    # Two independent robust fits
    m_az, mask_az, r2_az, rmse_az, mae_az = _two_pass_fit(
        az_lin,
        off_az,
        degree=degree,
        zthr=zscore,
        ridge_alpha=ridge_alpha,
        fourier_k=fourier_k,
        periods_deg=periods_deg,
        sector_edges_deg=sector_edges_deg,
    )
    m_el, mask_el, r2_el, rmse_el, mae_el = _two_pass_fit(
        az_lin,
        off_el,
        degree=degree,
        zthr=zscore,
        ridge_alpha=ridge_alpha,
        fourier_k=fourier_k,
        periods_deg=periods_deg,
        sector_edges_deg=sector_edges_deg,
    )

    meta = ModelMetadata(
        degree=degree,
        zscore=zscore,
        ridge_alpha=ridge_alpha,
        cut_deg=cut,
        input_offset_unit=input_offset_unit,
        notes=notes or "",
        fourier_k=int(fourier_k),
        periods_deg=list(periods_deg or []),
        sector_edges_deg=list(sector_edges_deg or []),
        az_lin_min_deg=az_lin_min,
        az_lin_max_deg=az_lin_max,
        timestamp_utc=datetime.datetime.utcnow().isoformat(timespec="seconds") + "Z",
        data_hash=_sha256_of_file_strip_comments(path),
        source_path=str(path),
        library_version="az_model 1.0.0",
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

    return ModelBundle(
        az_model=m_az,
        el_model=m_el,
        meta=meta,
        diag=diag,
        az_lin=az_lin,
        offset_az=off_az,
        offset_el=off_el,
    )


def _unwrap_single(az_deg: float, cut_deg: float) -> float:
    az = float(az_deg) % 360.0
    return az if az >= cut_deg else az + 360.0


def predict_offsets_deg(
    bundle: ModelBundle, az_deg: float, allow_extrapolation: bool = False
) -> Tuple[float, float]:
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


def model_summary_axis(bundle: ModelBundle, axis: str) -> str:
    """
    Build a human-readable summary for a single axis ('az' or 'el').

    The summary includes:
      - metadata (fit parameters, data range, timestamps)
      - diagnostics for the requested axis only
      - an explicit analytic equation of the fitted curve
        * Polynomial case: y = c0 + c1·x + c2·x^2 + ...
        * Linear-basis (poly + Fourier): polynomial + sum_k (A_k cos(...) + B_k sin(...))
    """
    axis = axis.lower().strip()
    if axis not in ("az", "el"):
        raise ValueError("axis must be 'az' or 'el'")

    def _poly_to_str(p: Polynomial) -> str:
        """Return a polynomial as an explicit equation in x=az_lin (degrees)."""
        terms = []
        for i, c in enumerate(p.coef):
            if abs(c) < 1e-15:
                continue
            if i == 0:
                terms.append(f"{c:+.6e}")
            elif i == 1:
                terms.append(f"{c:+.6e}·x")
            else:
                terms.append(f"{c:+.6e}·x^{i}")
        eq = " ".join(terms) if terms else "0"
        return f"y(x) = {eq}"

    def _fourier_equation(m) -> str:
        """
        Return an explicit analytic equation for a LinearBasisModel:
          y(x) = Σ_{i=0..d} a_i·x^i
               + Σ_{k=1..K} [A_k cos(2π·k·x / P_k) + B_k sin(2π·k·x / P_k)]
        Assumes:
          - m.coef contains [poly terms..., A1, B1, A2, B2, ...]
          - m.poly_degree, m.fourier_k, m.periods_deg are defined
        """
        # polynomial part
        coef = np.asarray(m.coef, dtype=float)
        d = int(m.poly_degree)
        terms = []
        for i in range(d + 1):
            a = float(coef[i])
            if abs(a) < 1e-15:
                continue
            if i == 0:
                terms.append(f"{a:+.6e}")
            elif i == 1:
                terms.append(f"{a:+.6e}·x")
            else:
                terms.append(f"{a:+.6e}·x^{i}")

        # Fourier part
        kmax = int(getattr(m, "fourier_k", 0))
        periods = getattr(m, "periods_deg", None)
        if periods is None or len(periods) == 0:
            # default to 360° if not provided per-harmonic
            periods = [360.0] * kmax

        base = d + 1
        for k in range(1, kmax + 1):
            Ak = float(coef[base + 2 * (k - 1) + 0])
            Bk = float(coef[base + 2 * (k - 1) + 1])
            if abs(Ak) < 1e-15 and abs(Bk) < 1e-15:
                continue
            Pk = float(periods[k - 1] if k - 1 < len(periods) else periods[-1])
            # Use degrees for x; 2π·k·x / Pk (x in degrees)
            terms.append(f"{Ak:+.6e}·cos(2π·{k}·x/{Pk:g})")
            terms.append(f"{Bk:+.6e}·sin(2π·{k}·x/{Pk:g})")

        rhs = " ".join(terms) if terms else "0"
        return f"y(x) = {rhs}"

    def _equation_str(m) -> str:
        """Return an explicit equation string for model m (Polynomial or LinearBasisModel)."""
        if isinstance(m, Polynomial):
            return _poly_to_str(m)
        # Assume LinearBasisModel-like
        if hasattr(m, "coef") and hasattr(m, "poly_degree"):
            return _fourier_equation(m)
        return "y(x) = <unavailable>"

    md = bundle.meta
    dg = bundle.diag

    # Select the model and diagnostics for the requested axis only
    if axis == "az":
        model = bundle.az_model
        n_kept = getattr(dg, "n_kept_az", None)
        r2 = getattr(dg, "r2_az", float("nan"))
        rmse = getattr(dg, "rmse_az_deg", float("nan"))
        mae = getattr(dg, "mae_az_deg", float("nan"))
        axis_label = "AZ"
    else:
        model = bundle.el_model
        n_kept = getattr(dg, "n_kept_el", None)
        r2 = getattr(dg, "r2_el", float("nan"))
        rmse = getattr(dg, "rmse_el_deg", float("nan"))
        mae = getattr(dg, "mae_el_deg", float("nan"))
        axis_label = "EL"

    if model is None:
        return f"No model available for axis '{axis}'.\n"

    lines = []
    # --- Metadata (unified) ---
    lines.append(f"Pointing model summary ({axis_label} only, linearized azimuth)")
    lines.append(f"  degree:              {md.degree}")
    lines.append(f"  zscore:              {md.zscore}")
    lines.append(f"  ridge_alpha:         {md.ridge_alpha}")
    lines.append(f"  fourier_k:           {md.fourier_k}")
    if getattr(md, "periods_deg", None):
        lines.append(
            "  periods_deg:         " + ", ".join(f"{p:g}" for p in md.periods_deg)
        )
    if getattr(md, "sector_edges_deg", None):
        lines.append(
            "  sector_edges_deg:    " + ", ".join(f"{e:g}" for e in md.sector_edges_deg)
        )
    lines.append(f"  cut (deg):           {md.cut_deg:.6f}")
    lines.append(
        f"  az_lin range (deg):  [{md.az_lin_min_deg:.6f}, {md.az_lin_max_deg:.6f}]"
    )
    lines.append(f"  fitted at (UTC):     {md.timestamp_utc}")
    lines.append(f"  data hash:           {md.data_hash[:16]}...")
    lines.append(f"  source TSV:          {md.source_path}")
    if md.notes:
        lines.append(f"  notes:               {md.notes}")

    # --- Diagnostics (axis-only) ---
    lines.append("Diagnostics:")
    if n_kept is not None and hasattr(dg, "n_input"):
        lines.append(f"  n_input / kept_{axis}: {dg.n_input} / {n_kept}")
    lines.append(f"  R2:                  {r2:.5f}")
    lines.append(f"  RMSE (deg):          {rmse:.6e}")
    lines.append(f"  MAE  (deg):          {mae:.6e}")

    # --- Explicit analytic equation ---
    lines.append("Fit equation:")
    lines.append(f"  {axis_label}: {_equation_str(model)}")

    return "\n".join(lines) + "\n"


def model_summary(bundle: ModelBundle) -> str:
    """
    Backward-compatible wrapper that emits both axes if available.

    NOTE:
      - The CLI is expected to use `model_summary_axis(bundle, 'az'/'el')`
        when writing per-axis summaries, to avoid mixing axes.
    """
    parts = []
    if bundle.az_model is not None:
        parts.append(model_summary_axis(bundle, "az").rstrip())
    if bundle.el_model is not None:
        parts.append(model_summary_axis(bundle, "el").rstrip())
    if not parts:
        return "No models available.\n"
    return "\n\n".join(parts) + "\n"


# -------------------------
# Persistence
# -------------------------


def save_model_bundle(bundle: ModelBundle, path: str) -> None:
    """Save bundle to a .joblib file (backward-compatible).

    Format 1: both models are pure Polynomial -> store coef arrays.
    Format 2: at least one model is LinearBasisModel or any
    non-polynomial basis -> store a small descriptor next to coef.
    """

    def pack(m):
        # Pure Polynomial
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
            "sector_edges_deg": list(getattr(m, "sector_edges_deg", []) or []),
        }

    both_poly = isinstance(bundle.az_model, Polynomial) and isinstance(
        bundle.el_model, Polynomial
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


def _coerce_list_of_floats(val):
    if val is None:
        return []
    if isinstance(val, (tuple, list)):
        return [float(x) for x in val]
    if isinstance(val, str) and val.strip():
        return [float(x) for x in val.split(",")]
    return []


def _load_metadata_from_dict(m: dict) -> ModelMetadata:
    base = {
        "degree": m.get("degree", 1),
        "zscore": m.get("zscore", 3.0),
        "ridge_alpha": m.get("ridge_alpha", 0.0),
        "cut_deg": m.get("cut_deg", 0.0),
        "input_offset_unit": m.get("input_offset_unit", "arcmin"),
        "notes": m.get("notes", ""),
        "fourier_k": int(m.get("fourier_k", 0)),
        "periods_deg": _coerce_list_of_floats(m.get("periods_deg", [])),
        "sector_edges_deg": _coerce_list_of_floats(m.get("sector_edges_deg", [])),
        "az_lin_min_deg": float(m.get("az_lin_min_deg", float("nan"))),
        "az_lin_max_deg": float(m.get("az_lin_max_deg", float("nan"))),
        "timestamp_utc": m.get("timestamp_utc", ""),
        "data_hash": m.get("data_hash", ""),
        "source_path": m.get("source_path", ""),
        "library_version": m.get("library_version", ""),
    }
    return ModelMetadata(**base)


def load_model_bundle(path: str) -> ModelBundle:
    d = joblib.load(path)

    # --- Unpack models
    fmt = int(d.get("format", 1))

    def _unpack_model(mdict):
        """Turn a packed model dict into a callable model."""
        mtype = mdict.get("type", "poly")
        if mtype == "poly":
            return Polynomial(np.asarray(mdict["coef"], dtype=float))
        elif mtype == "linear_basis":
            return LinearBasisModel(
                coef=np.asarray(mdict["coef"], dtype=float),
                poly_degree=int(mdict.get("poly_degree", 0)),
                fourier_k=int(mdict.get("fourier_k", 0)),
                periods_deg=mdict.get("periods_deg", None),
                sector_edges_deg=mdict.get("sector_edges_deg", None),
            )
        else:
            raise ValueError(f"Unknown model type: {mtype!r}")

    if fmt == 1 and "az_coef" in d:
        # Legacy: pure polynomials saved as coef arrays
        az_model = Polynomial(np.asarray(d["az_coef"], dtype=float))
        el_model = Polynomial(np.asarray(d["el_coef"], dtype=float))
    else:
        # New format: packed descriptors per axis
        if "az_model" in d:
            az_model = _unpack_model(d["az_model"])
        else:
            # Backstop if someone wrote coef arrays with format!=1
            az_model = Polynomial(np.asarray(d["az_coef"], dtype=float))

        if "el_model" in d:
            el_model = _unpack_model(d["el_model"])
        else:
            el_model = Polynomial(np.asarray(d["el_coef"], dtype=float))

    # --- Metadata / diagnostics (tolerant to older/newer fields) ---
    meta = _load_metadata_from_dict(
        d["meta"]
    )  # keeps only known dataclass fields. :contentReference[oaicite:1]{index=1}

    diag_dict = d.get("diag", {})
    if isinstance(diag_dict, dict):
        diag = FitDiagnostics(**diag_dict)
    else:
        diag = diag_dict  # already a FitDiagnostics

    # Optional arrays if present; keep None otherwise. :contentReference[oaicite:2]{index=2}
    az_lin = d.get("az_lin")
    off_az = d.get("offset_az")
    off_el = d.get("offset_el")

    return ModelBundle(
        az_model=az_model,
        el_model=el_model,
        meta=meta,
        diag=diag,
        az_lin=az_lin,
        offset_az=off_az,
        offset_el=off_el,
    )
