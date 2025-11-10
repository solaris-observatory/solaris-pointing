import io
import os
import math
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# Import the module under test exactly as requested by the user
from solaris_pointing.offsets.fitting import model_1d


# -----------------------
# Helpers for building TSVs
# -----------------------


def _write_tsv(path: Path, header, rows):
    """Write a simple text file with a header and rows joined by newlines."""
    txt = "\n".join([header] + rows) + "\n"
    path.write_text(txt)


def _synth_data_regular(n=50, noise=0.0, unit="deg"):
    """Create a synthetic dataset with known offsets.

    The ground-truth functions (in degrees) are:
        offset_az(x) = 0.01 + 0.002*x + 0.0001*x^2
        offset_el(x) = -0.02 + 0.0005*x

    Optionally, Gaussian noise can be added to both axes. The function returns a
    pair (header, rows) suitable for writing to a TSV/CSV file. The offsets are
    converted to the requested *unit* for I/O, but the numeric model remains
    defined in degrees for clarity.
    """
    rng = np.random.default_rng(0)
    az = np.linspace(10, 350, n)
    off_az = 0.01 + 0.002 * az + 0.0001 * az**2
    off_el = -0.02 + 0.0005 * az
    if noise > 0:
        off_az = off_az + rng.normal(0, noise, size=az.size)
        off_el = off_el + rng.normal(0, noise, size=az.size)

    # Convert to requested unit for writing
    fac = {"deg": 1.0, "arcmin": 60.0, "arcsec": 3600.0}[unit]
    off_az_u = off_az * fac
    off_el_u = off_el * fac

    header = "azimuth\toffset_az\toffset_el"
    rows = [f"{a:.6f}\t{za:.8f}\t{ze:.8f}" for a, za, ze in zip(az, off_az_u, off_el_u)]
    return header, rows


# -----------------------
# Unit conversion + reader
# -----------------------


@pytest.mark.parametrize(
    "unit,expect",
    [
        ("deg", 1.0),
        ("degree", 1.0),
        ("degrees", 1.0),
        ("arcmin", 1 / 60.0),
        ("arcminute", 1 / 60.0),
        ("arcminutes", 1 / 60.0),
        ("arcsec", 1 / 3600.0),
        ("arcsecond", 1 / 3600.0),
        ("arcseconds", 1 / 3600.0),
        (None, 1.0),
    ],
)
def test_offset_unit_to_deg_factor(unit, expect):
    """_offset_unit_to_deg_factor should map unit strings to multiplicative factors."""
    assert math.isclose(model_1d._offset_unit_to_deg_factor(unit), expect)


def test_offset_unit_to_deg_factor_invalid():
    """Unknown units should raise ValueError."""
    with pytest.raises(ValueError):
        model_1d._offset_unit_to_deg_factor("gradians")


def test_read_offsets_tsv_tab_and_autodetect_and_comments(tmp_path: Path, monkeypatch):
    """Read both TSV and CSV files, exercising comments and fallback autodetect.

    The module tries TSV first (sep='\t') and falls back to autodetect (sep=None)
    only if the initial attempt fails. We monkeypatch pandas.read_csv to simulate
    failure on the first attempt for the CSV case and ensure we go through the
    autodetect branch.
    """
    # --- Tab-separated with comments and blank lines ---
    hdr, rows = _synth_data_regular(n=5, unit="arcmin")
    rows_c = ["# comment line", "", "   "] + rows
    p1 = tmp_path / "a.tsv"
    _write_tsv(p1, hdr, rows_c)

    df1 = model_1d.read_offsets_tsv(str(p1), input_offset_unit="arcmin")
    assert set(df1.columns) >= {"azimuth", "offset_az", "offset_el"}
    # Values must be converted to degrees
    assert df1["offset_az"].abs().max() < 20  # polynomial tops ~12.96°

    # --- Comma-separated: force autodetect path by failing the TSV-first attempt ---
    az = [0, 90, 180]
    off_a = [0.0, 0.1, -0.2]
    off_e = [0.05, 0.0, -0.1]
    text = "azimuth,offset_az,offset_el\n" + "\n".join(
        f"{a},{za},{ze}" for a, za, ze in zip(az, off_a, off_e)
    )
    p2 = tmp_path / "b.csv"
    p2.write_text(text)

    import pandas as _pd

    orig_read_csv = _pd.read_csv
    calls = {"n": 0}

    def fake_read_csv(*args, **kwargs):
        calls["n"] += 1
        # First call: module tries sep='\t' -> raise to trigger fallback
        if calls["n"] == 1 and kwargs.get("sep") == "\t":
            raise ValueError("Simulated TSV read failure to test autodetect")
        return orig_read_csv(*args, **kwargs)

    monkeypatch.setattr(_pd, "read_csv", fake_read_csv)
    df2 = model_1d.read_offsets_tsv(str(p2))
    assert len(df2) == 3
    assert np.isclose(df2["offset_az"].iloc[1], 0.1)


# -----------------------
# Angle utilities
# -----------------------


def test_wrap_and_sector_dummies():
    """_wrap_deg keeps angles in [0, 360), and sector dummies shape follows S-1 columns.

    For edges [60, 210] there are S=3 sectors -> (S-1)=2 one-hot columns.
    """
    A = np.array([-10, 0, 59, 60, 209, 210, 359, 360, 721], float)
    W = model_1d._wrap_deg(A)
    assert np.all((W >= 0) & (W < 360))

    # No edges => empty matrix
    D0 = model_1d._sector_dummy_columns(W, [])
    assert D0.shape == (W.size, 0)

    # With edges [60,210] we get 3 sectors -> (S-1)=2 columns
    D = model_1d._sector_dummy_columns(W, [60, 210])
    assert D.shape == (W.size, 2)  # Sectors=3 -> (S-1)=2 columns

    # Rows in sector [60,210) should be one-hot across the 2 columns
    sel = (W >= 60) & (W < 210)
    assert (D[sel].sum(axis=1) == 1).all()


def test_circular_gaps_and_unwrap():
    """unwrap_azimuth returns a linearized azimuth consistent with _unwrap_single."""
    # Single point path
    idx, start, end = model_1d._circular_gaps_deg(np.array([42.0]))
    assert idx == 0 and start == 42.0 and end == 42.0

    # Two clusters to force a large gap around ~180°
    az = np.concatenate([np.linspace(10, 40, 5), np.linspace(200, 220, 5)])
    az_lin, cut, lo, hi = model_1d.unwrap_azimuth(az)
    assert lo <= az_lin.min() <= az_lin.max() <= hi
    assert 0 <= cut < 360  # reasonable cut angle

    # _unwrap_single consistent with unwrap_azimuth
    for az_, az_lin_ in zip(az, az_lin):
        u = model_1d._unwrap_single(az_, cut)
        assert math.isclose(u, az_lin_)


# -----------------------
# Robust utilities
# -----------------------


def test_mad_and_robust_mask():
    """_mad is zero for distributions with a zero median absolute deviation,
    positive for genuinely spread data; robust mask tolerates zero-scale.
    """
    # Case with median=0 and absolute deviations mostly zero -> MAD == 0
    x = np.array([0, 0, 0, 10, 0, 0], float)
    assert model_1d._mad(x) == 0.0
    # With MAD=0, robust mask should fall back to keeping everything
    mask_x = model_1d._robust_mask_from_residuals(x, 2.5)
    assert mask_x.all()

    # Case with genuine spread -> MAD > 0
    y = np.array([-1.0, 0.0, 1.0], float)
    assert model_1d._mad(y) > 0.0
    mask_y = model_1d._robust_mask_from_residuals(y, 1.0)
    assert mask_y.any()

    # Zero-scale path with all zeros -> all True
    mask0 = model_1d._robust_mask_from_residuals(np.zeros(10), 2.5)
    assert mask0.all()


# -----------------------
# Design matrices and solvers
# -----------------------


def test_design_matrices_and_fit_paths():
    """Design matrices shapes and solver return types should be consistent."""
    x = np.linspace(0, 300, 20)

    # Polynomial only
    Xp = model_1d._design_matrix_poly(x, 3)
    assert Xp.shape == (x.size, 4)

    Xl = model_1d._design_matrix_linear(x, 2, fourier_k=0)
    assert np.allclose(Xl, model_1d._design_matrix_poly(x, 2))

    # With Fourier + custom periods + sector dummies
    Xf = model_1d._design_matrix_full(
        x_deg=x, degree=2, fourier_k=2, periods_deg=[90, 45], sector_edges_deg=[60, 210]
    )
    # Columns = poly(3) + 2*(K=2)=4 + 2*len(periods)=4 + (S-1)=2  => 13? Wait, check:
    # For degree=2 -> poly columns = degree+1 = 3
    # Fourier_k=2 -> cos/sin pairs = 2*K = 4
    # periods=[90,45] -> extra cos/sin pairs per period = 2*len(periods) = 4
    # sector edges [60,210] -> S=3 -> (S-1)=2
    # Total = 3 + 4 + 4 + 2 = 13
    assert Xf.shape == (x.size, 13)

    # Solver returns Polynomial when no extras
    y = 1 + 2 * x + 0.1 * x**2
    m_poly = model_1d._fit_linear_ridge(x, y, degree=2, fourier_k=0, alpha=0.0)
    from numpy.polynomial import Polynomial

    assert isinstance(m_poly, Polynomial)

    # With extras it returns LinearBasisModel
    y2 = y + 0.5 * np.cos(np.deg2rad(x))
    m_lin = model_1d._fit_linear_ridge(x, y2, degree=2, fourier_k=1, alpha=0.1)
    assert isinstance(m_lin, model_1d.LinearBasisModel)

    # __call__ should return predictions with the same shape
    yhat = m_lin(x)
    assert yhat.shape == x.shape

    # describe() should contain key fields
    dsc = m_lin.describe()
    for k in ["type", "poly_degree", "fourier_k", "periods_deg", "sector_edges_deg"]:
        assert k in dsc


def test_two_pass_fit_with_fallback_mask():
    """Two-pass fit should fallback to all points if too few inliers survive pass-1."""
    x = np.linspace(0, 10, 6)
    y = np.array([0, 100, -100, 100, -100, 0], float)
    m, mask, r2, rmse, mae = model_1d._two_pass_fit(
        x, y, degree=2, zthr=0.1, ridge_alpha=0.0, fourier_k=0
    )
    assert mask.dtype == bool
    assert mask.sum() == x.size  # fallback path
    assert isinstance(r2, float) and isinstance(rmse, float) and isinstance(mae, float)


# -----------------------
# Public API end-to-end
# -----------------------


def test_fit_predict_summary_and_persistence_poly_only(tmp_path: Path):
    """End-to-end check for: fit -> predict -> summaries -> save/load
    (polynomial path)."""
    # Prepare a small dataset in degrees
    hdr, rows = _synth_data_regular(n=40, noise=0.0, unit="deg")
    p = tmp_path / "data.tsv"
    _write_tsv(p, hdr, rows)

    bundle = model_1d.fit_models_from_tsv(
        str(p),
        degree=2,
        zscore=2.5,
        ridge_alpha=0.0,
        input_offset_unit="deg",
    )

    # Predict within range
    off_az, off_el = model_1d.predict_offsets_deg(bundle, az_deg=100.0)
    assert isinstance(off_az, float) and isinstance(off_el, float)

    # Outside range (upper side): request az that unwraps to > (max + margin)
    az_out = bundle.meta.az_lin_max_deg + 6.0  # margin in code is 5 deg
    with pytest.raises(ValueError):
        model_1d.predict_offsets_deg(bundle, az_deg=az_out, allow_extrapolation=False)

    # If allow_extrapolation=True, it should return floats instead
    off_az2, off_el2 = model_1d.predict_offsets_deg(
        bundle, az_deg=az_out, allow_extrapolation=True
    )
    assert isinstance(off_az2, float) and isinstance(off_el2, float)

    # model_summary_axis for both axes (Polynomial path)
    s_az = model_1d.model_summary_axis(bundle, "az")
    s_el = model_1d.model_summary_axis(bundle, "el")
    assert "Pointing model summary (AZ only" in s_az
    assert "Fit equation:" in s_az and "y(x) =" in s_az
    assert "Pointing model summary (EL only" in s_el

    # model_summary aggregates both
    s_both = model_1d.model_summary(bundle)
    assert "AZ only" in s_both and "EL only" in s_both

    # Save (format 1) as both polynomials
    out = tmp_path / "m.joblib"
    model_1d.save_model(bundle, str(out))
    assert out.exists()

    # Load back (format 1) and exercise summaries again
    b2 = model_1d.load_model(str(out))
    _ = model_1d.model_summary_axis(b2, "az")
    _ = model_1d.model_summary(b2)


def test_fit_predict_summary_and_persistence_linear_basis(tmp_path: Path, monkeypatch):
    """End-to-end check for linear basis models (Fourier/periods/sectors)."""
    # Dataset in arcsec to exercise conversion + comments stripping + autodetect
    hdr, rows = _synth_data_regular(n=60, noise=0.0, unit="arcsec")
    rows = [
        "# some header comment",
        hdr.replace("\t", ","),
        *[r.replace("\t", ",") for r in rows],
    ]
    p = tmp_path / "c.csv"
    p.write_text("\n".join(rows) + "\n")

    # Force the TSV-first attempt to fail so the code uses the autodetect branch
    import pandas as _pd

    orig_read_csv = _pd.read_csv
    calls = {"n": 0}

    def fake_read_csv(*args, **kwargs):
        calls["n"] += 1
        if calls["n"] == 1 and kwargs.get("sep") == "\t":
            raise ValueError("Simulated TSV read failure for autodetect path")
        return orig_read_csv(*args, **kwargs)

    monkeypatch.setattr(_pd, "read_csv", fake_read_csv)

    bundle = model_1d.fit_models_from_tsv(
        str(p),
        degree=2,
        zscore=3.0,
        ridge_alpha=0.05,
        fourier_k=2,
        periods_deg=[90.0],
        sector_edges_deg=[60.0, 210.0],
        input_offset_unit="arcsec",
    )

    # Predictions & summaries (LinearBasisModel path)
    off_az, off_el = model_1d.predict_offsets_deg(bundle, az_deg=123.0)
    assert isinstance(off_az, float) and isinstance(off_el, float)
    saz = model_1d.model_summary_axis(bundle, "az")
    assert "cos(2π·1·x/90)" in saz or "cos(2π·1·x/90.0)" in saz

    # Save -> format 2; Load (packed descriptors)
    out = tmp_path / "m2.joblib"
    model_1d.save_model(bundle, str(out))
    b2 = model_1d.load_model(str(out))

    # Ensure loaded models are callable LinearBasisModel and keep fields
    assert isinstance(b2.az_model, model_1d.LinearBasisModel)
    assert isinstance(b2.el_model, model_1d.LinearBasisModel)
    assert b2.meta.fourier_k == 2
    assert isinstance(b2.diag.rmse_az_deg, float)

    # Invalid axis should raise
    with pytest.raises(ValueError):
        model_1d.model_summary_axis(bundle, "foo")


# -----------------------
# Metadata utilities
# -----------------------


def test_coerce_list_and_load_metadata_roundtrip():
    """_coerce_list_of_floats and _load_metadata_from_dict
    should be robust to inputs."""
    assert model_1d._coerce_list_of_floats(None) == []
    assert model_1d._coerce_list_of_floats([1, "2", 3.5]) == [1.0, 2.0, 3.5]
    assert model_1d._coerce_list_of_floats("1,2,3") == [1.0, 2.0, 3.0]
    # Other strings should fallback to an empty list
    assert model_1d._coerce_list_of_floats("  ") == []

    # load metadata should coerce types and ignore unknown fields
    meta = model_1d._load_metadata_from_dict(
        {
            "degree": 3,
            "zscore": 2.5,
            "ridge_alpha": 0.1,
            "cut_deg": 12.34,
            "input_offset_unit": "arcmin",
            "notes": "x",
            "fourier_k": 2,
            "periods_deg": ["90", "45"],
            "sector_edges_deg": ["60", "210"],
            "az_lin_min_deg": 100.0,
            "az_lin_max_deg": 200.0,
            "timestamp_utc": "2025-10-30T00:00:00Z",
            "data_hash": "abc",
            "source_path": "file.tsv",
            "library_version": "lib 1.0",
            "extra_ignored": 123,
        }
    )
    assert meta.degree == 3 and meta.fourier_k == 2
    assert meta.periods_deg == [90.0, 45.0]
    assert meta.sector_edges_deg == [60.0, 210.0]


# -----------------------
# Additional branch coverage
# -----------------------


def test_read_offsets_tsv_missing_columns_raises(tmp_path: Path):
    """Ensure the 'missing required columns' error branch is executed."""
    p = tmp_path / "bad.csv"
    p.write_text("azimuth,offset_az,WRONG\n1,0.1,0.2\n", encoding="utf-8")
    # Force autodetect path so pandas parses header as separate columns
    import pandas as _pd

    orig_read_csv = _pd.read_csv
    calls = {"n": 0}

    def fake_read_csv(*args, **kwargs):
        calls["n"] += 1
        if calls["n"] == 1 and kwargs.get("sep") == "\t":
            raise ValueError("force autodetect")
        return orig_read_csv(*args, **kwargs)

    import solaris_pointing.offsets.fitting.model_1d as M
    import importlib

    importlib.reload(M)
    from solaris_pointing.offsets.fitting import model_1d as _m

    _pd_read = _pd.read_csv
    try:
        _pd.read_csv = fake_read_csv
        with pytest.raises(ValueError):
            _m.read_offsets_tsv(str(p))
    finally:
        _pd.read_csv = _pd_read


def test_design_matrix_full_skips_none_and_nonpositive_periods():
    """Hit the 'continue' branches for periods None and <= 0"""
    x = np.linspace(0, 100, 5)
    X = model_1d._design_matrix_full(
        x_deg=x,
        degree=1,
        fourier_k=0,
        periods_deg=[None, 0, -10, 30],  # only last is valid
        sector_edges_deg=None,
    )
    # degree=1 -> 2 cols; plus only one valid custom period
    # -> +2 cols => 4 columns total
    assert X.shape == (x.size, 4)


def test_equation_unavailable_branch_and_period_default_and_zero_terms():
    """Cover: periods None defaulting, zero Ak/Bk 'continue', and
    unavailable model branch."""
    # LinearBasisModel with k=1, periods None => defaults to [360.0]
    # Coef layout: [poly c0,c1] + [A1,B1] ; set A1=B1=0 to trigger continue
    coef = np.array([1.0, 0.0, 0.0, 0.0])
    m = model_1d.LinearBasisModel(
        coef=coef, poly_degree=1, fourier_k=1, periods_deg=None
    )
    # Minimal bundle-like with required meta/diag fields
    meta = model_1d.ModelMetadata(
        degree=1,
        zscore=2.5,
        ridge_alpha=0.0,
        cut_deg=0.0,
        input_offset_unit="deg",
        notes="note present",
        fourier_k=1,
        periods_deg=[],
        sector_edges_deg=[],
        az_lin_min_deg=0.0,
        az_lin_max_deg=1.0,
        timestamp_utc="2025-01-01T00:00:00Z",
        data_hash="a" * 64,
        source_path="x.tsv",
        library_version="lib",
    )
    diag = model_1d.FitDiagnostics(
        n_input=1,
        n_kept_az=1,
        n_kept_el=1,
        r2_az=1.0,
        r2_el=1.0,
        rmse_az_deg=0.0,
        rmse_el_deg=0.0,
        mae_az_deg=0.0,
        mae_el_deg=0.0,
    )
    bundle = model_1d.ModelBundle(az_model=m, el_model=m, meta=meta, diag=diag)
    s = model_1d.model_summary_axis(bundle, "az")
    assert "notes:" in s  # hit notes branch

    # 'No model available' branch
    bundle2 = model_1d.ModelBundle(az_model=None, el_model=None, meta=meta, diag=diag)
    assert "No model available" in model_1d.model_summary_axis(bundle2, "az")
    # And combined 'No models available.' from wrapper
    assert model_1d.model_summary(bundle2).startswith("No models available.")

    # Unavailable equation branch: pass object without required attrs
    class Odd:
        def __call__(self, x):  # never used here
            return np.zeros_like(np.asarray(x, float))

    bundle3 = model_1d.ModelBundle(az_model=Odd(), el_model=Odd(), meta=meta, diag=diag)
    s3 = model_1d.model_summary_axis(bundle3, "az")
    assert "<unavailable>" in s3


def test_load_model_backstops_and_mixed_types(tmp_path: Path):
    """Cover load_model branches: unpack poly in format 2, and
    backstops for missing keys."""
    # Mixed types (format 2): az poly dict, el linear_basis dict
    poly = {"type": "poly", "coef": [1.0, 2.0]}
    lin = {
        "type": "linear_basis",
        "coef": [1.0, 0.0, 0.0],
        "poly_degree": 1,
        "fourier_k": 1,
        "periods_deg": [],
        "sector_edges_deg": [],
    }
    meta = {
        "degree": 1,
        "zscore": 2.5,
        "ridge_alpha": 0.0,
        "cut_deg": 0.0,
        "input_offset_unit": "deg",
        "notes": "",
        "fourier_k": 1,
        "periods_deg": [],
        "sector_edges_deg": [],
        "az_lin_min_deg": 0.0,
        "az_lin_max_deg": 1.0,
        "timestamp_utc": "t",
        "data_hash": "h",
        "source_path": "p",
        "library_version": "v",
    }
    diag = {
        "n_input": 1,
        "n_kept_az": 1,
        "n_kept_el": 1,
        "r2_az": 1.0,
        "r2_el": 1.0,
        "rmse_az_deg": 0.0,
        "rmse_el_deg": 0.0,
        "mae_az_deg": 0.0,
        "mae_el_deg": 0.0,
    }
    d1 = {"format": 2, "az_model": poly, "el_model": lin, "meta": meta, "diag": diag}
    p1 = tmp_path / "m_fmt2_mixed.joblib"
    import joblib as _joblib
    import numpy as _np

    _joblib.dump(d1, p1)
    b1 = model_1d.load_model(str(p1))
    from numpy.polynomial import Polynomial as _Poly

    assert isinstance(b1.az_model, _Poly)
    assert isinstance(b1.el_model, model_1d.LinearBasisModel)

    # Backstops: format=2 but missing 'az_model'/'el_model' -> use 'az_coef'/'el_coef'
    d2 = {
        "format": 2,
        "az_coef": [0.0, 1.0],
        "el_coef": [2.0, 0.0],
        "meta": meta,
        "diag": diag,
    }
    p2 = tmp_path / "m_fmt2_backstop.joblib"
    _joblib.dump(d2, p2)
    b2 = model_1d.load_model(str(p2))
    assert isinstance(b2.az_model, _Poly) and isinstance(b2.el_model, _Poly)

    # diag not a dict -> else branch assigning as-is
    diag_obj = model_1d.FitDiagnostics(**diag)
    d3 = {
        "format": 2,
        "az_model": poly,
        "el_model": poly,
        "meta": meta,
        "diag": diag_obj,
    }
    p3 = tmp_path / "m_fmt2_diagobj.joblib"
    _joblib.dump(d3, p3)
    b3 = model_1d.load_model(str(p3))
    assert isinstance(b3.diag, model_1d.FitDiagnostics)


def test_sha256_strip_comments_exception_branch(monkeypatch, tmp_path: Path):
    """Trigger the 'except' branch in _sha256_of_file_strip_comments."""

    # Build a fake object returned by open(..., 'rb') that yields strings (no .decode)
    class FakeFile:
        def __init__(self):
            self._n = 0

        def __iter__(self):
            return self

        def __next__(self):
            if self._n == 0:
                self._n += 1
                return (
                    "this is a str, not bytes"  # will trigger AttributeError on .decode
                )
            raise StopIteration

        # Context manager compatibility
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

    # Patch the builtins.open used by the module
    import builtins

    monkeypatch.setattr(builtins, "open", lambda *a, **k: FakeFile())

    # The function should handle the exception and still return a hex digest
    h = model_1d._sha256_of_file_strip_comments("ignored")
    assert isinstance(h, str) and len(h) == 64


def test_save_model_pack_poly_branch(tmp_path: Path):
    """
    Cover line ~894: in save_model.pack(), hit the 'poly' branch when packing
    a Polynomial inside a format-2 bundle (mixed models).
    """
    import numpy as np
    import joblib
    from numpy.polynomial import Polynomial

    # az -> Polynomial, el -> LinearBasisModel (forces format=2 and pack() on both)
    az_poly = Polynomial([0.5, 0.1])  # degree 1
    el_lin = model_1d.LinearBasisModel(
        coef=np.array([0.0, 1.0, 0.0]),  # [c0, c1, A1] minimal layout
        poly_degree=1,
        fourier_k=1,
        periods_deg=[],
        sector_edges_deg=[],
    )

    meta = model_1d.ModelMetadata(
        degree=1,
        zscore=2.5,
        ridge_alpha=0.0,
        cut_deg=0.0,
        input_offset_unit="deg",
        notes="",
        fourier_k=1,
        periods_deg=[],
        sector_edges_deg=[],
        az_lin_min_deg=0.0,
        az_lin_max_deg=1.0,
        timestamp_utc="2025-01-01T00:00:00Z",
        data_hash="h" * 64,
        source_path="p.tsv",
        library_version="v",
    )
    diag = model_1d.FitDiagnostics(
        n_input=1,
        n_kept_az=1,
        n_kept_el=1,
        r2_az=1.0,
        r2_el=1.0,
        rmse_az_deg=0.0,
        rmse_el_deg=0.0,
        mae_az_deg=0.0,
        mae_el_deg=0.0,
    )
    bundle = model_1d.ModelBundle(
        az_model=az_poly, el_model=el_lin, meta=meta, diag=diag
    )

    out = tmp_path / "mix.joblib"
    model_1d.save_model(bundle, str(out))

    d = joblib.load(out)
    assert d["format"] == 2
    # Must have used pack() with type='poly' for az_model
    assert isinstance(d.get("az_model"), dict)
    assert d["az_model"].get("type") == "poly"


def test_load_model_unknown_type_raises(tmp_path: Path):
    """
    In load_model._unpack_model(), unknown model type triggers ValueError.
    """
    import joblib

    poly = {"type": "poly", "coef": [1.0, 2.0]}
    weird = {"type": "weird", "coef": [0.0]}  # unknown type -> should raise

    meta = {
        "degree": 1,
        "zscore": 2.5,
        "ridge_alpha": 0.0,
        "cut_deg": 0.0,
        "input_offset_unit": "deg",
        "notes": "",
        "fourier_k": 0,
        "periods_deg": [],
        "sector_edges_deg": [],
        "az_lin_min_deg": 0.0,
        "az_lin_max_deg": 1.0,
        "timestamp_utc": "2025-01-01T00:00:00Z",
        "data_hash": "h" * 64,
        "source_path": "p.tsv",
        "library_version": "v",
    }
    diag = {
        "n_input": 1,
        "n_kept_az": 1,
        "n_kept_el": 1,
        "r2_az": 1.0,
        "r2_el": 1.0,
        "rmse_az_deg": 0.0,
        "rmse_el_deg": 0.0,
        "mae_az_deg": 0.0,
        "mae_el_deg": 0.0,
    }

    payload = {
        "format": 2,
        "az_model": weird,
        "el_model": poly,
        "meta": meta,
        "diag": diag,
    }
    p = tmp_path / "unknown_type.joblib"
    joblib.dump(payload, p)

    import pytest

    with pytest.raises(ValueError, match="Unknown model type"):
        model_1d.load_model(str(p))
