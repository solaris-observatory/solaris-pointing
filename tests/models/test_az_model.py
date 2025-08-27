import io
import math
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest

# Module under test
import solaris_pointing.models.az_model as m


def write_tsv(path: Path, df: pd.DataFrame, with_header: bool = True) -> None:
    """Write a TSV file with two commented lines to exercise comment parsing."""
    content = io.StringIO()
    content.write("# This is a header comment line\n")
    content.write("# Another comment\n")
    df.to_csv(content, sep="\t", index=False, header=with_header)
    path.write_text(content.getvalue(), encoding="utf-8")


def make_linear_dataset(n=20, slope_az=0.1, bias_az=1.0, slope_el=-0.05, bias_el=0.5):
    """Create a linear dataset (degrees) for both offsets vs azimuth."""
    az = np.linspace(0.0, 180.0, n)
    off_az = slope_az * az + bias_az
    off_el = slope_el * az + bias_el
    el = np.linspace(10.0, 70.0, n)  # elevation present but unused for fitting
    df = pd.DataFrame(
        {
            "azimuth": az,
            "elevation": el,
            "offset_az": off_az,
            "offset_el": off_el,
        }
    )
    return df, (slope_az, bias_az, slope_el, bias_el)


def test_read_offsets_tsv_deg_and_arcsec_and_comments(tmp_path: Path):
    """read_offsets_tsv parses comments, coerces numerics, drops NaNs, and
    converts arcsec to deg."""
    df_deg, _ = make_linear_dataset(n=5)
    # Inject a NaN row to verify drop
    df_deg.loc[2, "offset_el"] = np.nan

    # 1) File with offsets already in degrees
    p_deg = tmp_path / "deg.tsv"
    write_tsv(p_deg, df_deg)
    out_deg = m.read_offsets_tsv(str(p_deg), input_offset_unit="deg")
    assert set(out_deg.columns) == set(m.REQUIRED_COLUMNS)
    assert len(out_deg) == 4  # one row with NaN was dropped
    assert out_deg["offset_az"].dtype.kind in "fc"
    assert out_deg["offset_el"].dtype.kind in "fc"

    # 2) File with offsets in arcseconds (convert to degrees)
    df_arcsec = df_deg.copy()
    df_arcsec["offset_az"] = df_arcsec["offset_az"] * 3600.0
    df_arcsec["offset_el"] = df_arcsec["offset_el"] * 3600.0
    p_arcsec = tmp_path / "arcsec.tsv"
    write_tsv(p_arcsec, df_arcsec)
    out_arcsec = m.read_offsets_tsv(str(p_arcsec), input_offset_unit="arcsec")

    az_arc = out_arcsec["offset_az"].to_numpy()
    az_deg = out_deg["offset_az"].to_numpy()
    el_arc = out_arcsec["offset_el"].to_numpy()
    el_deg = out_deg["offset_el"].to_numpy()
    np.testing.assert_allclose(az_arc, az_deg)
    np.testing.assert_allclose(el_arc, el_deg)

    # 3) Unknown unit should raise
    with pytest.raises(ValueError):
        _ = m.read_offsets_tsv(str(p_deg), input_offset_unit="radians")


def test_read_offsets_tsv_missing_columns(tmp_path: Path):
    """Missing required columns should raise ValueError."""
    df, _ = make_linear_dataset(n=5)
    # Drop one required column
    df = df.drop(columns=["offset_el"])
    p = tmp_path / "missing.tsv"
    write_tsv(p, df)
    with pytest.raises(ValueError):
        _ = m.read_offsets_tsv(str(p), input_offset_unit="deg")


def test_fit_models_linear_perfect_r2_and_summary(tmp_path: Path):
    """Perfect linear data should yield R^2==1 (after outlier handling)."""
    df, _ = make_linear_dataset(
        n=30, slope_az=0.2, bias_az=0.3, slope_el=-0.1, bias_el=1.1
    )
    models = m.fit_models(
        df["azimuth"],
        df["offset_az"],
        df["offset_el"],
        degree=1,
        zscore=3.0,
    )

    # az_model and el_model should be numpy.polynomial.Polynomial
    from numpy.polynomial import Polynomial

    assert isinstance(models.az_model, Polynomial)
    assert isinstance(models.el_model, Polynomial)

    # R^2 must be exactly 1 (perfect fit on filtered data)
    assert math.isclose(models.az_r2, 1.0, rel_tol=0, abs_tol=1e-12)
    assert math.isclose(models.el_r2, 1.0, rel_tol=0, abs_tol=1e-12)

    # as_tuple helper
    az_model, el_model = models.as_tuple()
    assert az_model is models.az_model and el_model is models.el_model

    # Summary should contain key phrases
    summary = m.model_summary(models)
    assert "offset_az = P_az(azimuth)" in summary
    assert "offset_el = P_el(azimuth)" in summary
    assert "Units: degrees" in summary


def test_fit_models_not_enough_points_raises():
    """If there are fewer than degree+1 points after filtering, raise ValueError."""
    df, _ = make_linear_dataset(n=3)
    with pytest.raises(ValueError):
        _ = m.fit_models(df["azimuth"], df["offset_az"], df["offset_el"], degree=3)


def test_outlier_rejection_and_prediction(tmp_path: Path):
    """A large outlier in target should be rejected; predictions should be
    accurate."""
    df, params = make_linear_dataset(
        n=40, slope_az=0.15, bias_az=-0.2, slope_el=0.05, bias_el=0.1
    )
    s_az, b_az, s_el, b_el = params
    # Inject an obvious outlier in both targets
    df.loc[10, "offset_az"] += 100.0
    df.loc[20, "offset_el"] -= 100.0

    models = m.fit_models(df["azimuth"], df["offset_az"], df["offset_el"], degree=1)

    # After outlier removal, R^2 should be (almost) 1 for linear data
    assert models.az_r2 > 0.999
    assert models.el_r2 > 0.999

    # Predict at a known azimuth and compare to ground-truth linear functions
    az_test = 77.0
    d_az_pred, d_el_pred = m.predict_offsets_deg(
        models.az_model, models.el_model, az_test
    )
    d_az_true = s_az * az_test + b_az
    d_el_true = s_el * az_test + b_el
    assert abs(d_az_pred - d_az_true) < 1e-6
    assert abs(d_el_pred - d_el_true) < 1e-6


def test_fit_models_from_tsv_arcsec(tmp_path: Path):
    """fit_models_from_tsv should read and convert arcsec to degrees before fitting."""
    df, _ = make_linear_dataset(
        n=12, slope_az=0.05, bias_az=0.0, slope_el=-0.02, bias_el=0.2
    )
    # Store offsets in arcseconds in the TSV
    df_arcsec = df.copy()
    df_arcsec["offset_az"] *= 3600.0
    df_arcsec["offset_el"] *= 3600.0
    p = tmp_path / "input_arcsec.tsv"
    write_tsv(p, df_arcsec)

    models = m.fit_models_from_tsv(
        str(p), degree=1, zscore=3.0, input_offset_unit="arcsec"
    )
    assert models.az_r2 > 0.999999
    assert models.el_r2 > 0.999999


def test_save_and_load_models_roundtrip(tmp_path: Path):
    """save_models and load_models should persist and restore Polynomial models."""
    df, _ = make_linear_dataset(
        n=25, slope_az=0.12, bias_az=0.05, slope_el=0.03, bias_el=-0.04
    )
    models = m.fit_models(df["azimuth"], df["offset_az"], df["offset_el"], degree=1)

    az_path = tmp_path / "az_model.joblib"
    el_path = tmp_path / "el_model.joblib"
    m.save_models(models, str(az_path), str(el_path))

    az_model, el_model = m.load_models(str(az_path), str(el_path))
    # Predict something just to ensure models work after reload
    a, b = m.predict_offsets_deg(az_model, el_model, 33.3)
    assert isinstance(a, float) and isinstance(b, float)


def test_load_models_type_error(tmp_path: Path):
    """load_models should raise TypeError if loaded objects are not Polynomial."""
    bad_az = tmp_path / "bad_az.joblib"
    bad_el = tmp_path / "bad_el.joblib"
    joblib.dump({"not": "a polynomial"}, bad_az)
    joblib.dump(123, bad_el)

    with pytest.raises(TypeError):
        _ = m.load_models(str(bad_az), str(bad_el))


def test_remove_outliers_zero_std_branch():
    """Exercise the zero-std branch of _remove_outliers_z (mask should be all True)."""
    x = np.array([10.0, 20.0, 30.0, 40.0])
    y = np.array([5.0, 5.0, 5.0, 5.0])
    x2, y2 = m._remove_outliers_z(x, y, z=3.0)
    np.testing.assert_array_equal(x2, x)
    np.testing.assert_array_equal(y2, y)


def test_constant_offsets_r2_is_one():
    """When ss_tot == 0 (constant target), R^2 should be 1.0 per implementation."""
    az = np.linspace(0.0, 10.0, 10)
    off_az = np.full_like(az, 0.123)  # constant
    off_el = np.full_like(az, -0.456)  # constant
    models = m.fit_models(az, off_az, off_el, degree=0, zscore=3.0)
    assert models.az_r2 == pytest.approx(1.0, abs=0, rel=0)
    assert models.el_r2 == pytest.approx(1.0, abs=0, rel=0)
