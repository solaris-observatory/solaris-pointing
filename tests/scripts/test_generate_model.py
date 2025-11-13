# tests/scripts/test_generate_model.py
"""
End-to-end and unit tests for `generate_model.py`.

Scope:
- `fit`, `predict`, `merge` commands and artifact generation;
- helper resolvers/parsers;
- property-based testing (Hypothesis) for robustness;
- subprocess-level integration to simulate real CLI usage.

All tests run in a temporary working directory, enforce a headless Matplotlib
backend ("Agg"), and avoid GUI assumptions.
"""

from __future__ import annotations

import io
import json
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Callable, Iterable

import numpy as np
import pandas as pd
import pytest
from hypothesis import given, settings, strategies as st

# ----------------------------------------------------------------------------
# Repo root and CLI script resolution (absolute, independent from cwd changes)
# ----------------------------------------------------------------------------

# tests/scripts/generate_model.py -> repo_root = parents[2]
_REPO_ROOT = Path(__file__).resolve().parents[2]


def _resolve_script_module_path() -> Path:
    """
    Prefer 'scripts/generate_model.py', fallback to 'generate_model.py',
    both under repo root.

    Why: we need an absolute path that does not depend on cwd since tests chdir into
    tmp dirs.
    """
    candidates = [
        _REPO_ROOT / "scripts" / "generate_model.py",
        _REPO_ROOT / "generate_model.py",
    ]
    for c in candidates:
        if c.exists():
            return c.resolve()
    raise FileNotFoundError(
        f"Cannot find 'scripts/generate_model.py' nor 'generate_model.py' "
        f"under repo root: {_REPO_ROOT}"
    )


_SCRIPT_PATH = _resolve_script_module_path()

# Import the CLI module by filename (not by package name), so tests can call `main()`.
import importlib.util as _importlib_util  # noqa: E402

_spec = _importlib_util.spec_from_file_location("generate_model_loaded", _SCRIPT_PATH)
_generate_model = _importlib_util.module_from_spec(_spec)  # type: ignore[arg-type]
assert _spec and _spec.loader
_spec.loader.exec_module(_generate_model)  # type: ignore[assignment]

# Canonical symbols under test
main: Callable[..., int] = _generate_model.main
_resolve_model_path_for_predict: Callable[..., str] = (
    _generate_model._resolve_model_path_for_predict
)
_parse_csv_floats: Callable[..., list[float] | None] = _generate_model._parse_csv_floats
_resolve_input_tsv: Callable[..., str] = _generate_model._resolve_input_tsv


# -------------------------------------
# Test helpers
# -------------------------------------


def _switch_mpl_backend_agg() -> None:
    """Force a non-interactive Matplotlib backend to be safe in CI/headless runs."""
    import matplotlib

    matplotlib.use("Agg", force=True)
    try:
        import matplotlib.pyplot as plt  # noqa: F401

        plt.switch_backend("Agg")
    except Exception:
        # Some Matplotlib versions may not support switch_backend;
        # forcing above is enough.
        pass


def make_tsv(tmp: Path, name: str, n: int = 48) -> Path:
    """
    Create a small but realistic TSV under offsets/{name} with the expected columns.

    Notes (why):
    - Offsets are smooth functions of azimuth with light noise so that fits are stable.
    - Includes a well-formed ISO timestamp column for period detection.
    """
    offsets_dir = tmp / "offsets"
    offsets_dir.mkdir(parents=True, exist_ok=True)

    az = np.linspace(0.0, 350.0, n)
    rng = np.random.default_rng(12345)
    off_az = (
        0.01
        + 1e-4 * (az - 180)
        + 3e-4 * np.sin(np.deg2rad(az))
        + rng.normal(0, 5e-5, size=n)
    )
    off_el = (
        -0.005
        + 8e-5 * (az - 120)
        + 2e-4 * np.cos(np.deg2rad(2 * az))
        + rng.normal(0, 5e-5, size=n)
    )
    ts = pd.date_range("2025-08-01T10:00:00Z", periods=n, freq="5min")

    df = pd.DataFrame(
        {
            "timestamp": ts.astype(str),
            "azimuth": az,
            "offset_az": off_az,
            "offset_el": off_el,
        }
    )
    path = offsets_dir / name
    df.to_csv(path, sep="\t", index=False)
    return path


@pytest.fixture(autouse=True)
def _cwd_tmp_and_mpl(tmp_path, monkeypatch):
    """Operate inside a temp directory and force headless plotting."""
    monkeypatch.chdir(tmp_path)
    _switch_mpl_backend_agg()
    yield


# ------------------------------
# Unit tests: FIT
# ------------------------------


def test_fit_both_axes_produces_artifacts(tmp_path):
    """`fit` on both axes should generate per-axis models, unified model,
    plots, summaries, and sidecars."""
    make_tsv(tmp_path, "foo.tsv", n=60)
    rc = main(
        ["fit", "foo.tsv", "--degree", "2", "--zscore", "2.5", "--plot-unit", "arcmin"]
    )
    assert rc == 0

    models = Path("models")
    # Per-axis models and unified bundle
    assert (models / "foo_az.joblib").exists()
    assert (models / "foo_el.joblib").exists()
    assert (models / "foo.joblib").exists()
    # Per-axis and unified summaries
    assert (models / "foo_summary_az.txt").exists()
    assert (models / "foo_summary_el.txt").exists()
    assert (models / "foo_summary.txt").exists()
    # Per-axis plots
    assert (models / "foo_az.png").exists()
    assert (models / "foo_el.png").exists()
    # Sidecars contain backend kind (default 1d)
    meta_az = json.loads(
        (models / "foo_az.joblib.meta.json").read_text(encoding="utf-8")
    )
    meta_el = json.loads(
        (models / "foo_el.joblib.meta.json").read_text(encoding="utf-8")
    )
    assert meta_az.get("backend_kind") == "1d"
    assert meta_el.get("backend_kind") == "1d"


def test_fit_multiple_files_produces_combined_plots(tmp_path):
    """`fit` with multiple input TSVs should save combined plots for AZ and EL."""
    make_tsv(tmp_path, "a.tsv", n=40)
    make_tsv(tmp_path, "b.tsv", n=40)
    rc = main(["fit", "a.tsv", "b.tsv", "--degree", "2", "--plot-unit", "arcmin"])
    assert rc == 0
    assert (Path("models") / "a+b_az.png").exists()
    assert (Path("models") / "a+b_el.png").exists()


# -----------------------------
# Unit tests: PREDICT
# -----------------------------


def test_predict_both_axes_output_format(tmp_path, ANGLE_4DEC_RE):
    """
    `predict` with no axis selector prints both AZ and EL offsets,
    with 4 decimals and units.
    The azimuth is echoed with 4 decimals and a trailing degree symbol.
    """
    make_tsv(tmp_path, "foo.tsv", n=60)
    assert main(["fit", "foo.tsv"]) == 0

    # Capture stdout by temporarily redirecting sys.stdout
    buf = io.StringIO()
    old = sys.stdout
    try:
        sys.stdout = buf
        assert main(["predict", "foo", "--azimuth", "12.0", "--unit", "arcsec"]) == 0
    finally:
        sys.stdout = old

    out = buf.getvalue().strip()
    assert "az=12.0000°" in out
    # Look for numeric fields with exactly 4 decimals.
    vals = re.findall(r"[-+]?\d+\.\d{4}", out)
    assert len(vals) >= 3
    assert ANGLE_4DEC_RE.match(vals[-1])


def test_predict_az_only_and_el_only(tmp_path):
    """Axis-specific prediction works by stem and by explicit per-axis model path."""
    make_tsv(tmp_path, "foo.tsv", n=60)
    assert main(["fit", "foo.tsv"]) == 0

    # AZ only by stem
    buf = io.StringIO()
    old = sys.stdout
    try:
        sys.stdout = buf
        assert (
            main(["predict", "foo", "--az", "--azimuth", "33.25", "--unit", "arcmin"])
            == 0
        )
    finally:
        sys.stdout = old
    out_az = buf.getvalue().strip()
    assert out_az.startswith("[AZ] az=33.2500°") or "->  [AZ]" in out_az
    assert "offset_az=" in out_az
    assert out_az.endswith(" arcmin")

    # EL only by explicit per-axis file: do NOT pass --el here
    # (resolver would re-append ext/suffix)
    buf = io.StringIO()
    old = sys.stdout
    try:
        sys.stdout = buf
        assert (
            main(
                [
                    "predict",
                    "models/foo_el.joblib",
                    "--azimuth",
                    "33.25",
                    "--unit",
                    "arcmin",
                ]
            )
            == 0
        )
    finally:
        sys.stdout = old
    out_el = buf.getvalue().strip()
    # The CLI prints the echo "az=..." then both axis lines;
    # just assert EL info is present.
    assert "az=33.2500°" in out_el
    assert "[EL]" in out_el and "offset_el=" in out_el
    assert out_el.endswith(" arcmin")


def test_predict_conflicting_axis_flags_errors(tmp_path):
    """`--az` and `--el` are mutually exclusive and should cause early exit."""
    make_tsv(tmp_path, "foo.tsv", n=20)
    assert main(["fit", "foo.tsv"]) == 0
    with pytest.raises(SystemExit):
        main(["predict", "foo", "--az", "--el", "--azimuth", "10"])


# -------------------------
# Unit tests: MERGE
# -------------------------


def test_merge_builds_unified_from_per_axis(tmp_path):
    """`merge` combines per-axis bundles into a unified `<stem>.joblib`
    and writes a sidecar."""
    make_tsv(tmp_path, "bar.tsv", n=50)
    assert main(["fit", "bar.tsv", "--az"]) == 0
    assert main(["fit", "bar.tsv", "--el"]) == 0

    models = Path("models")
    assert (models / "bar_az.joblib").exists()
    assert (models / "bar_el.joblib").exists()
    assert not (models / "bar.joblib").exists()

    assert main(["merge", "bar"]) == 0
    assert (models / "bar.joblib").exists()
    assert (models / "bar.joblib.meta.json").exists()


def test_merge_missing_axis_raises(tmp_path):
    """`merge` should fail if one of the per-axis bundles is missing."""
    make_tsv(tmp_path, "baz.tsv", n=30)
    assert main(["fit", "baz.tsv", "--az"]) == 0
    el_path = Path("models/baz_el.joblib")
    if el_path.exists():
        el_path.unlink()
    with pytest.raises(SystemExit):
        main(["merge", "baz"])


# ------------------------------------------
# Unit tests: Helpers (resolvers & parser)
# ------------------------------------------


def test_resolve_model_path_for_predict_variants():
    """Path resolution for predict honors default models dir and axis suffix rules."""
    assert _resolve_model_path_for_predict("alpacino", axis="az") == os.path.join(
        "models", "alpacino_az.joblib"
    )
    assert _resolve_model_path_for_predict("alpacino", axis="el") == os.path.join(
        "models", "alpacino_el.joblib"
    )
    assert _resolve_model_path_for_predict("foo_el", axis="el").endswith(
        "foo_el.joblib"
    )
    assert _resolve_model_path_for_predict("models/x", axis=None) == os.path.join(
        "models", "x.joblib"
    )


def test_parse_csv_floats_ok_and_invalid():
    """CSV float parser accepts blanks/whitespace and rejects invalid tokens."""
    assert _parse_csv_floats("") is None
    vals = _parse_csv_floats("6, 11.25,  90")
    assert vals == [6.0, 11.25, 90.0]
    with pytest.raises(ValueError):
        _parse_csv_floats("6, nope, 90")


def test_resolve_input_tsv_prefix():
    """Input TSV resolver should prepend `offsets/` only when
    no directory is provided."""
    assert _resolve_input_tsv("a.tsv") == os.path.join("offsets", "a.tsv")
    assert _resolve_input_tsv("data/a.tsv").endswith("data/a.tsv")


# ---------------------------------------
# Unit tests: Summaries smoke checks
# ---------------------------------------


def test_summary_files_nonempty_and_contain_mad_lines(tmp_path):
    """Summaries should include MAD_t/MAD_i lines and the Python function
    header for each axis."""
    make_tsv(tmp_path, "sum.tsv", n=40)
    assert main(["fit", "sum.tsv", "--plot-unit", "arcsec"]) == 0
    s_az = Path("models/sum_summary_az.txt").read_text(encoding="utf-8")
    s_el = Path("models/sum_summary_el.txt").read_text(encoding="utf-8")
    assert "MAD_t (arcsec):" in s_az and "MAD_i (arcsec):" in s_az
    assert "def az_offset(az):" in s_az
    assert "MAD_t (arcsec):" in s_el and "MAD_i (arcsec):" in s_el
    assert "def el_offset(az):" in s_el


# --------------------------------------
# Hypothesis: `_parse_csv_floats`
# --------------------------------------


def _floats_to_csv(xs: Iterable[float]) -> str:
    """Render floats similarly to a user-provided CSV string, possibly with
    extra spaces/empties."""
    if not xs:
        return ""
    # Insert sporadic extra spaces and empty tokens to stress the parser.
    parts: list[str] = []
    for i, x in enumerate(xs):
        if i % 3 == 0:
            parts.append(f" {x} ")
        else:
            parts.append(str(x))
        if i % 5 == 4:
            parts.append("")  # represent accidental extra comma -> empty token
    return ",".join(parts)


@given(
    st.lists(
        st.floats(allow_nan=False, allow_infinity=False, width=32),
        min_size=0,
        max_size=10,
    )
)
@settings(deadline=None, max_examples=200)
def test_parse_csv_floats_roundtrip(xs: list[float]):
    """For valid numeric CSVs, parsed list should match the original
    floats list (ignoring empty tokens)."""
    csv = _floats_to_csv(xs)
    parsed = _parse_csv_floats(csv)
    if len(xs) == 0:
        assert parsed is None
    else:
        assert parsed is not None
        # Use approx to avoid tiny repr differences (why).
        assert pytest.approx(parsed) == xs


# Tokens that are definitely not parseable by float(), to avoid false negatives
# (e.g., 'Infinity').
_NON_NUMERIC_TOKENS = [
    "foo",
    "bar",
    "N/A",
    "NULL",
    "--",
    "abc123x",
    "nanx",
    "+infz",
    "??",
    "x.y.z",
]


@given(st.sampled_from(_NON_NUMERIC_TOKENS))
@settings(deadline=None, max_examples=50)
def test_parse_csv_floats_rejects_invalid_tokens(s: str):
    """Any CSV containing clearly non-numeric tokens must raise ValueError."""
    csv = f"1.0, {s}, 2.0"
    with pytest.raises(ValueError):
        _parse_csv_floats(csv)


# -----------------------------------------------
# Hypothesis: `_resolve_model_path_for_predict`
# -----------------------------------------------


@st.composite
def path_and_axis(draw):
    """Generate user paths and axis choices to validate path resolution rules."""
    # Either a bare name or a path with directory separators.
    bare = draw(st.booleans())
    stem = draw(st.from_regex(r"[A-Za-z0-9_\-]{1,12}", fullmatch=True))
    suffix = draw(st.sampled_from(["", "_az", "_el"]))
    ext = draw(st.sampled_from(["", ".joblib"]))
    base = stem + suffix + ext
    if bare:
        user_path = base
    else:
        # Use a simple explicit directory to assert "do not prepend models/"
        user_path = os.path.join("some", "dir", base)
    axis = draw(st.sampled_from([None, "az", "el"]))
    return user_path, axis


@given(path_and_axis())
@settings(deadline=None, max_examples=200)
def test_resolve_model_path_for_predict_hypothesis(case):
    """
    Resolution rules:
    - If no directory, prefix 'models/'.
    - If axis specified, ensure proper _az/_el suffix and `.joblib`.
    - If directory present, do not inject 'models/'.
    """
    user_path, axis = case
    out = _resolve_model_path_for_predict(user_path, axis)

    has_dir = (os.sep in user_path) or (os.altsep and os.altsep in user_path)
    if not has_dir:
        assert out.startswith("models" + os.sep)
    else:
        assert out.startswith("some" + os.sep + "dir")

    assert out.endswith(".joblib")

    base_in = os.path.splitext(os.path.basename(user_path))[0]
    base_out = os.path.splitext(os.path.basename(out))[0]

    # Axis rule:
    # - if input already ends with _az/_el, keep stem and just ensure .joblib;
    # - otherwise, if axis is given, ensure suffix _{axis} was applied.
    already_axis = base_in.endswith("_az") or base_in.endswith("_el")
    if axis in ("az", "el"):
        if already_axis:
            assert base_out.startswith(base_in)
        else:
            assert base_out.endswith(f"_{axis}")


# -----------------------------------
# Subprocess integration tests
# -----------------------------------


def _python_exe() -> str:
    """Path to the Python executable for subprocess runs."""
    return sys.executable


def _script_path() -> Path:
    """Return ABSOLUTE path to the CLI script file to execute via
    subprocess (cwd is tmp)."""
    return _SCRIPT_PATH  # already absolute


def test_cli_subprocess_fit_and_predict(tmp_path, ANGLE_4DEC_RE):
    """
    Run the real script via subprocess:
    - fit (both axes);
    - predict with arcsec units;
    - verify artifacts and stdout format.
    """
    make_tsv(tmp_path, "sps.tsv", n=48)
    env = os.environ.copy()
    env["MPLBACKEND"] = "Agg"  # ensure headless for subprocess too

    # Fit
    cp = subprocess.run(
        [_python_exe(), str(_script_path()), "fit", "sps.tsv", "--plot-unit", "arcsec"],
        cwd=tmp_path,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        check=False,
    )
    assert cp.returncode == 0, f"STDERR:\n{cp.stderr}"
    assert (tmp_path / "models/sps_az.joblib").exists()
    assert (tmp_path / "models/sps_el.joblib").exists()
    assert (tmp_path / "models/sps.joblib").exists()

    # Predict both axes
    cp2 = subprocess.run(
        [
            _python_exe(),
            str(_script_path()),
            "predict",
            "sps",
            "--azimuth",
            "15.5",
            "--unit",
            "arcsec",
        ],
        cwd=tmp_path,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        check=False,
    )
    assert cp2.returncode == 0, f"STDERR:\n{cp2.stderr}"
    out = cp2.stdout.strip()
    assert "az=15.5000°" in out
    vals = re.findall(r"[-+]?\d+\.\d{4}", out)
    assert len(vals) >= 3
    assert ANGLE_4DEC_RE.match(vals[-1])


def test_cli_subprocess_merge(tmp_path):
    """Run per-axis fits and then merge via subprocess; expect unified
    bundle and sidecar."""
    make_tsv(tmp_path, "mrg.tsv", n=36)
    env = os.environ.copy()
    env["MPLBACKEND"] = "Agg"

    for flag in ("--az", "--el"):
        cp = subprocess.run(
            [_python_exe(), str(_script_path()), "fit", "mrg.tsv", flag],
            cwd=tmp_path,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=False,
        )
        assert cp.returncode == 0, f"STDERR:\n{cp.stderr}"

    cp2 = subprocess.run(
        [_python_exe(), str(_script_path()), "merge", "mrg"],
        cwd=tmp_path,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        check=False,
    )
    assert cp2.returncode == 0, f"STDERR:\n{cp2.stderr}"
    assert (tmp_path / "models/mrg.joblib").exists()
    assert (tmp_path / "models/mrg.joblib.meta.json").exists()
