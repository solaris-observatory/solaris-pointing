from __future__ import annotations
import re
from hypothesis import given, settings, HealthCheck, strategies as st

from solaris_pointing.offsets.io import write_offsets_tsv, Metadata, Measurement


# --- Local helpers (avoid function-scoped fixtures in @given tests) ---


def _finite_float():
    # Finite floats, no NaN/Inf
    return st.floats(allow_nan=False, allow_infinity=False, width=64)


def _angle_deg():
    # Match validation range you expect for az/offsets
    return st.floats(
        min_value=-360.0, max_value=720.0, allow_nan=False, allow_infinity=False
    )


def _elev_deg():
    return st.floats(
        min_value=-90.0, max_value=120.0, allow_nan=False, allow_infinity=False
    )


def _humidity_frac_opt():
    return st.one_of(
        st.none(),
        st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False),
    )


def _temp_opt():
    return st.one_of(st.none(), _finite_float())


def _pressure_opt():
    return st.one_of(st.none(), _finite_float())


# Strategy that builds a Measurement object
_MEASUREMENT_STRAT = st.builds(
    Measurement,
    timestamp_iso=st.just("2025-08-01T10:00:00Z"),
    azimuth_deg=_angle_deg(),
    elevation_deg=_elev_deg(),
    offset_az_deg=_finite_float(),
    offset_el_deg=_finite_float(),
    temperature_c=_temp_opt(),
    pressure_hpa=_pressure_opt(),
    humidity_frac=_humidity_frac_opt(),
)

# Regex for floats with exactly 4 decimals
_ANGLE_4DEC_RE = re.compile(r"^-?\d+\.\d{4}$")


def _parse_noncomment_header_and_rows(text: str) -> tuple[str, list[str]]:
    """Return first non-comment, non-blank line (header) and subsequent data rows."""
    lines = [ln.rstrip("\r\n") for ln in text.splitlines()]
    header = None
    rows: list[str] = []
    for ln in lines:
        stripped = ln.strip()
        if not stripped or stripped.startswith("#"):
            continue
        if header is None:
            header = stripped
        else:
            rows.append(stripped)
    if header is None:
        raise AssertionError("no header found in provided text")
    return header, rows


# --- Property-based test (no function-scoped fixtures) ---


@settings(max_examples=60, suppress_health_check=[HealthCheck.too_slow])
@given(st_rows=st.lists(_MEASUREMENT_STRAT, min_size=1, max_size=12))
def test_property_write_and_format(tmp_path_factory, st_rows):
    # Fresh temp dir for each Hypothesis example
    tmpdir = tmp_path_factory.mktemp("prop")
    p = tmpdir / "prop.tsv"

    # Build Metadata inline (no fixture)
    md = Metadata(
        location="MZS, Antarctica",
        antenna_diameter_m=2.0,
        frequency_hz=100e9,
        software_version="2025.08.05",
        created_at_iso="2025-08-05T11:00:00Z",
    )

    write_offsets_tsv(str(p), md, st_rows, append=False)
    text = p.read_text(encoding="utf-8")

    header, rows = _parse_noncomment_header_and_rows(text)
    # header schema stable
    assert header == (
        "timestamp\tazimuth\televation\toffset_az\toffset_el\t"
        "temperature\tpressure\thumidity"
    )

    # every row must have 8 columns, with 4-dec angles and "NaN" for missing env fields
    for line in rows:
        cols = line.split("\t")
        assert len(cols) == 8

        # timestamp string (not empty)
        assert cols[0]

        # angles fixed 4 decimals
        for i in (1, 2, 3, 4):
            assert _ANGLE_4DEC_RE.match(cols[i]), f"bad angle format: {cols[i]}"

        # env fields: any string is okay, but if input None -> "NaN"
        assert cols[5] != ""
        assert cols[6] != ""
        assert cols[7] != ""


# --- Regular (non-Hypothesis) test: fixtures are fine here ---


def test_append_after_property_file(
    tmp_path, md, sample_row, parse_noncomment_header_and_rows
):
    # ensure append branch is also executed after a non-empty file from property test
    p = tmp_path / "prop.tsv"
    write_offsets_tsv(str(p), md, [sample_row], append=False)
    write_offsets_tsv(str(p), md, [sample_row], append=True)
    text = p.read_text(encoding="utf-8")
    _, rows = parse_noncomment_header_and_rows(text)
    assert len(rows) == 2
