from __future__ import annotations
import re
from pathlib import Path
import pytest
from hypothesis import strategies as st

from solaris_pointing.offsets.io import Metadata, Measurement

# ---------- Shared fixtures ----------


@pytest.fixture
def md() -> Metadata:
    """Provide a fixed Metadata object for reproducible tests."""
    return Metadata(
        location="MZS, Antarctica",
        antenna_diameter_m=2.0,
        frequency_ghz=100e9,
        software_version="2025.08.05",
        created_at_iso="2025-08-05T11:00:00Z",
    )


@pytest.fixture
def sample_row() -> Measurement:
    """Provide a representative Measurement row."""
    return Measurement(
        map_id="250103T000235_OASI",
        timestamp_iso="2025-08-01T10:00:00Z",
        azimuth_deg=123.456,
        elevation_deg=45.789,
        offset_az_deg=0.0034,
        offset_el_deg=-0.0023,
        temperature_c=None,
        pressure_hpa=None,
        humidity_frac=None,
    )


@pytest.fixture
def ANGLE_4DEC_RE():
    """Regex for floats with exactly 4 decimals."""
    return re.compile(r"^-?\d+\.\d{4}$")


@pytest.fixture
def parse_noncomment_header_and_rows():
    """Return first non-comment header and data rows from TSV text."""

    def _parser(text: str) -> tuple[str, list[str]]:
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

    return _parser


# ---------- Hypothesis strategies ----------


def finite_float():
    return st.floats(allow_nan=False, allow_infinity=False, width=64)


def angle_deg():
    return st.floats(
        min_value=-360.0, max_value=720.0, allow_nan=False, allow_infinity=False
    )


def elev_deg():
    return st.floats(
        min_value=-90.0, max_value=120.0, allow_nan=False, allow_infinity=False
    )


def humidity_frac_opt():
    return st.one_of(
        st.none(),
        st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False),
    )


def temp_opt():
    return st.one_of(st.none(), finite_float())


def pressure_opt():
    return st.one_of(st.none(), finite_float())


@pytest.fixture
def measurement_strategy():
    """Provide a Hypothesis strategy that generates Measurement objects."""
    return st.builds(
        Measurement,
        timestamp_iso=st.just("2025-08-01T10:00:00Z"),
        azimuth_deg=angle_deg(),
        elevation_deg=elev_deg(),
        offset_az_deg=finite_float(),
        offset_el_deg=finite_float(),
        temperature_c=temp_opt(),
        pressure_hpa=pressure_opt(),
        humidity_frac=humidity_frac_opt(),
    )
