from __future__ import annotations
import math
import pytest

from solaris_pointing.offset_io.tsv import Metadata, Measurement


def test_metadata_valid(md: Metadata):
    assert md.location.startswith("MZS")
    assert md.antenna_diameter_m == 2.0
    assert md.frequency_hz == 100e9


@pytest.mark.parametrize("diam", [0.0, -1.0])
def test_metadata_invalid_diameter(diam: float):
    with pytest.raises(ValueError):
        Metadata("X", diam, 1.0, "v")


@pytest.mark.parametrize("freq", [0.0, -1.0])
def test_metadata_invalid_frequency(freq: float):
    with pytest.raises(ValueError):
        Metadata("X", 1.0, freq, "v")


def test_measurement_valid_none_env_ok():
    m = Measurement(
        timestamp_iso="2025-08-01T10:00:00Z",
        azimuth_deg=0.0,
        elevation_deg=0.0,
        offset_az_deg=0.0,
        offset_el_deg=0.0,
        temperature_c=None,
        pressure_hpa=None,
        humidity_frac=None,
    )
    assert m.azimuth_deg == 0.0


@pytest.mark.parametrize("az", [-361.0, 721.0])
def test_measurement_az_range(az: float):
    with pytest.raises(ValueError):
        Measurement("2025-08-01T10:00:00Z", az, 0.0, 0.0, 0.0)


@pytest.mark.parametrize("elv", [-91.0, 121.1])
def test_measurement_el_range(elv: float):
    with pytest.raises(ValueError):
        Measurement("2025-08-01T10:00:00Z", 0.0, elv, 0.0, 0.0)


@pytest.mark.parametrize("bad", [math.inf, -math.inf, math.nan])
def test_measurement_offsets_finite(bad: float):
    with pytest.raises(ValueError):
        Measurement("2025-08-01T10:00:00Z", 0.0, 0.0, bad, 0.0)
    with pytest.raises(ValueError):
        Measurement("2025-08-01T10:00:00Z", 0.0, 0.0, 0.0, bad)


@pytest.mark.parametrize("h", [-0.1, 1.1])
def test_measurement_humidity_bounds(h: float):
    with pytest.raises(ValueError):
        Measurement("2025-08-01T10:00:00Z", 0.0, 0.0, 0.0, 0.0, humidity_frac=h)
