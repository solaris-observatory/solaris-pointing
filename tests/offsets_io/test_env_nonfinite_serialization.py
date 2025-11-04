from __future__ import annotations
import math

from solaris_pointing.offset_io import Metadata, Measurement, write_offsets_tsv


# Integration test: ensure env fields that are non-finite are serialized as "NaN".
def test_env_fields_nonfinite_are_serialized_as_NaN(tmp_path):
    p = tmp_path / "env_nonfinite.tsv"

    md = Metadata(
        location="MZS, Antarctica",
        antenna_diameter_m=2.0,
        frequency_hz=100e9,
        software_version="2025.08.05",
        created_at_iso="2025-08-05T11:00:00Z",
    )

    # humidity_frac cannot be non-finite (it would fail validation); keep it None.
    row = Measurement(
        timestamp_iso="2025-08-01T10:00:00Z",
        azimuth_deg=123.456,  # must be finite (validated)
        elevation_deg=45.789,  # must be finite (validated)
        offset_az_deg=0.0034,  # must be finite (validated)
        offset_el_deg=-0.0023,  # must be finite (validated)
        temperature_c=math.nan,  # -> "NaN"
        pressure_hpa=math.inf,  # -> "NaN"
        humidity_frac=None,  # -> "NaN"
    )

    write_offsets_tsv(str(p), md, [row], append=False)
    text = p.read_text(encoding="utf-8")

    # Check the last three columns (temperature, pressure, humidity)
    # are serialized as "NaN".
    last_line = [ln for ln in text.splitlines() if ln and not ln.startswith("#")][-1]
    cols = last_line.split("\t")
    assert cols[5] == "NaN"  # temperature
    assert cols[6] == "NaN"  # pressure
    assert cols[7] == "NaN"  # humidity
