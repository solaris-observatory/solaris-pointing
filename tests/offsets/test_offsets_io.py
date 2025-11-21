import io as _pyio
import os
import runpy
from pathlib import Path
from datetime import datetime, timezone

import pytest

from solaris_pointing.offsets import io


def test_metadata_created_iso_or_now(monkeypatch):
    # Explicit timestamp path
    ts = "2025-01-01T00:00:00Z"
    md = io.Metadata(
        site_location="X",
        antenna_diameter_m=1.0,
        frequency_ghz=100.0,
        software_url="https://...",
        created_at_iso=ts,
    )
    assert md.created_iso_or_now() == ts

    # "Now" path: patch datetime to return a known instant
    class DummyDT:
        @staticmethod
        def now(tz=None):
            assert tz is timezone.utc
            return datetime(2025, 1, 2, 3, 4, 5, tzinfo=timezone.utc)

    monkeypatch.setattr(io, "datetime", DummyDT)
    md2 = io.Metadata(
        site_location="Y",
        antenna_diameter_m=2.0,
        frequency_ghz=90.0,
        software_url="https://...",
        created_at_iso=None,
    )
    assert md2.created_iso_or_now() == "2025-01-02T03:04:05Z"


def test_measurement_validation_errors():
    # map_id must be non-empty string
    with pytest.raises(ValueError, match="map_id"):
        io.Measurement(
            map_id="",
            timestamp_iso="2025-01-03T00:02:35Z",
            azimuth_deg=0.0,
            elevation_deg=0.0,
            offset_az_deg=0.0,
            offset_el_deg=0.0,
        )
    # timestamp must contain 'T'
    with pytest.raises(ValueError, match="ISO 8601"):
        io.Measurement(
            map_id="id",
            timestamp_iso="2025-01-0300:02:35Z",
            azimuth_deg=0.0,
            elevation_deg=0.0,
            offset_az_deg=0.0,
            offset_el_deg=0.0,
        )


def test_private_formatters_and_row_line():
    m = io.Measurement(
        map_id="A",
        timestamp_iso="2025-01-03T00:02:35Z",
        azimuth_deg=14.6622,
        elevation_deg=37.7847,
        offset_az_deg=-0.08294,
        offset_el_deg=0.07004,
        temperature_c=None,
        pressure_hpa=1013.25,
        humidity_frac=0.45,
    )
    # _fmt_4dec_or_nan
    assert io._fmt_4dec_or_nan(1.23456) == "1.2346"
    assert io._fmt_4dec_or_nan(None) == "NaN"
    # _fmt_default_or_nan
    assert io._fmt_default_or_nan("x") == "x"
    assert io._fmt_default_or_nan(12.3) == "12.3"
    assert io._fmt_default_or_nan(None) == "NaN"
    # _row_to_line formatting, including decimals and NaN
    line = io._row_to_line(m)
    parts = line.strip().split("\t")
    assert parts[0] == "A" and parts[1] == "2025-01-03T00:02:35Z"
    # 4-decimal fields
    assert parts[2] == "14.6622" and parts[3] == "37.7847"
    assert parts[4] == "-0.0829" and parts[5] == "0.0700"
    assert parts[6] == "NaN" and parts[7] == "1013.25" and parts[8] == "0.45"


def test_expected_columns_and_header_line(tmp_path: Path):
    expected = io._expected_columns()
    assert expected == [
        "map_id",
        "timestamp",
        "azimuth",
        "elevation",
        "offset_az",
        "offset_el",
        "temperature",
        "pressure",
        "humidity",
    ]
    # Ensure header writer matches expected (exact tokens, tab-separated)
    p = tmp_path / "h.tsv"
    with p.open("w", encoding="utf-8") as f:
        io._write_column_header(f)
    header = p.read_text(encoding="utf-8").splitlines()[0]
    assert header.split("\t") == expected


def test_read_header_tokens_ok_with_bom_and_comments(tmp_path: Path):
    # Build a file with BOM, blank lines and comments before the header
    p = tmp_path / "a.tsv"
    content = (
        "\ufeff\n"
        "# comment\n"
        "\n"
        "map_id\ttimestamp\tazimuth\televation\toffset_az\toffset_el\t"
        "temperature\tpressure\thumidity\n"
        "row\t2025-01-01T00:00:00Z\t0\t0\t0\t0\tNaN\tNaN\tNaN\n"
    )
    p.write_text(content, encoding="utf-8")
    tokens = io._read_header_tokens(p)
    assert tokens == io._expected_columns()


def test_read_header_tokens_no_header_raises(tmp_path: Path):
    p = tmp_path / "empty.tsv"
    p.write_text("# only comments\n# and blanks\n\n", encoding="utf-8")
    with pytest.raises(io.SchemaMismatchError):
        io._read_header_tokens(p)


def _make_demo_md():
    return io.Metadata(
        site_location="Mario Zucchelli Station, Antarctica",
        antenna_diameter_m=1.2,
        frequency_ghz=100,
        software_url="https://...",
        created_at_iso="2025-01-03T00:00:00Z",
    )


def _make_demo_rows(n=1, start=0):
    rows = []
    for i in range(start, start + n):
        rows.append(
            io.Measurement(
                map_id=f"ID{i}",
                timestamp_iso="2025-01-03T00:02:35Z",
                azimuth_deg=10.0 + i,
                elevation_deg=20.0 + i,
                offset_az_deg=0.1 * i,
                offset_el_deg=-0.2 * i,
                temperature_c=None,
                pressure_hpa=None,
                humidity_frac=None,
            )
        )
    return rows


def _read_data_lines(path: Path):
    lines = path.read_text(encoding="utf-8").splitlines()
    # Skip metadata comments and the header line
    data = [ln for ln in lines if ln and not ln.lstrip().startswith("#")]
    assert data, "file should contain at least a header line"
    return data[1:]  # drop header


def test_write_offsets_tsv_overwrite_and_append_ok(tmp_path: Path):
    p = tmp_path / "out.tsv"
    md = _make_demo_md()

    # Overwrite (append=False): creates file with metadata + header + rows
    io.write_offsets_tsv(p, md, _make_demo_rows(2), append=False)
    data = _read_data_lines(p)
    assert len(data) == 2

    # Append with matching header: appends more rows
    io.write_offsets_tsv(p, md, _make_demo_rows(3, start=2), append=True)
    data = _read_data_lines(p)
    assert len(data) == 5


def test_write_offsets_tsv_append_header_mismatch_raises(tmp_path: Path):
    p = tmp_path / "bad.tsv"
    md = _make_demo_md()
    # Create a file with a wrong header (e.g., missing humidity)
    bad_header = (
        "map_id\ttimestamp\tazimuth\televation\toffset_az\toffset_el\t"
        "temperature\tpressure\n"
    )
    p.write_text("# meta\n" + bad_header, encoding="utf-8")

    with pytest.raises(io.SchemaMismatchError):
        io.write_offsets_tsv(p, md, _make_demo_rows(1), append=True)


def test_write_offsets_tsv_append_on_file_without_header_falls_back_to_overwrite(
    tmp_path: Path,
):
    p = tmp_path / "noheader.tsv"
    md = _make_demo_md()
    p.write_text("# just comments\n# more\n", encoding="utf-8")

    io.write_offsets_tsv(p, md, _make_demo_rows(2), append=True)
    data = _read_data_lines(p)
    assert len(data) == 2  # overwritten with exactly the new rows


def test___main___demo_is_executed(monkeypatch, tmp_path: Path):
    # Run the module's demo by executing the file as __main__, in a temp cwd
    monkeypatch.chdir(tmp_path)
    # Locate the source file of the imported module
    src = Path(io.__file__)
    # Execute as a script to hit the __main__ block
    runpy.run_path(str(src), run_name="__main__")
    # A file named example.tsv should have been created
    out = tmp_path / "example.tsv"
    assert out.exists()
    # It should contain a header and at least one row
    lines = out.read_text(encoding="utf-8").splitlines()
    assert any(not ln.startswith("#") for ln in lines)


def test_overwrite_cleanup_unlink_ok(tmp_path: Path, monkeypatch):
    """
    1) pre-create the target file so Path.exists() on 'p' is True,
    2) force an exception before tmp.replace(p) by raising in _write_column_header,
    3) assert that the .tmp file is removed by the cleanup.
    """
    p = tmp_path / "demo.tsv"
    p.write_text("# pre-existing file\n", encoding="utf-8")  # ensure p.exists() is True
    tmp = p.with_suffix(p.suffix + ".tmp")

    # Make _write_column_header raise to bail out before tmp.replace(p)
    def boom(*a, **k):
        raise RuntimeError("force failure before replace")

    monkeypatch.setattr(io, "_write_column_header", boom)

    md = io.Metadata(
        site_location="X",
        antenna_diameter_m=1.0,
        frequency_ghz=100.0,
        software_url="https://...",
    )

    with pytest.raises(RuntimeError, match="force failure"):
        io.write_offsets_tsv(p, md, rows=[], append=False)

    # The .tmp file must not exist anymore (cleanup succeeded)
    assert not tmp.exists()


def test_overwrite_cleanup_unlink_oserror(tmp_path: Path, monkeypatch):
    """
    1) pre-create the target file so Path.exists() on 'p' is True,
    2) force an exception before tmp.replace(p) by raising in _write_column_header,
    3) monkeypatch Path.unlink to raise OSError for the specific .tmp path,
    4) assert that the except branch is exercised and .tmp remains on disk.
    """
    p = tmp_path / "demo2.tsv"
    p.write_text("# pre-existing file\n", encoding="utf-8")
    tmp = p.with_suffix(p.suffix + ".tmp")

    # Make _write_column_header raise to bail out before tmp.replace(p)
    def boom(*a, **k):
        raise RuntimeError("force failure before replace")

    monkeypatch.setattr(io, "_write_column_header", boom)

    # Monkeypatch Path.unlink to raise OSError only for our tmp path
    import pathlib

    real_unlink = pathlib.Path.unlink

    def fake_unlink(self, *a, **k):
        if self == tmp:
            raise OSError("simulated unlink failure")
        return real_unlink(self, *a, **k)

    monkeypatch.setattr(pathlib.Path, "unlink", fake_unlink)

    md = io.Metadata(
        site_location="Y",
        antenna_diameter_m=2.0,
        frequency_ghz=90.0,
        software_url="https://...",
    )

    with pytest.raises(RuntimeError, match="force failure"):
        io.write_offsets_tsv(p, md, rows=[], append=False)

    # Because unlink raised OSError, the except branch swallowed it
    # and the tmp still exists
    assert tmp.exists()


def test_write_metadata_block_full_coverage():
    from io import StringIO
    import solaris_pointing.offsets.io as io_mod

    md = io_mod.Metadata(
        # Site info
        site_location="Antarctica",
        site_code="MZS",
        data_code="OASI",
        site_lat=-74.7,
        site_lon=164.1,
        site_height=25.0,
        # Telescope
        antenna_diameter_m=1.2,
        frequency_ghz=100.0,
        # Algorithm
        algo="sun_maps",
        az_offset_bias=0.12,
        el_offset_bias=-0.34,
        refraction="enabled",
        pressure_hpa=990.0,
        temperature_c=-5.0,
        humidity_frac=0.45,
        obswl_mm=3.0,
        peak_frac=0.75,
        central_power_frac=0.60,
        # Software info
        software_url="https://repo",
        software_commit="abc123",
        # Run info
        config_file="/path/to/config.toml",
        created_at_iso="2025-02-01T12:34:56Z",
    )

    buf = StringIO()
    io_mod._write_metadata_block(buf, md)
    txt = buf.getvalue()
    lines = txt.splitlines()

    # Title line
    assert lines[0].startswith("# === Metadata")

    # SITE
    assert "# [Site]" in txt
    assert "#  Location: Antarctica" in txt
    assert "#  Code: MZS" in txt
    assert "#  Code from data: OASI" in txt
    assert "#  Latitude (deg): -74.7" in txt
    assert "#  Longitude (deg): 164.1" in txt
    assert "#  Height (m): 25.0" in txt

    # TELESCOPE
    assert "# [Telescope]" in txt
    assert "#   Diameter (m)    : 1.2" in txt
    assert "#   Frequency (GHz) : 100.0" in txt

    # ALGORITHM
    assert "# [Algorithm]" in txt
    assert "#  Name: sun_maps" in txt
    assert "#  AZ offset bias (deg): 0.12" in txt
    assert "#  EL offset bias (deg): -0.34" in txt
    assert "#  Refraction: enabled" in txt
    assert "#  Pressure (hPa): 990.0" in txt
    assert "#  Temperature (C): -5.0" in txt
    assert "#  Humidity (frac): 0.45" in txt
    assert "#  Wavelength (mm): 3.0" in txt
    assert "#  Peak fraction: 0.75" in txt
    assert "#  Central power frac: 0.6" in txt

    # SOFTWARE
    assert "# [Software]" in txt
    assert "#  Repository: https://repo" in txt
    assert "#  Commit SHA: abc123" in txt

    # RUN
    assert "# [Run]" in txt
    assert "#  Config file: /path/to/config.toml" in txt
    assert "#  Created at (UTC): 2025-02-01T12:34:56Z" in txt

    # Trailer
    assert lines[-1].startswith("# ==")
