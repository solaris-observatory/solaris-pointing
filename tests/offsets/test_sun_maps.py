# tests/offsets/test_sun_maps.py
from __future__ import annotations

import sys
import types
from pathlib import Path
from datetime import datetime, timezone, timedelta
from argparse import Namespace
import importlib
import builtins

import numpy as np
import pytest


# ---------------------------------------------------------------------
# Fakes: solaris_pointing.offsets.io
# ---------------------------------------------------------------------
def _install_fake_io(tmp: Path, written: list[tuple[str, list]]):
    # Create ONLY the target module to avoid shadowing real packages
    io = types.ModuleType("solaris_pointing.offsets.io")

    class Metadata:
        def __init__(self, **kw):  # pragma: no cover (init trivial)
            self.__dict__.update(kw)

    class Measurement:
        def __init__(self, **kw):  # pragma: no cover (init trivial)
            self.__dict__.update(kw)

    def write_offsets_tsv(out_fname: str, md, record, append=True):
        # Track writes for asserts and write a minimal file physically
        written.append((out_fname, record))
        p = Path(out_fname)
        p.parent.mkdir(parents=True, exist_ok=True)
        mode = "a" if append else "w"
        with open(p, mode, encoding="utf-8") as f:
            for r in record:
                f.write(f"{r.map_id}\t{r.timestamp_iso}\n")

    io.Metadata = Metadata
    io.Measurement = Measurement
    io.write_offsets_tsv = write_offsets_tsv

    # Register only the leaf module. Parents (solaris_pointing, offsets)
    # must remain the real installed packages.
    sys.modules["solaris_pointing.offsets.io"] = io


# ---------------------------------------------------------------------
# Fakes: astropy (Time, EarthLocation, AltAz, get_sun, units)
# ---------------------------------------------------------------------
def _install_fake_astropy():
    # Create proper modules; avoid SimpleNamespace in sys.modules
    class U:
        deg = "deg"
        m = "m"
        hPa = "hPa"
        mm = "mm"
        deg_C = "deg_C"

    u = U

    class FakeSun:
        def __init__(self, az, alt):
            self._az = az
            self._alt = alt

        def transform_to(self, altaz):
            return self  # already "transformed"

        @property
        def az(self):
            class V:
                def __init__(self, val):
                    self.value = val

                def to(self, unit):
                    return self

            return V(self._az)

        @property
        def alt(self):
            class V:
                def __init__(self, val):
                    self.value = val

                def to(self, unit):
                    return self

            return V(self._alt)

    class Time:
        def __init__(self, dt, scale="utc"):
            self.dt = dt
            self.scale = scale

    class EarthLocation:
        def __init__(self, lat, lon, height):
            self.lat, self.lon, self.height = lat, lon, height

    class AltAz:
        def __init__(self, **kw):
            self.kw = kw  # just to accept args

    # get_sun produce un sole con az/el “finti” dipendenti da secondi
    def get_sun(t: Time):
        s = int(t.dt.timestamp()) % 360
        az = float((s * 1.0) % 360)
        alt = float(((s // 2) % 90) - 10)
        return FakeSun(az, alt)

    astropy_mod = types.ModuleType("astropy")
    coords = types.ModuleType("astropy.coordinates")
    time_mod = types.ModuleType("astropy.time")
    units_mod = types.ModuleType("astropy.units")

    coords.EarthLocation = EarthLocation
    coords.AltAz = AltAz
    coords.get_sun = get_sun
    time_mod.Time = Time
    # expose units as attribute 'u' on the units module
    # expose both styles: module-level and 'u' namespace
    units_mod.deg = 1.0
    units_mod.m = 1.0
    units_mod.hPa = 1.0
    units_mod.mm = 1.0
    units_mod.u = u
    units_mod.deg_C = 1.0

    sys.modules["astropy"] = astropy_mod
    sys.modules["astropy.coordinates"] = coords
    sys.modules["astropy.time"] = time_mod
    sys.modules["astropy.units"] = units_mod


# ---------------------------------------------------------------------
# Utility: write tab-separated files with header the way reader expects
# ---------------------------------------------------------------------
def _write_path_file(p: Path, rows: list[tuple[str, float, float, int]]):
    # Header must be the very first line; parser uses f.seek(len(header))
    header = "UTC\tAzimuth\tElevation\tF0\n"
    with open(p, "w", encoding="utf-8", newline="") as f:
        f.write(header)
        for utc, az, el, f0 in rows:
            f.write(f"{utc}\t{az}\t{el}\t{f0}\n")


def _write_sky_file(p: Path, rows: list[tuple[str, float]]):
    header = "UTC\tSignal\n"
    with open(p, "w", encoding="utf-8", newline="") as f:
        f.write(header)
        for utc, sig in rows:
            f.write(f"{utc}\t{sig}\n")


# ---------------------------------------------------------------------
# Import target module with fakes installed
# ---------------------------------------------------------------------
@pytest.fixture()
def sun_maps(tmp_path, monkeypatch):
    # install fakes
    written = []
    _install_fake_io(tmp_path, written)
    _install_fake_astropy()
    # import target
    if "solaris_pointing.offsets.algos.sun_maps" in sys.modules:
        del sys.modules["solaris_pointing.offsets.algos.sun_maps"]
    module = importlib.import_module("solaris_pointing.offsets.algos.sun_maps")
    return module, written


# ---------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------
def test_parse_and_readers_and_segments_and_wrap_and_nearest(sun_maps, tmp_path):
    sm, _ = sun_maps

    # parse_iso_utc: naive vs with tz
    dt_naive = sm.parse_iso_utc("2025-01-01T00:00:00.000")
    assert dt_naive.tzinfo is not None and dt_naive.utcoffset() == timedelta(0)
    dt_tz = sm.parse_iso_utc("2025-01-01T00:00:00+03:00")
    assert dt_tz.utcoffset() == timedelta(0)

    # find_scan_segments: multiple patterns
    arr = np.array([0, 1, 1, 0, 1, 0, 1, 1], dtype=int)
    segs = sm.find_scan_segments(arr)
    assert segs == [(1, 3), (4, 5), (6, 8)]

    # wrap_az
    assert sm.wrap_az(361.5) == pytest.approx(1.5)

    # nearest_path_observed edges and middle
    t = np.array([10.0, 20.0, 30.0], dtype=float)
    az = np.array([5.0, 15.0, 25.0], dtype=float)
    el = np.array([50.0, 60.0, 70.0], dtype=float)
    args = Namespace()
    # left edge
    a0, e0 = sm.nearest_path_observed(5.0, t, az, el, args)
    assert (a0, e0) == (5.0 % 360, 50.0)
    # right edge
    a1, e1 = sm.nearest_path_observed(31.0, t, az, el, args)
    assert (a1, e1) == (25.0 % 360, 70.0)
    # middle nearer to 20
    a2, e2 = sm.nearest_path_observed(24.9, t, az, el, args)
    assert (a2, e2) == (15.0 % 360, 60.0)


def test_choose_scan_centroid_branching(sun_maps):
    sm, _ = sun_maps
    args = Namespace(site_location="Foo", peak_frac=0.75, central_power_frac=0.60)

    # No segments -> None
    t_path = np.array([0.0, 1.0, 2.0], dtype=float)
    f0 = np.array([0, 0, 0], dtype=int)
    assert (
        sm.choose_scan_and_centroid_time(t_path, f0, np.array([]), np.array([]), args)
        is None
    )

    # No sky -> None
    f0 = np.array([1, 1, 1], dtype=int)
    assert (
        sm.choose_scan_and_centroid_time(t_path, f0, np.array([]), np.array([]), args)
        is None
    )

    # Build two scans; first yields no keep, second yields one sample -> fallback
    t_path = np.array([0.0, 1.0, 2.0, 10.0, 11.0, 12.0], dtype=float)
    f0 = np.array([1, 1, 0, 1, 1, 0], dtype=int)
    # sky aligned to both windows
    sky_times = np.array([0.2, 0.6, 10.1], dtype=float)
    # first scan too small to pass threshold; second will pass with one sample
    sky_vals = np.array([1.0, 1.0, 100.0], dtype=float)
    out = sm.choose_scan_and_centroid_time(t_path, f0, sky_times, sky_vals, args)
    assert out is not None
    seg, tcent, cp = out
    assert seg == (3, 5)
    assert tcent == pytest.approx(10.1)
    assert cp == pytest.approx(100.0)

    # Case with duplicate timestamps (non-increasing dt) -> fallback to mean
    t_path = np.array([0.0, 5.0], dtype=float)
    f0 = np.array([1, 1], dtype=int)
    sky_times = np.array([0.0, 0.0, 0.0, 0.1], dtype=float)
    sky_vals = np.array([10.0, 10.0, 10.0, 10.0], dtype=float)
    out2 = sm.choose_scan_and_centroid_time(t_path, f0, sky_times, sky_vals, args)
    assert out2 is not None
    seg2, tcent2, cp2 = out2
    assert seg2 == (0, 2)
    assert tcent2 == pytest.approx(np.mean(sky_times))


def test_compute_ephem_on_off(sun_maps):
    sm, _ = sun_maps
    dt = datetime(2025, 1, 1, tzinfo=timezone.utc)
    # OFF
    args_off = Namespace(
        site_location="Antarctica",
        site_lat=0.0,
        site_lon=0.0,
        site_height=0.0,
        enable_refraction=False,
        pressure=990.0,
        temperature=-5.0,
        humidity=0.5,
        obswl=3.0,
    )
    az, el = sm.compute_ephem(dt, args_off)
    assert 0 <= az < 360 and -90 <= el <= 90
    # ON (branch with refraction params)
    args_on = Namespace(
        site_location="Foo",
        site_lat=1.0,
        site_lon=2.0,
        site_height=3.0,
        enable_refraction=True,
        pressure=990.0,
        temperature=-5.0,
        humidity=0.5,
        obswl=3.0,
    )
    az2, el2 = sm.compute_ephem(dt, args_on)
    assert 0 <= az2 < 360 and -90 <= el2 <= 90


def test_readers_and_process_and_append(sun_maps, tmp_path):
    sm, written = sun_maps

    # Build .path / .sky with:
    # - prima riga malformata (verrà saltata)
    # - scans: due segmenti contiguous F0==1, il secondo sarà scelto
    t0 = datetime(2025, 1, 1, 0, 0, 0, tzinfo=timezone.utc)

    def iso(dt):
        return dt.isoformat(timespec="milliseconds")

    path_rows = [
        ("malformed_row", 0, 0, 0),  # verrà scartata
        (iso(t0 + timedelta(seconds=0)), 10.0, 50.0, 1),
        (iso(t0 + timedelta(seconds=1)), 20.0, 60.0, 1),
        (iso(t0 + timedelta(seconds=2)), 30.0, 70.0, 0),
        (iso(t0 + timedelta(seconds=10)), 40.0, 80.0, 1),
        (iso(t0 + timedelta(seconds=11)), 50.0, 85.0, 1),
        (iso(t0 + timedelta(seconds=12)), 55.0, 86.0, 0),
    ]

    sky_rows = [
        ("garbage", 0.0),  # malformata, scartata
        (iso(t0 + timedelta(seconds=0.4)), 1.0),
        (iso(t0 + timedelta(seconds=0.6)), 1.0),
        (iso(t0 + timedelta(seconds=10.1)), 100.0),  # secondo scan
        (iso(t0 + timedelta(seconds=10.2)), 120.0),
    ]

    data = tmp_path / "data"
    data.mkdir()
    _write_path_file(data / "map.path", path_rows)
    _write_sky_file(data / "map.sky", sky_rows)

    args = Namespace(
        site_location="Antarctica",
        peak_frac=0.75,
        central_power_frac=0.60,
        az_offset_bias=0.10,
        el_offset_bias=-0.05,
        site_lat=0.0,
        site_lon=0.0,
        site_height=0.0,
        enable_refraction=False,
        pressure=990.0,
        temperature=-5.0,
        humidity=0.5,
        obswl=3.0,
    )

    row = sm.process_map("map", str(data / "map.path"), str(data / "map.sky"), args)
    assert row is not None
    map_id, ts_iso, az_obs, el_obs, daz, del_ = row
    assert map_id == "map"
    # append_result_tsv
    out = tmp_path / "out.tsv"
    sm.append_result_tsv(str(out), row)
    assert out.exists()
    assert written and written[0][0].endswith("out.tsv")


def test_readers_skip_bad_rows_more_cases(sun_maps, tmp_path):
    sm, _ = sun_maps
    d = tmp_path / "io"
    d.mkdir()

    bad_path = d / "bad.path"
    bad_path.write_text(
        "UTC\tAzimuth\tElevation\tF0\n"
        "totally_bad_line\n"
        "2025-01-01T00:00:00.000Z\tA\tB\tC\n"
        "2025-01-01T00:00:01.000Z\t10.0\t20.0\t1\n",
        encoding="utf-8",
    )

    bad_sky = d / "bad.sky"
    bad_sky.write_text(
        "UTC\tSignal\n"
        "garbage\n"
        "2025-01-01T00:00:00.000Z\tX\n"
        "2025-01-01T00:00:01.050Z\t11.0\n",
        encoding="utf-8",
    )

    path_rows = sm.read_path_file(str(bad_path))
    assert len(path_rows) == 1
    pr = path_rows[0]
    assert pr.f0 == 1
    assert pr.az_deg == 10.0
    assert pr.el_deg == 20.0

    sky_rows = sm.read_sky_file(str(bad_sky), Namespace())
    assert len(sky_rows) == 1
    sr = sky_rows[0]
    assert sr.signal == 11.0


def test_choose_scan_centroid_none_when_no_keep(sun_maps):
    sm, _ = sun_maps
    t_path = np.array([0.0, 1.0, 2.0, 10.0, 11.0], dtype=float)
    f0 = np.array([1, 1, 0, 1, 0], dtype=int)
    sky_times = np.array([0.2, 0.8, 10.4], dtype=float)
    sky_vals = np.array([0.1, 0.2, 0.05], dtype=float)
    args = Namespace(site_location="Foo", peak_frac=1.1, central_power_frac=0.95)
    out = sm.choose_scan_and_centroid_time(t_path, f0, sky_times, sky_vals, args)
    assert out is None


def test_process_map_returns_none_when_no_centroid(sun_maps, tmp_path):
    sm, written = sun_maps
    d = tmp_path / "no_centroid"
    d.mkdir()
    path = d / "m.path"
    sky = d / "m.sky"
    path.write_text(
        "UTC\tAzimuth\tElevation\tF0\n"
        "2025-01-01T00:00:00.000Z\t10.0\t50.0\t1\n"
        "2025-01-01T00:00:01.000Z\t20.0\t60.0\t1\n"
        "2025-01-01T00:00:02.000Z\t30.0\t70.0\t0\n",
        encoding="utf-8",
    )
    sky.write_text(
        "UTC\tSignal\n2025-01-01T00:00:00.500Z\t0.05\n2025-01-01T00:00:01.000Z\t0.06\n",
        encoding="utf-8",
    )
    args = Namespace(
        site_location="Antarctica",
        peak_frac=1.1,  # forza no-keep
        central_power_frac=0.95,
        az_offset_bias=0.0,
        el_offset_bias=0.0,
        site_lat=0.0,
        site_lon=0.0,
        site_height=0.0,
        enable_refraction=False,
        pressure=990.0,
        temperature=-5.0,
        humidity=0.5,
        obswl=3.0,
    )
    row = sm.process_map("m", str(path), str(sky), args)
    assert row is None
    assert written == []


def test_compute_ephem_import_error_branch(monkeypatch, sun_maps):
    sm, _ = sun_maps
    import builtins

    real_import = builtins.__import__

    def fake_import(name, *a, **kw):
        if name.startswith("astropy"):
            raise ImportError("simulated missing astropy")
        return real_import(name, *a, **kw)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    with pytest.raises(SystemExit) as ex:
        sm.compute_ephem(
            datetime(2025, 1, 1, tzinfo=timezone.utc),
            Namespace(
                site_location="Antarctica",
                site_lat=0.0,
                site_lon=0.0,
                site_height=0.0,
                enable_refraction=False,
                pressure=990.0,
                temperature=-5.0,
                humidity=0.5,
                obswl=3.0,
            ),
        )
    assert "requires Astropy" in str(ex.value)


def test_read_path_file_skips_empty_rows(sun_maps, tmp_path):
    """Ensure read_path_file skips completely empty data rows."""
    sm, _ = sun_maps
    p = tmp_path / "path_with_empty.path"
    p.write_text(
        "UTC\tAzimuth\tElevation\tF0\n\n2025-01-01T00:00:00.000Z\t10.0\t20.0\t1\n",
        encoding="utf-8",
    )

    rows = sm.read_path_file(str(p))
    assert len(rows) == 1
    row = rows[0]
    assert row.az_deg == 10.0
    assert row.el_deg == 20.0
    assert row.f0 == 1


def test_read_sky_file_header_none_and_empty_rows(sun_maps, tmp_path):
    """Cover header == None branch and empty row branch in read_sky_file."""
    sm, _ = sun_maps
    d = tmp_path / "sky_cases"
    d.mkdir()

    # Case 1: completely empty file -> header is None -> early return
    empty = d / "empty.sky"
    empty.write_text("", encoding="utf-8")
    rows_empty = sm.read_sky_file(str(empty), Namespace())
    assert rows_empty == []

    # Case 2: normal header with a completely empty data row
    with_empty = d / "with_empty.sky"
    with_empty.write_text(
        "UTC\tSignal\n\n2025-01-01T00:00:00.000Z\t5.0\n",
        encoding="utf-8",
    )
    rows = sm.read_sky_file(str(with_empty), Namespace())
    assert len(rows) == 1
    assert rows[0].signal == 5.0


def test_read_sky_file_missing_utc_column_raises(sun_maps, tmp_path):
    """Ensure read_sky_file raises when the mandatory UTC column is missing."""
    sm, _ = sun_maps
    p = tmp_path / "missing_utc.sky"
    p.write_text(
        "Time\tSignal\n2025-01-01T00:00:00.000Z\t1.0\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError) as ex:
        sm.read_sky_file(str(p), Namespace())
    assert "Missing required column 'UTC'" in str(ex.value)


def test_read_sky_file_default_falls_back_to_signal0(sun_maps, tmp_path):
    """If 'Signal' is absent, the reader must fall back to 'Signal0'."""
    sm, _ = sun_maps
    p = tmp_path / "signal0_only.sky"
    p.write_text(
        "UTC\tSignal0\n2025-01-01T00:00:00.000Z\t42.0\n",
        encoding="utf-8",
    )

    rows = sm.read_sky_file(str(p), Namespace())
    assert len(rows) == 1
    assert rows[0].signal == 42.0


def test_read_sky_file_selects_requested_signal_column(sun_maps, tmp_path):
    """If args.signal is set, the reader must use the corresponding SignalN column."""
    sm, _ = sun_maps
    p = tmp_path / "multi_signal.sky"
    # Mix whitespace to exercise robust header parsing (spaces, not only tabs).
    p.write_text(
        "UTC   Signal0   Signal1\n2025-01-01T00:00:00.000Z   1.0   99.0\n",
        encoding="utf-8",
    )

    rows = sm.read_sky_file(str(p), Namespace(signal=1))
    assert len(rows) == 1
    assert rows[0].signal == 99.0


def test_read_sky_file_requested_signal_missing_raises(sun_maps, tmp_path):
    """If args.signal requests a non-existent SignalN column, raise with
    available Signal* columns."""
    sm, _ = sun_maps
    p = tmp_path / "only_signal0.sky"
    p.write_text(
        "UTC\tSignal0\n2025-01-01T00:00:00.000Z\t5.0\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError) as ex:
        sm.read_sky_file(str(p), Namespace(signal=2))
    msg = str(ex.value)
    assert "Requested Signal2" in msg
    assert "Available" in msg
    assert "Signal0" in msg


def test_read_sky_file_no_usable_signal_column_raises(sun_maps, tmp_path):
    """If no signal column is found, raise a helpful error."""
    sm, _ = sun_maps
    p = tmp_path / "no_signal_column.sky"
    p.write_text(
        "UTC\tPower\n2025-01-01T00:00:00.000Z\t123.0\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError) as ex:
        sm.read_sky_file(str(p), Namespace())
    msg = str(ex.value)
    assert "No usable signal column" in msg


def test_choose_scan_centroid_denominator_zero_fallback(sun_maps):
    """
    Force the defensive branch where denom <= 0 and the centroid falls back
    to the simple mean of timestamps.
    """
    sm, _ = sun_maps

    # Single scan spanning [0, 3]
    t_path = np.array([0.0, 3.0], dtype=float)
    f0 = np.array([1, 1], dtype=int)

    # Four samples with symmetric positive/negative values so that
    # the time-weighted sum of mid_s * dt is zero.
    sky_times = np.array([0.0, 1.0, 2.0, 3.0], dtype=float)
    sky_vals = np.array([-3.0, -1.0, 1.0, 3.0], dtype=float)

    # Negative peak_frac ensures all samples are kept, so that denom == 0
    args = Namespace(site_location="Foo", peak_frac=-2.0, central_power_frac=0.0)

    out = sm.choose_scan_and_centroid_time(t_path, f0, sky_times, sky_vals, args)
    assert out is not None
    seg, t_centroid, _cp = out
    assert seg == (0, 2)
    # Fallback centroid should be the mean of all timestamps
    assert t_centroid == pytest.approx(np.mean(sky_times))


def test_choose_scan_centroid_none_after_power_filter(sun_maps):
    """
    Ensure we hit the branch where scans are found but all are rejected
    by the central_power_frac threshold (kept list is empty).
    """
    sm, _ = sun_maps
    t_path = np.array([0.0, 1.0], dtype=float)
    f0 = np.array([1, 1], dtype=int)
    sky_times = np.array([0.0, 0.5, 1.0], dtype=float)
    sky_vals = np.array([1.0, 1.0, 1.0], dtype=float)

    # peak_frac = 0 keeps all samples; central_power_frac > 1 forces rejection
    args = Namespace(site_location="Foo", peak_frac=0.0, central_power_frac=2.0)

    out = sm.choose_scan_and_centroid_time(t_path, f0, sky_times, sky_vals, args)
    assert out is None


def test_choose_scan_centroid_continues_on_degenerate_segment(monkeypatch, sun_maps):
    """
    Monkeypatch find_scan_segments to return a degenerate (b <= a) segment
    and ensure the loop continues without using it.
    """
    sm, _ = sun_maps

    def fake_find_scan_segments(f0_array):
        # A single degenerate segment that should trigger the 'continue'.
        return [(1, 1)]

    monkeypatch.setattr(sm, "find_scan_segments", fake_find_scan_segments)

    t_path = np.array([0.0, 1.0], dtype=float)
    f0 = np.array([1, 1], dtype=int)
    sky_times = np.array([0.0, 0.5, 1.0], dtype=float)
    sky_vals = np.array([10.0, 20.0, 30.0], dtype=float)
    args = Namespace(site_location="Foo", peak_frac=0.75, central_power_frac=0.60)

    out = sm.choose_scan_and_centroid_time(t_path, f0, sky_times, sky_vals, args)
    # No valid segments remain, so the function returns None
    assert out is None


def test_choose_scan_centroid_continues_when_sv_slice_empty(sun_maps):
    """
    Use a fake sky_vals object so that the slice has size == 0 and the
    branch `if sv.size == 0: continue` is taken.
    """
    sm, _ = sun_maps

    t_path = np.array([0.0, 2.0], dtype=float)
    f0 = np.array([1, 1], dtype=int)
    sky_times = np.array([0.5, 1.5], dtype=float)

    class SkyValsFake:
        def __getitem__(self, _slice):
            class Slice:
                # Only 'size' is needed for this branch
                size = 0

            return Slice()

    sky_vals = SkyValsFake()
    args = Namespace(site_location="Foo", peak_frac=0.75, central_power_frac=0.60)

    out = sm.choose_scan_and_centroid_time(t_path, f0, sky_times, sky_vals, args)
    assert out is None


def test_choose_scan_centroid_skips_nonpositive_scan_max(sun_maps):
    """
    Ensure the branch `if scan_max <= 0: continue` is executed by providing
    non-positive sky values within a valid time window.
    """
    sm, _ = sun_maps
    t_path = np.array([0.0, 2.0], dtype=float)
    f0 = np.array([1, 1], dtype=int)
    sky_times = np.array([0.5, 1.5], dtype=float)
    # All zeros -> percentile is 0.0 -> scan_max <= 0
    sky_vals = np.array([0.0, 0.0], dtype=float)
    args = Namespace(site_location="Foo", peak_frac=0.75, central_power_frac=0.60)

    out = sm.choose_scan_and_centroid_time(t_path, f0, sky_times, sky_vals, args)
    assert out is None


def test_process_map_returns_none_when_no_rows(sun_maps, tmp_path):
    """
    Cover the early return in process_map when either path_rows or sky_rows
    is empty (here: empty .path file).
    """
    sm, written = sun_maps
    d = tmp_path / "no_rows"
    d.mkdir()

    empty_path = d / "m.path"
    empty_path.write_text("", encoding="utf-8")

    sky = d / "m.sky"
    sky.write_text(
        "UTC\tSignal\n2025-01-01T00:00:00.000Z\t1.0\n",
        encoding="utf-8",
    )

    args = Namespace(
        site_location="Antarctica",
        peak_frac=0.75,
        central_power_frac=0.60,
        az_offset_bias=0.0,
        el_offset_bias=0.0,
        site_lat=0.0,
        site_lon=0.0,
        site_height=0.0,
        enable_refraction=False,
        pressure=990.0,
        temperature=-5.0,
        humidity=0.5,
        obswl=3.0,
    )

    row = sm.process_map("m", str(empty_path), str(sky), args)
    assert row is None
    assert written == []
