
from datetime import datetime, timezone
import os

from solaris_pointing.offset_core.discovery import discover_maps, parse_map_id_timestamp

def _write_minimal_files(tmpdir, map_id="250106T010421", suffix="OASI"):
    p = tmpdir / f"{map_id}_{suffix}.path"
    s = tmpdir / f"{map_id}_{suffix}.sky"
    # minimal headers + one row
    p.write_text("s\tms\tX\taz\tel\tX\tX\tflag\n0\t0\t0\t10\t20\t0\t0\t1\n")
    s.write_text("s\tms\tX\tpower\n0\t0\t0\t40000\n")
    return str(p), str(s)

def test_parse_map_id_timestamp():
    dt = parse_map_id_timestamp("250106T010421")
    assert dt == datetime(2025,1,6,1,4,21, tzinfo=timezone.utc)

def test_discover_single_pair(tmp_path):
    _write_minimal_files(tmp_path)
    maps = discover_maps(str(tmp_path), None, None)
    assert len(maps) == 1
    assert maps[0].map_id == "250106T010421"

def test_discover_time_window(tmp_path):
    _write_minimal_files(tmp_path, map_id="250106T010421")
    _write_minimal_files(tmp_path, map_id="250106T020000")
    maps = discover_maps(str(tmp_path), "2025-01-06T01:30:00Z", None)
    assert len(maps) == 1
    assert maps[0].map_id == "250106T020000"
