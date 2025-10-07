
import os
import sys
import subprocess
from pathlib import Path

def write_minimal_data(tmpdir, map_id="250106T010421", suffix="OASI"):
    p = tmpdir / f"{map_id}_{suffix}.path"
    s = tmpdir / f"{map_id}_{suffix}.sky"
    p.write_text("s\tms\tX\taz\tel\tX\tX\tflag\n0\t0\t0\t10\t20\t0\t0\t1\n")
    s.write_text("s\tms\tX\tpower\n0\t0\t0\t40000\n")
    return p, s

def test_runner_with_dummy_algo(tmp_path):
    # Layout
    project_root = Path(__file__).resolve().parents[2]
    scripts = project_root / "scripts" / "pointing_offsets_runner.py"
    data_dir = tmp_path / "data"
    data_dir.mkdir()

    # Data files
    write_minimal_data(data_dir, "250106T010421")
    env_csv = data_dir / "environment.csv"
    env_csv.write_text("map_id,temperature_c,pressure_hpa,humidity_frac\n"
                       "250106T010421,-28.5,690.2,0.55\n")

    out_tsv = tmp_path / "offsets.tsv"

    # Ensure package is importable: add src/ to PYTHONPATH
    src_dir = project_root / "src"
    env = os.environ.copy()
    env["PYTHONPATH"] = os.pathsep.join(filter(None, [env.get("PYTHONPATH",""), str(src_dir), str(project_root)]))

    # Run with dummy algorithm (zeros)
    cmd = [sys.executable, str(scripts),
           "--data-dir", str(data_dir),
           "--out", str(out_tsv),
           "--algo", "algo_dummy",
           "--verbose"]
    proc = subprocess.run(cmd, env=env, capture_output=True, text=True)
    assert proc.returncode == 0, proc.stderr

    # File created and contains at least one data row
    assert out_tsv.exists()
    txt = out_tsv.read_text()
    assert "timestamp_iso" in txt.splitlines()[0].lower()  # header exists
    assert "2025-01-06T01:04:21Z" in txt
