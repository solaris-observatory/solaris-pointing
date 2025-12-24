# tests/scripts/test_generate_offsets.py
"""
End-to-end and error-path tests for `generate_offsets.py`.

Covers:
- CLI parsing and subprocess execution;
- discovery of .path/.sky pairs, exclusion of '...T<HHMMSS>b...' stems;
- inclusive date filters: --date-start / --date-end;
- happy-path TSV append and progress prints;
- error branches: missing module, missing required callables, data not found,
  and no valid pairs under --data.

All tests run in a temporary working directory and set PYTHONPATH to include a
temporary `src/` that hosts a fake algorithm module.
"""

from __future__ import annotations

import os
import sys
import subprocess
from pathlib import Path

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _python_exe() -> str:
    return sys.executable


def _script_path() -> Path:
    # Assuming this test lives in tests/scripts/, go two levels up to repo root
    return Path(__file__).resolve().parents[2] / "scripts" / "generate_offsets.py"


def _write_fake_algo(root: Path, missing: str | None = None) -> None:
    """
    Create a minimal fake algorithm module at:
      root/src/solaris_pointing/offsets/algos/sun_maps.py

    - process_map(map_id, path_fname, sky_fname, params) returns a dict with id,
      or None to simulate a skip if "SKIP" is in map_id.
    - append_result_tsv(out_path, res, params) appends one line per result.
    - If `missing` is "append", omit append_result_tsv to trigger the error.
    """
    pkg = root / "src" / "solaris_pointing" / "offsets" / "algos"
    pkg.mkdir(parents=True, exist_ok=True)
    (root / "src" / "solaris_pointing" / "__init__.py").write_text("", encoding="utf-8")
    (root / "src" / "solaris_pointing" / "offsets" / "__init__.py").write_text(
        "", encoding="utf-8"
    )
    (
        root / "src" / "solaris_pointing" / "offsets" / "algos" / "__init__.py"
    ).write_text("", encoding="utf-8")

    code = [
        "def process_map(map_id, path_fname, sky_fname, params):",
        "    if 'SKIP' in map_id:",
        "        return None",
        "    return {'id': map_id, 'path': str(path_fname), 'sky': str(sky_fname)}",
        "",
    ]
    if missing != "append":
        code += [
            "def append_result_tsv(out_path, res, params=None):",
            "    with open(out_path, 'a', encoding='utf-8') as f:",
            "        f.write(res['id'] + '\\n')",
            "",
        ]
    (pkg / "sun_maps.py").write_text("\n".join(code), encoding="utf-8")


def _make_scan(root: Path, stem: str) -> None:
    """
    Create a pair <stem>.path and <stem>.sky with trivial content.
    """
    d = root / "scans"
    d.mkdir(parents=True, exist_ok=True)
    for ext in (".path", ".sky"):
        (d / f"{stem}{ext}").write_text(f"{stem}{ext}\n", encoding="utf-8")


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_happy_path_and_progress_and_filters(tmp_path: Path, monkeypatch):
    # Arrange: fake algorithm and input scans
    _write_fake_algo(tmp_path)

    # Dates (YYMMDD...): 250101, 250102, 250103
    _make_scan(tmp_path, "250101T195415_OASI")
    _make_scan(tmp_path, "250101T195415bOASI")  # must be excluded
    _make_scan(tmp_path, "250102T001125_OASI_SKIP")  # will be processed -> None
    _make_scan(tmp_path, "250103T001125_OASI")

    env = os.environ.copy()
    env["PYTHONPATH"] = str(tmp_path / "src")

    # 1) Default run: should process all valid (excluding 'b'), write TSV,
    #    print [OK] lines for the two non-skip stems, and one [WARN] for SKIP.
    cp = subprocess.run(
        [
            _python_exe(),
            str(_script_path()),
            "--data",
            str(tmp_path / "scans"),
            "--algo",
            "sun_maps",
            "--outdir",
            str(tmp_path / "out_default"),
        ],
        cwd=tmp_path,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        check=False,
    )
    assert cp.returncode == 0, f"STDERR:\n{cp.stderr}"
    out_tsv = tmp_path / "out_default" / "sun_maps.tsv"
    assert out_tsv.exists(), "TSV should be created"
    lines = out_tsv.read_text(encoding="utf-8").strip().splitlines()
    # Two non-skip valid stems: 250101..., 250103...
    assert lines == ["250101T195415_OASI", "250103T001125_OASI"]

    # stdout expectations (don't over-constrain the exact format)
    assert "[OK]" in cp.stdout
    assert "-> appended to" in cp.stdout
    assert "[WARN] Skipping 250102T001125_OASI_SKIP" in cp.stdout

    # 2) Filter: --date-start 2025-01-02 should include only 250103... and SKIP
    cp2 = subprocess.run(
        [
            _python_exe(),
            str(_script_path()),
            "--data",
            str(tmp_path / "scans"),
            "--algo",
            "sun_maps",
            "--outdir",
            str(tmp_path / "out_start"),
            "--date-start",
            "2025-01-02",
        ],
        cwd=tmp_path,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        check=False,
    )
    assert cp2.returncode == 0, f"STDERR:\n{cp2.stderr}"
    out_tsv2 = tmp_path / "out_start" / "sun_maps.tsv"
    assert out_tsv2.exists()
    lines2 = out_tsv2.read_text(encoding="utf-8").strip().splitlines()
    assert lines2 == ["250103T001125_OASI"]

    # 3) Filter: --date-end 2025-01-01 should include only 250101...
    cp3 = subprocess.run(
        [
            _python_exe(),
            str(_script_path()),
            "--data",
            str(tmp_path / "scans"),
            "--algo",
            "sun_maps",
            "--outdir",
            str(tmp_path / "out_end"),
            "--date-end",
            "2025-01-01",
        ],
        cwd=tmp_path,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        check=False,
    )
    assert cp3.returncode == 0, f"STDERR:\n{cp3.stderr}"
    out_tsv3 = tmp_path / "out_end" / "sun_maps.tsv"
    assert out_tsv3.exists()
    lines3 = out_tsv3.read_text(encoding="utf-8").strip().splitlines()
    assert lines3 == ["250101T195415_OASI"]

    # 4) Filter: both bounds inclusive -> only 250101.. and 250102..(SKIP) in range,
    #    but SKIP does not append; so TSV must contain only 250101..
    cp4 = subprocess.run(
        [
            _python_exe(),
            str(_script_path()),
            "--data",
            str(tmp_path / "scans"),
            "--algo",
            "sun_maps",
            "--outdir",
            str(tmp_path / "out_range"),
            "--date-start",
            "2025-01-01",
            "--date-end",
            "2025-01-02",
        ],
        cwd=tmp_path,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        check=False,
    )
    assert cp4.returncode == 0, f"STDERR:\n{cp4.stderr}"
    out_tsv4 = tmp_path / "out_range" / "sun_maps.tsv"
    assert out_tsv4.exists()
    lines4 = out_tsv4.read_text(encoding="utf-8").strip().splitlines()
    assert lines4 == ["250101T195415_OASI"]


def test_missing_module(tmp_path: Path):
    # No fake algo written -> import should fail
    env = os.environ.copy()
    env["PYTHONPATH"] = str(tmp_path / "src")
    scans = tmp_path / "scans"
    scans.mkdir(parents=True, exist_ok=True)

    cp = subprocess.run(
        [
            _python_exe(),
            str(_script_path()),
            "--data",
            str(scans),
            "--algo",
            "does_not_exist",
        ],
        cwd=tmp_path,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        check=False,
    )
    assert cp.returncode != 0
    assert "Could not import" in (cp.stderr + cp.stdout)


def test_missing_append_callable(tmp_path: Path):
    # Algo without append_result_tsv should trigger the error path
    _write_fake_algo(tmp_path, missing="append")
    env = os.environ.copy()
    env["PYTHONPATH"] = str(tmp_path / "src")
    scans = tmp_path / "scans"
    scans.mkdir(parents=True, exist_ok=True)

    # Put at least one valid pair to reach the getattr check
    _make_scan(tmp_path, "250101T195415_OASI")

    cp = subprocess.run(
        [
            _python_exe(),
            str(_script_path()),
            "--data",
            str(tmp_path / "scans"),
            "--algo",
            "sun_maps",
        ],
        cwd=tmp_path,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        check=False,
    )
    assert cp.returncode != 0
    assert "does not expose a callable append_result_tsv" in (cp.stderr + cp.stdout)


def test_data_path_not_found(tmp_path: Path):
    _write_fake_algo(tmp_path)
    env = os.environ.copy()
    env["PYTHONPATH"] = str(tmp_path / "src")

    missing_dir = tmp_path / "nope"

    cp = subprocess.run(
        [
            _python_exe(),
            str(_script_path()),
            "--data",
            str(missing_dir),
            "--algo",
            "sun_maps",
        ],
        cwd=tmp_path,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        check=False,
    )
    # The script raises FileNotFoundError (uncaught): non-zero exit expected
    assert cp.returncode != 0
    assert "--data path not found" in (cp.stderr + cp.stdout)


def test_no_valid_pairs_message(tmp_path: Path):
    _write_fake_algo(tmp_path)
    env = os.environ.copy()
    env["PYTHONPATH"] = str(tmp_path / "src")

    scans = tmp_path / "scans"
    scans.mkdir(parents=True, exist_ok=True)
    # Create only unmatched files to force "No valid ... pairs"
    (scans / "250101T195415_OASI.path").write_text("x\n", encoding="utf-8")

    cp = subprocess.run(
        [
            _python_exe(),
            str(_script_path()),
            "--data",
            str(scans),
            "--algo",
            "sun_maps",
        ],
        cwd=tmp_path,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        check=False,
    )
    # Should exit cleanly and print the "No valid ..." message
    assert cp.returncode == 0, f"STDERR:\n{cp.stderr}"
    assert "No valid <map_id>.path / <map_id>.sky pairs found." in cp.stdout


def test_examples_option_outputs_docstring_section(tmp_path: Path, monkeypatch):
    """
    Verify that `--examples` prints the 'Command-line usage examples' section
    extracted from the module docstring, including the title and at least part
    of the example body. Also ensure no other required arguments (like --algo)
    are needed when --examples is supplied.
    """
    # Arrange: create fake algorithm so PYTHONPATH is valid
    _write_fake_algo(tmp_path)

    env = os.environ.copy()
    env["PYTHONPATH"] = str(tmp_path / "src")

    # Act: run the script with only --examples
    cp = subprocess.run(
        [
            _python_exe(),
            str(_script_path()),
            "--examples",
        ],
        cwd=tmp_path,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        check=False,
    )

    # Assert: clean exit
    assert cp.returncode == 0, f"STDERR:\n{cp.stderr}"

    out = cp.stdout

    title = "Command-line usage examples"

    # 1) Output must contain the title
    assert title in out, "Output must include the examples title"

    # 2) Output must contain at least one example command
    assert "python" in out, "Output should contain example commands"

    # 3) Output must include a separator of >= 10 hyphens
    assert any(("-" * 10) in line for line in out.splitlines()), (
        "Output should contain a hyphen delimiter"
    )

    # 4) Output should contain at least one CLI switch
    assert "--data" in out or "--algo" in out, (
        "Example body should include CLI switches"
    )


def _write_fake_algo_with_path_basename(root: Path) -> None:
    """
    Fake algorithm that writes both map_id and the basename of the chosen .path.
    This lets us verify the discovery picks <base>_offset.path when two candidates
    exist.
    """
    pkg = root / "src" / "solaris_pointing" / "offsets" / "algos"
    pkg.mkdir(parents=True, exist_ok=True)
    (root / "src" / "solaris_pointing" / "__init__.py").write_text("", encoding="utf-8")
    (root / "src" / "solaris_pointing" / "offsets" / "__init__.py").write_text(
        "", encoding="utf-8"
    )
    (
        root / "src" / "solaris_pointing" / "offsets" / "algos" / "__init__.py"
    ).write_text("", encoding="utf-8")

    code = [
        "from pathlib import Path",
        "",
        "def process_map(map_id, path_fname, sky_fname, params):",
        "    return {'id': map_id, 'path': str(path_fname), 'sky': str(sky_fname)}",
        "",
        "def append_result_tsv(out_path, res, params=None):",
        "    with open(out_path, 'a', encoding='utf-8') as f:",
        "        f.write(res['id'] + '\\t' + Path(res['path']).name + '\\n')",
        "",
    ]
    (pkg / "sun_maps.py").write_text("\n".join(code), encoding="utf-8")


def test_two_path_candidates_choose_offset(tmp_path: Path):
    _write_fake_algo_with_path_basename(tmp_path)

    env = os.environ.copy()
    env["PYTHONPATH"] = str(tmp_path / "src")

    scans = tmp_path / "scans"
    a = scans / "A"
    b = scans / "B"
    a.mkdir(parents=True, exist_ok=True)
    b.mkdir(parents=True, exist_ok=True)

    base = "251219T040038_ROSA"

    # One .sky
    (a / f"{base}.sky").write_text("x\n", encoding="utf-8")

    # Two candidate .path files, possibly in different subdirectories
    (a / f"{base}.path").write_text("x\n", encoding="utf-8")
    (b / f"{base}_offset.path").write_text("x\n", encoding="utf-8")

    cp = subprocess.run(
        [
            _python_exe(),
            str(_script_path()),
            "--data",
            str(scans),
            "--algo",
            "sun_maps",
            "--outdir",
            str(tmp_path / "out_offset_pick"),
        ],
        cwd=tmp_path,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        check=False,
    )
    assert cp.returncode == 0, f"STDERR:\n{cp.stderr}"

    out_tsv = tmp_path / "out_offset_pick" / "sun_maps.tsv"
    line = out_tsv.read_text(encoding="utf-8").strip()
    map_id, chosen_path_basename = line.split("\t")
    assert map_id == base
    assert chosen_path_basename == f"{base}_offset.path"


def test_sky_without_any_matching_path_is_error(tmp_path: Path):
    _write_fake_algo(tmp_path)
    env = os.environ.copy()
    env["PYTHONPATH"] = str(tmp_path / "src")

    scans = tmp_path / "scans"
    scans.mkdir(parents=True, exist_ok=True)

    base = "251219T040038_ROSA"
    (scans / f"{base}.sky").write_text("x\n", encoding="utf-8")

    cp = subprocess.run(
        [
            _python_exe(),
            str(_script_path()),
            "--data",
            str(scans),
            "--algo",
            "sun_maps",
        ],
        cwd=tmp_path,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        check=False,
    )
    assert cp.returncode != 0
    assert "No .path file found for .sky base" in (cp.stderr + cp.stdout)


def test_two_candidates_missing_exact_offset_is_error(tmp_path: Path):
    _write_fake_algo(tmp_path)
    env = os.environ.copy()
    env["PYTHONPATH"] = str(tmp_path / "src")

    scans = tmp_path / "scans"
    d1 = scans / "D1"
    d2 = scans / "D2"
    d1.mkdir(parents=True, exist_ok=True)
    d2.mkdir(parents=True, exist_ok=True)

    base = "251219T040038_ROSA"
    (d1 / f"{base}.sky").write_text("x\n", encoding="utf-8")

    # Two candidates that start with base, but no exact "<base>_offset.path"
    (d1 / f"{base}.path").write_text("x\n", encoding="utf-8")
    (d2 / f"{base}_extra.path").write_text("x\n", encoding="utf-8")

    cp = subprocess.run(
        [
            _python_exe(),
            str(_script_path()),
            "--data",
            str(scans),
            "--algo",
            "sun_maps",
        ],
        cwd=tmp_path,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        check=False,
    )
    assert cp.returncode != 0
    out = cp.stderr + cp.stdout
    assert "Ambiguous .path candidates (2 found)" in out
    assert f"expected: {base}_offset.path" in out


def test_more_than_two_candidates_is_error(tmp_path: Path):
    _write_fake_algo(tmp_path)
    env = os.environ.copy()
    env["PYTHONPATH"] = str(tmp_path / "src")

    scans = tmp_path / "scans"
    d1 = scans / "D1"
    d2 = scans / "D2"
    d3 = scans / "D3"
    d1.mkdir(parents=True, exist_ok=True)
    d2.mkdir(parents=True, exist_ok=True)
    d3.mkdir(parents=True, exist_ok=True)

    base = "251219T040038_ROSA"
    (d1 / f"{base}.sky").write_text("x\n", encoding="utf-8")

    # Three candidates that start with base
    (d1 / f"{base}.path").write_text("x\n", encoding="utf-8")
    (d2 / f"{base}_offset.path").write_text("x\n", encoding="utf-8")
    (d3 / f"{base}_whatever.path").write_text("x\n", encoding="utf-8")

    cp = subprocess.run(
        [
            _python_exe(),
            str(_script_path()),
            "--data",
            str(scans),
            "--algo",
            "sun_maps",
        ],
        cwd=tmp_path,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        check=False,
    )
    assert cp.returncode != 0
    assert "Ambiguous .path candidates (>2 found)" in (cp.stderr + cp.stdout)
