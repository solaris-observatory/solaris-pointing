#!/usr/bin/env python3
"""
Discover Sun scan pairs (.path/.sky), compute pointing offsets, write a TSV.

This CLI is a thin driver around algorithms in
`solaris_pointing.offsets.algos.<algo>`. It discovers valid map pairs
(recursive under --data), imports the selected algorithm, and calls its
public API to compute and append results to a single TSV file.

The default behavior (no date filters) matches current `sun_maps`. You can
optionally restrict processing to a date range (inclusive) parsed from each
map stem (format `YYMMDDTHHMMSS...`, years 2000–2099).

-------------------------------------------------------------------------------
Key features
-------------------------------------------------------------------------------
- Recursively discover <stem>.path and <stem>.sky under --data.
- Exclude stems that contain 'T<HHMMSS>b' (e.g., 250101T210109bOASI).
- Optional inclusive date filters: --date-start / --date-end.
- Pass-through site/refraction params for AltAz construction (when enabled).
- Append all results to a single TSV named <algo>.tsv under --outdir.
- Progress line per map: `[OK] MM/N: <map_id> -> appended to <algo>.tsv`.

-------------------------------------------------------------------------------
Input / output conventions
-------------------------------------------------------------------------------
- Input directory: provided via --data (searched recursively).
- Valid pair: both `<stem>.path` and `<stem>.sky` exist and do not match
  the exclusion rule `T\\d{6}b`.
- Output TSV path: `<outdir>/<algo>.tsv` (directory is created if missing).

-------------------------------------------------------------------------------
Command-line usage examples
-------------------------------------------------------------------------------
1) Minimal run (default behavior, no date filter):
   python scripts/generate_offsets.py --data scans/ --algo sun_maps

2) Only maps on/after a date (inclusive):
   python scripts/generate_offsets.py --data scans/ --algo sun_maps \
       --date-start 2025-01-01

3) Only maps up to a date (inclusive):
   python scripts/generate_offsets.py --data scans/ --algo sun_maps \
       --date-end 2025-01-02

4) Maps between two dates (inclusive):
   python scripts/generate_offsets.py --data scans/ --algo sun_maps \
       --date-start 2025-01-01 --date-end 2025-01-03

5) Enable refraction with site meteo (used only with --enable-refraction):
   python scripts/generate_offsets.py --data scans/ --algo sun_maps \
       --enable-refraction --pressure 990 --temperature -5 \
       --humidity 0.5 --obswl 3.0

6) Apply fixed biases to computed offsets (degrees):
   python scripts/generate_offsets.py --data scans/ --algo sun_maps \
       --az-offset-bias 0.10 --el-offset-bias -0.05

7) Write results under a custom output directory:
   python scripts/generate_offsets.py --data scans/ --algo sun_maps \
       --outdir offsets_run_42

-------------------------------------------------------------------------------
Parameters (selected)
-------------------------------------------------------------------------------
--algo (str, required)        Algorithm module under
                              `solaris_pointing.offsets.algos`, e.g. sun_maps.
--data (str)                  Root directory for discovery (recursive).
--outdir (str)                Directory for the output TSV (default: ./offsets).

--date-start (YYYY-MM-DD)     Inclusive start date. If provided alone, include
                              all maps on/after this date.
--date-end   (YYYY-MM-DD)     Inclusive end date. If provided alone, include
                              all maps on/before this date.
                              If both are provided: keep maps in [start, end].

--site-lon/--site-lat         Observatory coordinates (degrees).
--site-height (m)             Observatory height above sea level (meters).
--enable-refraction           Enable AltAz refraction using the meteo params.
--pressure (hPa)              Used only when --enable-refraction is set.
--temperature (°C)            Used only when --enable-refraction is set.
--humidity (0..1)             Used only when --enable-refraction is set.
--obswl (mm)                  Observing wavelength, used with refraction.

--az-offset-bias (deg)        Additive bias applied to delta_az (degrees).
--el-offset-bias (deg)        Additive bias applied to delta_el (degrees).
--peak-frac (0..1]            Threshold for scan peak selection.
--central-power-frac (0..1]   Fraction of MaxCentralPower to keep scans.

-------------------------------------------------------------------------------
Notes
-------------------------------------------------------------------------------
- Date parsing is based on the stem prefix `YYMMDDTHHMMSS...`. Stems without
  a valid timestamp are processed only when no date filter is requested.
- Excluded stems (rule `T\\d{6}b`) are never processed.
- The TSV is appended incrementally; each `[OK]` line signals a successful
  append for the current map.

"""

from __future__ import annotations

import argparse
import importlib
import re
from types import SimpleNamespace
from typing import Optional
from pathlib import Path
from datetime import date as _date

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=(
            "Import an algorithm as `solaris_pointing.offsets.algos.<algo>` and "
            "invoke its `run(params)` with a Namespace of parameters."
        )
    )
    p.add_argument(
        "--algo",
        required=False,
        help=(
            "Algorithm module name to import from "
            "`solaris_pointing.offsets.algos`. Example: --algo sun_maps"
        ),
    )
    p.add_argument(
        "--examples", action="store_true", help="Show usage examples and exit."
    )
    p.add_argument(
        "--data",
        default=None,
        help=(
            "Root directory for input discovery (recursively). If omitted, the "
            "algorithm decides its default (usually current directory)."
        ),
    )
    p.add_argument(
        "--outdir",
        default="offsets",
        help=(
            "Directory where the output TSV will be written. "
            "If it starts with '/', the path is absolute; otherwise it is "
            "relative to the current working directory. Default: ./offsets"
        ),
    )

    # Baseline parameters (mirroring current sun_maps needs)
    p.add_argument(
        "--site-lon",
        type=float,
        default=164.1000,
        help="Observatory longitude in degrees (default: 164.1000).",
    )
    p.add_argument(
        "--site-lat",
        type=float,
        default=-74.6950,
        help="Observatory latitude in degrees (default: -74.6950).",
    )
    p.add_argument(
        "--site-height",
        type=float,
        default=30.0,
        help="Observatory height in meters (default: 30.0).",
    )

    p.add_argument(
        "--az-offset-bias",
        type=float,
        default=0.0,
        help="Bias to add to delta_az in degrees (default: 0.0).",
    )
    p.add_argument(
        "--el-offset-bias",
        type=float,
        default=0.0,
        help="Bias to add to delta_el in degrees (default: 0.0).",
    )

    p.add_argument(
        "--peak-frac",
        type=float,
        default=0.75,
        help="Threshold for scan peak selection, in (0,1] (default: 0.75).",
    )
    p.add_argument(
        "--central-power-frac",
        type=float,
        default=0.60,
        help=("Fraction of MaxCentralPower to keep scans (default: 0.60)."),
    )

    p.add_argument(
        "--enable-refraction",
        action="store_true",
        default=False,
        help=(
            "Enable atmospheric refraction in AltAz. If set, the pressure/"
            "temperature/humidity/obswl values will be used."
        ),
    )
    p.add_argument(
        "--pressure",
        type=float,
        default=990.0,
        help=(
            "Observer pressure in hPa (default: 990.0). "
            "Used only if --enable-refraction."
        ),
    )
    p.add_argument(
        "--temperature",
        type=float,
        default=-5.0,
        help=(
            "Observer temperature in Celsius (default: -5.0). "
            "Used only if --enable-refraction."
        ),
    )
    p.add_argument(
        "--humidity",
        type=float,
        default=0.50,
        help=(
            "Relative humidity as fraction in [0,1] (default: 0.50). "
            "Used only if --enable-refraction."
        ),
    )
    p.add_argument(
        "--obswl",
        type=float,
        default=3.0,
        help=(
            "Observing wavelength in mm (default: 3.0). "
            "Used only if --enable-refraction."
        ),
    )

    p.add_argument(
        "--date-start",
        type=_date.fromisoformat,
        default=None,
        help=(
            "Inclusive start date (YYYY-MM-DD). "
            "If only this is given, include all data on/after this date."
        ),
    )
    p.add_argument(
        "--date-end",
        type=_date.fromisoformat,
        default=None,
        help=(
            "Inclusive end date (YYYY-MM-DD). "
            "If only this is given, include all data on/before this date."
        ),
    )
    return p


def extract_examples_from_docstring() -> str:
    """
    Extract the 'Command-line usage examples' section from the module docstring.

    The section begins at the line containing the exact title
    'Command-line usage examples' and ends at the next separator line,
    defined as any line containing at least 10 consecutive '-' characters.

    Returns a string containing the title plus the example block.
    """
    doc = __doc__ or ""
    title = "Command-line usage examples"

    # Locate the section header
    start_idx = doc.find(title)
    if start_idx == -1:
        return "No examples available."

    # Extract from the title to the end of the docstring
    block = doc[start_idx:].splitlines()

    # The first line is the title itself; keep it
    result_lines = [block[0].strip()]

    # Look for the terminating separator (>= 10 consecutive hyphens)
    separator_found = False
    hyphens = "-" * 10

    def wrap_example_line(line: str, width: int = 88) -> list[str]:
        """
        Wrap example lines preserving indentation, inserting a trailing backslash,
        and ensuring lines are only broken before tokens starting with '--'.
        """
        stripped = line.lstrip()
        indent_len = len(line) - len(stripped)
        indent = " " * indent_len

        tokens = stripped.split()
        if not tokens:
            return [line.rstrip()]

        out_lines = []
        current = indent + tokens[0]

        for tok in tokens[1:]:
            candidate = current + " " + tok

            if len(candidate) > width:
                # Can we break here? Only if tok is a switch
                if tok.startswith("--"):
                    out_lines.append(current + " \\")
                    current = indent + tok
                    continue
                # Otherwise, add anyway and DO NOT break the line here
                # (we try to break at the next switch instead)
                current = candidate
            else:
                current = candidate

        out_lines.append(current)
        return out_lines

    # Skip the title line and process subsequent lines
    for line in block[2:]:
        # Stop when encountering a separator line
        if hyphens in line:
            separator_found = True
            break
        result_lines.extend(wrap_example_line(line))

    title = result_lines[0].strip()
    body = "\n".join(result_lines[1:]).strip()
    return title, body


# ---------------------------------------------------------------------------
# Main entry
# ---------------------------------------------------------------------------
def main(argv: Optional[list[str]] = None) -> None:
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    if args.examples:
        title, body = extract_examples_from_docstring()
        line = "-" * len(title)
        print(f"\n{line}\n{title}\n{line}\n")
        print(f"{body}\n")
        return

    if not args.algo:
        parser.error("--algo is required unless --examples is provided.")

    # Build params Namespace (kept as Namespace for direct compatibility)
    params = SimpleNamespace(
        data=args.data,
        outdir=args.outdir,
        site_lon=args.site_lon,
        site_lat=args.site_lat,
        site_height=args.site_height,
        az_offset_bias=args.az_offset_bias,
        el_offset_bias=args.el_offset_bias,
        peak_frac=args.peak_frac,
        central_power_frac=args.central_power_frac,
        enable_refraction=args.enable_refraction,
        pressure=args.pressure,
        temperature=args.temperature,
        humidity=args.humidity,
        obswl=args.obswl,
        date_start=args.date_start,
        date_end=args.date_end,
    )

    # Import the algorithm module and call run(params)
    module_name = f"solaris_pointing.offsets.algos.{args.algo}"
    try:
        mod = importlib.import_module(module_name)
    except ModuleNotFoundError as e:
        raise SystemExit(
            f"ERROR: Could not import '{module_name}'. Ensure it is on PYTHONPATH "
            f"and exposes a `run(params)` function.\nOriginal error: {e}"
        )

    try:
        process_map = getattr(mod, "process_map")
    except AttributeError:
        raise SystemExit(
            f"ERROR: Module '{module_name}' does not expose a callable process_map()."
        )

    try:
        append_result_tsv = getattr(mod, "append_result_tsv")
    except AttributeError:
        raise SystemExit(
            f"ERROR: Module '{module_name}' does not expose a callable "
            f"append_result_tsv()."
        )

    # -----------------------------
    # Resolve output directory
    # -----------------------------
    # If params.outdir is not provided or empty, default to "offsets".
    out_arg = getattr(params, "outdir", None) or "offsets"
    outdir = Path(out_arg).expanduser()
    if not outdir.is_absolute():
        outdir = Path.cwd() / outdir
    outdir = outdir.resolve()
    outdir.mkdir(parents=True, exist_ok=True)

    # Compose final output TSV path: <module_stem>.tsv placed under outdir
    out_path = outdir / f"{args.algo}.tsv"
    out_name = out_path.name

    # -----------------------------
    # Discovery helpers
    # -----------------------------
    def _discover_pairs_recursive(root: Path):
        """
        Recursively find (.path, .sky) pairs under `root`, excluding names that
        contain a 'b' immediately after the time component: 'T\d{6}b'.
        Optionally filter by inclusive dates using args.date_start / args.date_end,
        where the date is parsed from the stem prefix 'YYMMDDTHHMMSS...'.
        Returns a list of tuples: (map_id, path_file, sky_file).
        """
        time_b_pat = re.compile(r"T\d{6}b")  # e.g., 250101T195415bOASI
        ts_pat = re.compile(
            r"^(?P<yy>\d{2})(?P<mo>\d{2})(?P<dd>\d{2})T"
            r"(?P<hh>\d{2})(?P<mi>\d{2})(?P<ss>\d{2})"
        )

        def _stem_date(stem: str):
            m = ts_pat.match(stem)
            if not m:
                return None
            yy = int(m.group("yy"))
            year = 2000 + yy  # pivot 2000-2099
            from datetime import date as _date

            return _date(year, int(m.group("mo")), int(m.group("dd")))

        path_files = {}
        sky_files = {}

        for pth in root.rglob("*.path"):
            name = pth.stem
            if time_b_pat.search(name):
                continue
            path_files[name] = pth

        for sky in root.rglob("*.sky"):
            name = sky.stem
            if time_b_pat.search(name):
                continue
            sky_files[name] = sky

        pairs = []
        for stem in sorted(set(path_files) & set(sky_files)):
            # Date filter (inclusive): only apply if at least one bound is given
            if args.date_start is not None or args.date_end is not None:
                d = _stem_date(stem)
                if d is None:
                    # If no date can be parsed, exclude when a filter is requested
                    continue
                if args.date_start is not None and d < args.date_start:
                    continue
                if args.date_end is not None and d > args.date_end:
                    continue
            map_id = stem  # e.g., 250101T195415_OASI
            pairs.append((map_id, str(path_files[stem]), str(sky_files[stem])))
        return pairs

    # -----------------------------
    # Discover pairs
    # -----------------------------
    if hasattr(params, "data") and params.data:
        data_arg = Path(params.data).expanduser()
        root = data_arg if data_arg.is_absolute() else Path.cwd() / data_arg
        root = root.resolve()
        if not root.exists():
            raise FileNotFoundError(f"--data path not found: {root}")
        pairs = _discover_pairs_recursive(root)
    else:
        raise FileNotFoundError(f"No data found in {root}")

    if not pairs:
        print("No valid <map_id>.path / <map_id>.sky pairs found.")
        return str(out_path)

    # -----------------------------
    # Process pairs and append results
    # -----------------------------
    total = len(pairs)

    for idx, (map_id, path_fname, sky_fname) in enumerate(pairs, start=1):
        try:
            res = process_map(map_id, path_fname, sky_fname, params)
            if res is None:
                print(f"[WARN] Skipping {map_id}: could not compute offsets.")
                continue
            append_result_tsv(str(out_path), res)
            width = len(str(total))
            print(f"[OK] {idx:0{width}d}/{total}: {map_id} -> appended to {out_name}")
        except Exception as e:
            print(f"[ERROR] {map_id}: {e}")

    print(f"[DRIVER] Output file: {out_path}")


if __name__ == "__main__":
    main()
