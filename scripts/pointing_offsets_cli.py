
#!/usr/bin/env python3
"""
pointing_offsets_cli.py
=======================

CLI runner that discovers maps, loads an algorithm plugin, and writes results
to a standardized TSV via `offset_io` (located at
`solaris_pointing.offset_core.offset_io`).

Features
--------
- Plugin architecture: algorithms live under `solaris_pointing.offset_algorithms`.
- Optional per-map environment via a single CSV (environment.csv).
- Clear separation: algorithms never write files; output is centralized here.

Usage examples
--------------
python scripts/pointing_offsets_cli.py --data-dir ./data --out ./data/offsets.tsv --algo algo_gauss2d --verbose

python scripts/pointing_offsets_cli.py --data-dir ./data --start 2025-01-01T00:00:00Z --end 2025-01-02T00:00:00Z     --env-csv ./data/environment.csv --algo algo_gauss2d
"""

from __future__ import annotations

import argparse
import importlib
import os
from typing import Dict, Optional, Tuple

from solaris_pointing.offset_core.model import Site, MapInput, Config, WriterFn
from solaris_pointing.offset_core.discovery import discover_maps
from solaris_pointing.offset_core.env_loader import load_environment_csv, validate_environment
from solaris_pointing.offset_core.io_writer import writer_factory
from solaris_pointing.offset_core.offset_io import Metadata  # offset_io moved under offset_core

def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Pointing offsets runner (plugin-based). "
                    "Loads an algorithm module and writes offsets.tsv via offset_io."
    )
    p.add_argument("--data-dir", default=".", help="Directory containing *.path/*.sky pairs (default: current dir)")
    p.add_argument("--out", default="./offsets.tsv", help="Output TSV path (default: ./offsets.tsv)")
    p.add_argument("--start", default=None, help="Start ISO timestamp (e.g. 2025-01-01T00:00:00Z)")
    p.add_argument("--end", default=None, help="End ISO timestamp (e.g. 2025-01-02T00:00:00Z)")

    # Site and metadata
    p.add_argument("--site-name", default="OASI, Antarctica")
    p.add_argument("--lat", type=float, default=-74.6933)
    p.add_argument("--lon", type=float, default=164.1000)
    p.add_argument("--elev", type=float, default=0.0)
    p.add_argument("--dish-m", type=float, default=2.0)
    p.add_argument("--freq-hz", type=float, default=100e9)
    p.add_argument("--sw-version", default="0.1.0")

    # Algorithm selection and parameters
    p.add_argument("--algo", default="algo_gauss2d",
                   help="Algorithm module name under 'solaris_pointing.offset_algorithms' (default: algo_gauss2d)")
    p.add_argument("--method", default="auto", choices=("auto","gauss2d","boresight1d"))
    p.add_argument("--fwhm", type=float, default=0.2, help="Approximate beam FWHM in degrees (default: 0.2)")
    p.add_argument("--grid-step", type=float, default=None, help="2D grid step in degrees (default: ~FWHM/3)")
    p.add_argument("--smooth-sigma", type=float, default=None, help="2D smoothing sigma in degrees (default: ~FWHM/4)")
    p.add_argument("--power-thresh", type=float, default=0.75, help="Fraction of local max to estimate central time (default: 0.75)")
    p.add_argument("--subscan-min-sec", type=float, default=1.0, help="Minimum subscan duration in seconds (default: 1.0)")
    p.add_argument("--signal-min", type=float, default=30000.0, help="Minimum raw signal threshold (default: 30000)")
    p.add_argument("--refraction", default="off", choices=("off","simple"),
                   help="Atmospheric refraction handling (default: off)")
    p.add_argument("--no-plots", action="store_true", help="Disable diagnostic plot generation")
    p.add_argument("--verbose", action="store_true", help="Verbose logging")

    # Environment CSV (Option B) + strictness
    p.add_argument("--env-csv", default=None,
                   help="CSV with per-map environment (map_id,temperature_c,pressure_hpa,humidity_frac). "
                        "If omitted, the runner will try '<data-dir>/environment.csv' if present.")
    p.add_argument("--env-strict", default="warn", choices=("warn","missing-error","ignore"),
                   help="How to treat environment mismatches. Default: warn.")
    return p

def main() -> None:
    ap = build_arg_parser()
    args = ap.parse_args()

    # Discover maps
    maps = discover_maps(args.data_dir, args.start, args.end)
    if args.verbose:
        print(f"Discovered {len(maps)} maps in {args.data_dir}")

    # Prepare site/config (passed to the algorithm)
    site = Site(
        name=args.site_name,
        latitude_deg=args.lat,
        longitude_deg=args.lon,
        elevation_m=args.elev,
    )
    cfg = Config(
        method=args.method,
        fwhm_deg=args.fwhm,
        grid_step_deg=args.grid_step,
        smooth_sigma_deg=args.smooth_sigma,
        power_thresh_frac=args.power_thresh,
        subscan_min_sec=args.subscan_min_sec,
        signal_min=args.signal_min,
        refraction=args.refraction,
        make_plots=(not args.no_plots),
        verbose=args.verbose,
    )

    # Prepare offset_io metadata
    md = Metadata(
        location=site.name,
        antenna_diameter_m=args.dish_m,
        frequency_hz=args.freq_hz,
        software_version=args.sw_version,
    )

    # Load environment CSV if provided or auto-detected
    env_csv_path = args.env_csv
    if env_csv_path is None:
        candidate = os.path.join(args.data_dir, "environment.csv")
        if os.path.exists(candidate):
            env_csv_path = candidate

    env_by_map: Dict[str, Tuple[Optional[float], Optional[float], Optional[float]]] = {}
    if env_csv_path is not None:
        if args.verbose:
            print(f"Loading environment CSV: {env_csv_path}")
        env_by_map = load_environment_csv(env_csv_path)
        warnings = validate_environment(env_by_map, maps)

        if args.env_strict == "missing-error":
            if warnings:
                for w in warnings:
                    print(f"ERROR: {w}")
                raise SystemExit(1)
        elif args.env_strict == "warn":
            for w in warnings:
                print(f"WARNING: {w}")
        # 'ignore' → no messages

    # Build timestamp_iso -> map_id mapping so the writer can attach env values
    ts_to_map: Dict[str, str] = {mp.map_timestamp_iso: mp.map_id for mp in maps}

    # Build writer with environment enrichment
    write_row = writer_factory(args.out, md, env_by_map, ts_to_map)

    # Import algorithm module dynamically (accept plain name or fully qualified)
    algo_name = args.algo
    if "." not in algo_name:
        algo_name = f"solaris_pointing.offset_algorithms.{algo_name}"
    algo = importlib.import_module(algo_name)
    if not hasattr(algo, "compute_offsets"):
        raise RuntimeError(f"Algorithm module '{algo_name}' has no 'compute_offsets' function.")

    # Hand off to the algorithm
    algo.compute_offsets(maps=maps, site=site, cfg=cfg, write_row=write_row)

    if args.verbose:
        print(f"Done. Results appended to: {args.out}")


if __name__ == "__main__":
    main()
