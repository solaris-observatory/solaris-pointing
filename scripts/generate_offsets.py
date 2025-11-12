
#!/usr/bin/env python3
"""
Generic driver for solaris_pointing offset-fitting algorithms.

- Parses a baseline parameter set (compatible with current `sun_maps`).
- Encapsulates them in an argparse.Namespace (keeps algorithms compatible).
- Dynamically imports `solaris_pointing.offsets.algos.<algo>` and calls
  its public `run(params)` function.

New:
- `--data` lets you specify the root directory where input files reside.
  The algorithm may use this field to discover files recursively.
"""
from __future__ import annotations

import argparse
import importlib
from types import SimpleNamespace
from typing import Optional


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
        required=True,
        help=(
            "Algorithm module name to import from "
            "`solaris_pointing.offsets.algos`. Example: --algo sun_maps"
        ),
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
    p.add_argument("--site-lon", type=float, default=164.1000,
                   help="Observatory longitude in degrees (default: 164.1000).")
    p.add_argument("--site-lat", type=float, default=-74.6950,
                   help="Observatory latitude in degrees (default: -74.6950).")
    p.add_argument("--site-height", type=float, default=30.0,
                   help="Observatory height in meters (default: 30.0)." )

    p.add_argument("--az-offset-bias", type=float, default=0.0,
                   help="Bias to add to delta_az in degrees (default: 0.0)." )
    p.add_argument("--el-offset-bias", type=float, default=0.0,
                   help="Bias to add to delta_el in degrees (default: 0.0)." )

    p.add_argument("--peak-frac", type=float, default=0.75,
                   help="Threshold for scan peak selection, in (0,1] (default: 0.75)." )
    p.add_argument("--central-power-frac", type=float, default=0.60,
                   help=("Fraction of MaxCentralPower to keep scans (default: 0.60)."))

    p.add_argument("--enable-refraction", action="store_true", default=False,
                   help=("Enable atmospheric refraction in AltAz. If set, the pressure/"
                         "temperature/humidity/obswl values will be used."))
    p.add_argument("--pressure", type=float, default=990.0,
                   help="Observer pressure in hPa (default: 990.0). Used only if --enable-refraction.")
    p.add_argument("--temperature", type=float, default=-5.0,
                   help="Observer temperature in Celsius (default: -5.0). Used only if --enable-refraction.")
    p.add_argument("--humidity", type=float, default=0.50,
                   help="Relative humidity as fraction in [0,1] (default: 0.50). Used only if --enable-refraction.")
    p.add_argument("--obswl", type=float, default=3.0,
                   help="Observing wavelength in mm (default: 3.0). Used only if --enable-refraction.")
    return p


# ---------------------------------------------------------------------------
# Main entry
# ---------------------------------------------------------------------------
def main(argv: Optional[list[str]] = None) -> None:
    parser = build_arg_parser()
    args = parser.parse_args(argv)

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

    if not hasattr(mod, "run") or not callable(getattr(mod, "run")):
        raise SystemExit(
            f"ERROR: Module '{module_name}' does not expose a callable run(params)."
        )

    result = mod.run(params)  # The algorithm is responsible for I/O and return value
    if result is not None:
        print(f"[DRIVER] Algorithm returned: {result}")


if __name__ == "__main__":
    main()
