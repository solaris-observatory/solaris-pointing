#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
az_model_cli.py
================

CLI to fit and use *azimuth-only* pointing models for telescope offsets.
Both offset_az and offset_el are modeled as polynomials of azimuth only.

Data format
-----------
Input is a TSV (tab-separated) with optional comment lines starting with '#'
and at least the following columns (all angles may be in degrees or arcsec):
    * azimuth
    * offset_az
    * offset_el
If your offsets are in arcseconds, set --input-offset-unit=arcsec.

Examples
--------
Fit models from the example dataset (offsets are in arcseconds) and save them
to the default "models/" directory. Also write a short summary text file.

    python scripts/az_model_cli.py \
        templates/output_offset_io_example.tsv \
        --input-offset-unit arcsec \
        --degree 3 \
        --summary models/fit_summary.txt

Fit models and save them to a custom directory (e.g. "custom_models/"):

    python scripts/az_model_cli.py \
        templates/output_offset_io_example.tsv \
        --input-offset-unit arcsec \
        --degree 3 \
        --save-az-model custom_models/az_model.joblib \
        --save-el-model custom_models/el_model.joblib \
        --summary custom_models/fit_summary.txt

Fit and show a plot with raw, filtered, and fitted curves; also save a PNG:

    python scripts/az_model_cli.py \
        templates/output_offset_io_example.tsv \
        --input-offset-unit arcsec \
        --degree 3 \
        --plot \
        --plot-file models/fit_plot.png

Use arcminutes as input unit and show the plot in arcminutes too:

    python scripts/az_model_cli.py \
        templates/output_offset_io_example.tsv \
        --input-offset-unit arcmin \
        --degree 3 \
        --plot \
        --plot-unit arcmin

Predict both offsets, in degrees, at azimuth 125.0 using the saved models:

    python scripts/az_model_cli.py --predict 125.0

Use custom model paths for prediction:

    python scripts/az_model_cli.py \
        --predict 125.0 \
        --az-model models/az_model.joblib \
        --el-model models/el_model.joblib

Notes
-----
* All modeling is done in **degrees**. If the input file stores offsets in
  arcseconds or arcminutes, the script converts them to degrees before fitting.
* Recommended workflow: re-fit the model every N days so that the approximation
  "offsets depend only on azimuth" remains valid for your observing window.
"""

from __future__ import annotations

import argparse
import os
from typing import Optional

import numpy as np

# Import the library from the project package layout
from solaris_pointing.fitting.az_model import (
    fit_models,
    fit_models_from_tsv,
    read_offsets_tsv,
    save_models,
    load_models,
    predict_offsets_deg,
    model_summary,
)


def _ensure_parent_dir(path: str) -> None:
    """Create the parent directory of *path* if it does not exist."""
    parent = os.path.dirname(path) or "."
    os.makedirs(parent, exist_ok=True)


def _remove_outliers_z(y: np.ndarray, z: float) -> np.ndarray:
    """Return boolean mask where |zscore(y)| < z; robust to zero/NaN std."""
    y = np.asarray(y, dtype=float)
    std = float(np.nanstd(y))
    if std == 0 or np.isnan(std):
        return np.ones_like(y, dtype=bool)
    zscores = np.abs((y - np.nanmean(y)) / std)
    return zscores < z


def _deg_to_factor(unit: str) -> float:
    """Return multiplier to convert degrees to the requested unit."""
    if unit == "deg":
        return 1.0
    if unit == "arcmin":
        return 60.0
    if unit == "arcsec":
        return 3600.0
    raise ValueError(f"Unsupported unit: {unit}")


def _plot_fit(
    df, models, degree: int, zscore: float, out_png: str | None, plot_unit: str = "deg"
):
    """Plot raw, filtered, and fitted curves for both offsets vs azimuth."""
    import matplotlib.pyplot as plt  # lazy import

    az = df["azimuth"].to_numpy(dtype=float)
    off_az = df["offset_az"].to_numpy(dtype=float)
    off_el = df["offset_el"].to_numpy(dtype=float)

    m_az, m_el = models.az_model, models.el_model
    grid = np.linspace(np.min(az), np.max(az), 400)

    mask_az = _remove_outliers_z(off_az, zscore)
    mask_el = _remove_outliers_z(off_el, zscore)

    # Convert degrees into requested plot unit
    fac = _deg_to_factor(plot_unit)
    off_az_p = off_az * fac
    off_el_p = off_el * fac
    m_az_p = m_az(grid) * fac
    m_el_p = m_el(grid) * fac

    fig, axs = plt.subplots(1, 2, figsize=(10, 4))

    # Left: offset_az
    ax = axs[0]
    ax.scatter(az, off_az_p, s=8, alpha=0.35, label="raw")
    ax.scatter(az[mask_az], off_az_p[mask_az], s=10, alpha=0.9, label="kept")
    ax.plot(grid, m_az_p, lw=1.5, label=f"fit (deg={degree})")
    ax.set_title("offset_az vs azimuth")
    ax.set_xlabel("azimuth [deg]")
    ax.set_ylabel(f"offset_az [{plot_unit}]")
    ax.grid(True, ls=":")
    ax.legend(loc="best", fontsize=8)

    # Right: offset_el
    ax = axs[1]
    ax.scatter(az, off_el_p, s=8, alpha=0.35, label="raw")
    ax.scatter(az[mask_el], off_el_p[mask_el], s=10, alpha=0.9, label="kept")
    ax.plot(grid, m_el_p, lw=1.5, label=f"fit (deg={degree})")
    ax.set_title("offset_el vs azimuth")
    ax.set_xlabel("azimuth [deg]")
    ax.set_ylabel(f"offset_el [{plot_unit}]")
    ax.grid(True, ls=":")
    ax.legend(loc="best", fontsize=8)

    fig.tight_layout()
    if out_png:
        _ensure_parent_dir(out_png)
        fig.savefig(out_png, dpi=150)
    plt.show()


def build_parser() -> argparse.ArgumentParser:
    """Build the CLI parser."""
    p = argparse.ArgumentParser(
        prog="az_model_cli.py",
        description=(
            "Fit azimuth-only offset models from a TSV file, or predict both "
            "offsets at a given azimuth using saved models."
        ),
    )
    p.add_argument(
        "tsv",
        nargs="?",
        help=(
            "Path to TSV with columns: azimuth, offset_az, offset_el. "
            "Use with fit mode."
        ),
    )
    p.add_argument(
        "--degree",
        type=int,
        default=3,
        help="Polynomial degree for both az and el models (default: 3).",
    )
    p.add_argument(
        "--zscore",
        type=float,
        default=3.0,
        help="Outlier rejection threshold on |z| (default: 3.0).",
    )
    p.add_argument(
        "--input-offset-unit",
        choices=["deg", "arcmin", "arcsec"],
        default="deg",
        help=(
            "Unit of offset_az/offset_el in the input TSV (default: deg). "
            "Use 'arcmin' or 'arcsec' if your file stores those units."
        ),
    )
    p.add_argument(
        "--save-az-model",
        default="models/az_model.joblib",
        help=(
            "Output path for the azimuth model file (default: models/az_model.joblib)."
        ),
    )
    p.add_argument(
        "--save-el-model",
        default="models/el_model.joblib",
        help=(
            "Output path for the elevation model file (default: "
            "models/el_model.joblib)."
        ),
    )
    p.add_argument(
        "--summary",
        default=None,
        help="Optional path to save a human-readable fit summary.",
    )
    p.add_argument(
        "--plot",
        action="store_true",
        help="Show a plot window with raw, kept, and fitted curves.",
    )
    p.add_argument(
        "--plot-file",
        default=None,
        help="Optional PNG path to save the plot figure.",
    )
    p.add_argument(
        "--plot-unit",
        choices=["deg", "arcmin", "arcsec"],
        default="deg",
        help=(
            "Unit for y-axis in plots (default: deg). Values are converted "
            "from degrees."
        ),
    )
    p.add_argument(
        "--predict",
        type=float,
        default=None,
        help=(
            "Predict mode: azimuth in degrees at which to predict both offsets. "
            "If set, fitting is skipped and saved models are loaded."
        ),
    )
    p.add_argument(
        "--az-model",
        default="models/az_model.joblib",
        help=(
            "Path to a saved azimuth model (used only with --predict). "
            "Default: models/az_model.joblib"
        ),
    )
    p.add_argument(
        "--el-model",
        default="models/el_model.joblib",
        help=(
            "Path to a saved elevation model (used only with --predict). "
            "Default: models/el_model.joblib"
        ),
    )
    return p


def main(argv: Optional[list[str]] = None) -> int:
    """Run the CLI."""
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.predict is not None:
        # Prediction mode
        az_model, el_model = load_models(args.az_model, args.el_model)
        off_az_deg, off_el_deg = predict_offsets_deg(az_model, el_model, args.predict)
        print(f"Input azimuth [deg]: {args.predict:.6f}")
        print(f"Predicted offset_az [deg]: {off_az_deg:.6f}")
        print(f"Predicted offset_el [deg]: {off_el_deg:.6f}")
        return 0

    # Fit mode
    if not args.tsv:
        parser.error("the following arguments are required for fit mode: tsv")

    # Read data and convert to degrees for modeling
    if args.input_offset_unit == "arcmin":
        df = read_offsets_tsv(args.tsv, input_offset_unit="deg")
        df["offset_az"] = df["offset_az"] / 60.0
        df["offset_el"] = df["offset_el"] / 60.0
    else:
        df = read_offsets_tsv(args.tsv, input_offset_unit=args.input_offset_unit)

    print("Assuming input offsets unit:", args.input_offset_unit)

    models = fit_models(
        df["azimuth"],
        df["offset_az"],
        df["offset_el"],
        degree=args.degree,
        zscore=args.zscore,
    )

    # Save models
    _ensure_parent_dir(args.save_az_model)
    _ensure_parent_dir(args.save_el_model)
    save_models(models, args.save_az_model, args.save_el_model)

    # Print and optionally save a human-readable summary
    summary = model_summary(models)
    print(summary)
    if args.summary:
        _ensure_parent_dir(args.summary)
        with open(args.summary, "w", encoding="utf-8") as f:
            f.write(summary)

    # Optional plotting
    if args.plot:
        _plot_fit(
            df,
            models,
            degree=args.degree,
            zscore=args.zscore,
            out_png=args.plot_file,
            plot_unit=args.plot_unit,
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
