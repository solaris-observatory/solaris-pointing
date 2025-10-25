#!/usr/bin/env python3
"""Fit azimuth-only pointing offsets.

This CLI wraps the core azimuth-only modeling library to fit, summarize,
and predict telescope pointing offsets directly from TSV input files
or saved model bundles (.joblib). It provides a convenient interface for
campaign data analysis, comparison across datasets, and understanding the
physical behavior of telescope pointing systems.

-------------------------------------------------------------------------------
Available subcommands
-------------------------------------------------------------------------------
fit       Fit az/el models from one or more TSV input files.
predict   Predict az/el offsets for a given azimuth using a saved model.
summary   Print model diagnostics and metadata.

-------------------------------------------------------------------------------
Command-line usage examples
-------------------------------------------------------------------------------
1. Fit a single file alpacino.tsv:

   python scripts/az_model_cli.py fit alpacino.tsv \
       --degree 3 --zscore-az 3.5 --zscore-el 3.5

   Default: reads from offsets/, saves to models/.
   (If you pass a path with a directory component, no offsets/ fallback is used.)

   Generated files:
   - models/alpacino.joblib
   - models/alpacino_summary.txt
   - models/alpacino_az.png
   - models/alpacino_el.png

2. Fit multiple files at once:

   python scripts/az_model_cli.py fit alpacino.tsv leone.tsv \
       --degree 3 --zscore-az 3.5 --zscore-el 3.5 \
       --fourier-k 2 --plot-unit arcmin

   Saves one bundle and one pair of PNG plots per input file, using each input
   filename stem. Also produces a combined curves-only plot if multiple files
   are provided. The combined output consists of two images (az/el), named by
   concatenating stems (in alphabetical order) using "+" as a separator:
   - models/<stem1+stem2+...>_az.png
   - models/<stem1+stem2+...>_el.png

3. Fit with Fourier harmonics and ridge regularization:

   python scripts/az_model_cli.py fit alpacino.tsv \
       --degree 3 --zscore-az 3.5 --zscore-el 3.5 \
       --fourier-k 2 --ridge-alpha 0.01

   The --ridge-alpha parameter (default 0.01) stabilizes high-degree fits
   or unevenly sampled data by shrinking coefficients slightly. Physically,
   it mitigates effects of near-collinearity among basis functions that can
   arise from mechanical geometries or non-uniform sampling over azimuth.

4. Predict from an existing model:

   python scripts/az_model_cli.py predict models/alpacino.joblib \
       --az 12.0 --unit arcsec --allow-extrapolation

   The prediction is printed in the requested unit. With --allow-extrapolation
   (available only for `predict`), the tool permits evaluation slightly outside
   the observed azimuth range used during fitting; accuracy outside this range
   is not guaranteed.

5. Fit with custom output locations:

   python scripts/az_model_cli.py fit alpacino.tsv leone.tsv \
       --save-model models/marongiu.joblib --summary results/summary.txt \
       --degree 3 --zscore-az 3.5 --zscore-el 3.5

   When multiple files are provided, the tool automatically appends the
   input filename stem before the extension for both --save-model and --summary.

-------------------------------------------------------------------------------
Input and output conventions
-------------------------------------------------------------------------------
- Input TSV files are searched under `offsets/` if no directory component is
  given in the path. If a directory is included, the file is taken as-is.
- Default output directory for models and plots is `models/`.
- Combined multi-file plots concatenate stems in alphabetical order using "+".

-------------------------------------------------------------------------------
Parameters and their physical meaning
-------------------------------------------------------------------------------
- `degree` (int): polynomial degree for the fit (1=linear, 2=quadratic, 3=cubic, ...)
  Higher degree increases flexibility but may overfit.

- `zscore-az`, `zscore-el` (float): robust outlier rejection thresholds based
  on MAD. Points with residuals beyond (threshold × MAD) are ignored by the
  final fit. Physically, these help exclude data affected by transient
  mechanical effects, wind, seeing, or other disturbances.

- `ridge-alpha` (float): L2 regularization strength (default 0.01);
  small positive values help stabilize high-degree polynomials. Physically,
  this limits sensitivity to redundant or correlated basis functions due to
  geometry or uneven azimuth coverage.

- `fourier-k` (int): number of Fourier harmonics to add (0 disables them).
  Physical meaning: captures periodic azimuthal errors (e.g., encoder defects,
  gear eccentricity, or cable-wrap effects) by adding sine/cosine terms that
  modulate the offset as a function of azimuth.

- `periods-deg` (str): comma-separated list of custom periods (degrees per cycle),
  e.g., "90,45". Use to model periodicities not tied to simple 1/rev harmonics
  (e.g., worm gear ratios or encoder line counts).
  Physical meaning: focuses the model on known mechanical cycles or periodic
  sources, reducing aliasing and improving interpretability.

- `sector-edges-deg` (str): comma-separated sector edges in degrees (e.g., 60,210).
  Physical meaning: introduces step-like (piecewise) biases between azimuth
  sectors to model static offsets caused by backlash, hysteresis, or mechanical
  regime changes (e.g., side-switching, assembly discontinuities).

- `input-offset-unit` (deg|arcmin|arcsec): unit of `offset_az/offset_el` in the TSV.
  Internally everything is converted to degrees before fitting. By default, the
  tool expects degrees, but you can override this with --input-offset-unit.

- `notes` (str): free text stored in the model metadata for traceability.

- `plot-unit` (deg|arcmin|arcsec): rendering unit for plots (Y axis only).
  In `fit`, it applies to the automatically saved figures.

- `save-model` (path): where to save the .joblib bundle. When fitting multiple
  files, the input filename stem is appended automatically before the extension.

- `summary` (path): where to save a human-readable text report. For multiple
  input files, the input stem is inserted before the extension.

- `allow-extrapolation` (predict only, bool): allow evaluation beyond the
  observed azimuth range (use with care; extrapolated values may not be
  physically reliable).

-------------------------------------------------------------------------------
Harmonic and sector modeling
-------------------------------------------------------------------------------
Harmonics (`fourier-k`) add sin(k·A) and cos(k·A) with A the wrapped azimuth,
to describe smooth periodic effects over one revolution (0–360°). They are
useful when cyclic sources exist -- encoder eccentricities, gear errors,
cable-wrap torsions, etc. The model adapts to repeating oscillations
over azimuth.

- **Custom periods** (`periods-deg`): specify explicit periods P (in degrees)
  for the harmonic terms (sin/cos with period P). This focuses the model on
  physically meaningful frequencies -- e.g., a 360/N gear tooth pattern --
  reducing aliasing and improving interpretability.

- **Sector edges** (`sector-edges-deg`): define boundaries (in degrees) that
  partition azimuth into regions. The model assigns distinct biases to
  each sector, representing discontinuities linked to backlash, play release,
  or mechanical side-switching. These terms are discrete, not smooth.

Optional parameters `--periods-deg` and `--sector-edges-deg` can be used
together within the same `fit` command.

-------------------------------------------------------------------------------
What the saved figures show
-------------------------------------------------------------------------------
Per-file plots visualize:
- The linearized azimuth (`az_lin`) on the x-axis.
- Scatter points for measured offsets and the fitted curve.
- Robust inlier/outlier masks using z-score thresholds from the bundle
  metadata when available, or CLI thresholds otherwise.
- Units displayed according to --plot-unit.
- Multi-file fits include an additional combined curves-only overlay
  (two PNGs, suffixes _az/_el) saved when more than one TSV is given to `fit`.
  In the combined plots, the legend lists file stems only, while fit parameters
  for each file are summarized in the figure title.

-------------------------------------------------------------------------------
Outputs and summaries
-------------------------------------------------------------------------------
Each fit produces:
- A .joblib bundle containing both azimuth and elevation models, metadata,
  and fit diagnostics.
- One or more .png plots showing inliers/outliers and fitted curves.
- A text summary file containing key statistics and any user-provided notes.

-------------------------------------------------------------------------------
Practical guidance for fitting
-------------------------------------------------------------------------------
- Start with a low polynomial degree (2 or 3), then add `fourier-k` if residuals
  show periodic structure over azimuth. Consider `periods-deg` when a source
  has a known geometric period.
- Use `sector-edges-deg` only when residuals exhibit step-like behavior across
  azimuth ranges, typical of backlash or regime switching.
- Avoid excessive complexity (high k, many edges) with small datasets, as it
  increases the risk of overfitting and reduces robustness.
- For best performance, provide at least one full revolution of azimuth data
  or more than 200 measurement points per dataset.
"""

from __future__ import annotations

import argparse
import sys
import os
import math
import itertools

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

try:
    # Package import (preferred path when installed as part of solaris_pointing)
    from solaris_pointing.fitting.az_model import (
        fit_models_from_tsv,
        load_model_bundle,
        save_model_bundle,
        predict_offsets_deg,
        model_summary,
        read_offsets_tsv,
        unwrap_azimuth,
        _mad,
    )
except Exception:
    # Fallback: allow running this CLI "standalone" next to az_model.py
    from az_model import (
        fit_models_from_tsv,
        load_model_bundle,
        save_model_bundle,
        predict_offsets_deg,
        model_summary,
        read_offsets_tsv,
        unwrap_azimuth,
        _mad,
    )



# -----
def _stem(path: str) -> str:
    return os.path.splitext(os.path.basename(path))[0]

def _with_ext_same_dir(path: str, new_ext: str) -> str:
    """Return path in the SAME directory as input, same stem, with new_ext
    (including dot)."""
    d = os.path.dirname(path)
    s = os.path.splitext(os.path.basename(path))[0]
    return os.path.join(d or ".", s + new_ext)


def _insert_stem_before_ext(base_path: str, stem: str) -> str:
    """Insert _{stem} before the extension of base_path.
    Example:
      - models/x.joblib -> models/x_{stem}.joblib
      - models/summary.txt -> models/summary_{stem}.txt
    If base_path has no extension, appends _{stem}.
    """
    base, ext = os.path.splitext(base_path)
    return f"{base}_{stem}{ext}"


def _ensure_dir(path: str) -> None:
    d = os.path.dirname(path)
    if d:
        os.makedirs(d, exist_ok=True)

def _parse_csv_floats(text: str):
    text = (text or '').strip()
    if not text:
        return None
    out = []
    for tok in text.split(','):
        tok = tok.strip()
        if not tok:
            continue
        try:
            out.append(float(tok))
        except Exception:
            raise ValueError(f"Invalid float in list: {tok!r}")
    return out if out else None


def _resolve_input_tsv(path: str) -> str:
    """If 'path' has no directory component, look for it under 'offsets/'."""
    if os.path.isabs(path) or os.sep in path:
        return path
    cand = os.path.join("offsets", path)
    return cand


def _default_models_dir() -> str:
    return "models"


def _default_model_path_for_stem(stem: str) -> str:
    return os.path.join(_default_models_dir(), stem + ".joblib")


def _default_plot_paths_for_stem(stem: str) -> tuple[str, str]:
    base = os.path.join(_default_models_dir(), stem + ".png")
    root, ext = os.path.splitext(base)
    return f"{root}_az{ext}", f"{root}_el{ext}"


# -------
# Helpers
# -------

def _to_deg_from_unit(x: np.ndarray, unit: str) -> np.ndarray:
    if unit == "deg":
        return x
    elif unit == "arcmin":
        return x / 60.0
    elif unit == "arcsec":
        return x / 3600.0
    else:
        raise ValueError(f"Unknown unit: {unit!r}")


def _axis_factor_for_unit(unit: str) -> float:
    if unit == "deg":
        return 1.0
    elif unit == "arcmin":
        return 60.0
    elif unit == "arcsec":
        return 3600.0
    else:
        raise ValueError(f"Unknown unit: {unit!r}")


# -----------
# Subcommands
# -----------

def cmd_fit(args: argparse.Namespace) -> int:

    # Accept multiple TSV inputs
    raw_paths = args.tsv if isinstance(args.tsv, (list, tuple)) else [args.tsv]
    tsv_paths = [_resolve_input_tsv(p) for p in raw_paths]
    # (stem, bundle, path, az_lin, off_az, off_el, period_tuple)
    bundles: list[tuple[str, any, str, np.ndarray, np.ndarray, np.ndarray, tuple[str|None, str|None]]] = []

    # Ensure output base directory exists (models/) for defaults
    os.makedirs(_default_models_dir(), exist_ok=True)

    # Fit per file
    for path in tsv_paths:
        work_path = path

        # Optional unit conversion to degrees
        if args.input_offset_unit != "deg":
            df = read_offsets_tsv(work_path).copy()
            df["offset_az"] = _to_deg_from_unit(
                df["offset_az"].to_numpy(float), args.input_offset_unit
            )
            df["offset_el"] = _to_deg_from_unit(
                df["offset_el"].to_numpy(float), args.input_offset_unit
            )
            tmp_path = work_path + ".deg.tmp.tsv"
            df.to_csv(tmp_path, sep="\t", index=False)
            work_path = tmp_path

        bundle = fit_models_from_tsv(
            path=work_path,
            degree=args.degree,
            zscore_az=args.zscore_az,
            zscore_el=args.zscore_el,
            ridge_alpha=args.ridge_alpha,
            notes=args.notes,
            fourier_k=args.fourier_k,
            periods_deg=_parse_csv_floats(args.periods_deg),
            sector_edges_deg=_parse_csv_floats(args.sector_edges_deg),
        )

        stem = _stem(path)

        # Decide save path for model
        if args.save_model:
            model_path = _insert_stem_before_ext(args.save_model, stem) \
                if len(tsv_paths) > 1 else args.save_model
        else:
            model_path = _default_model_path_for_stem(stem)

        _ensure_dir(model_path)
        save_model_bundle(bundle, model_path)
        print(f"Saved model: {model_path}")

        # Summary path
        if args.summary:
            summ_path = _insert_stem_before_ext(args.summary, stem) \
                if len(tsv_paths) > 1 else args.summary
        else:
            summ_path = os.path.join(_default_models_dir(), f"{stem}_summary.txt")

        _ensure_dir(summ_path)
        with open(summ_path, "w", encoding="utf-8") as f:
            f.write(model_summary(bundle))
        print(f"Wrote summary: {summ_path}")

        # Read back data for plotting & period detection
        dfp = read_offsets_tsv(work_path)
        az = (dfp["azimuth"].to_numpy(float) % 360.0)
        off_az = dfp["offset_az"].to_numpy(float)
        off_el = dfp["offset_el"].to_numpy(float)
        az_lin, cut, lo, hi = unwrap_azimuth(az)

        # --- Compute (YY/MM/DD -- YY/MM/DD) period tuple for this file ---
        period_tuple: tuple[str | None, str | None] = (None, None)
        try:
            cols = [c.strip() for c in dfp.columns]
            if "timestamp" in cols:
                s = dfp["timestamp"].astype(str).str.strip()
                ts = pd.to_datetime(s, utc=True, errors="coerce")
                if ts.isna().all():
                    ts = pd.to_datetime(s, format="%Y-%m-%dT%H:%M:%S.%fZ",
                                        utc=True, errors="coerce")
                if ts.isna().all():
                    ts = pd.to_datetime(s, format="%Y-%m-%dT%H:%M:%SZ",
                                        utc=True, errors="coerce")
                if ts.isna().all():
                    ts = pd.to_datetime(s.str.replace("Z", "+00:00", regex=False),
                                        utc=True, errors="coerce")
                ts = ts.dropna()
                if len(ts) > 0:
                    oldest = ts.min().to_pydatetime()
                    newest = ts.max().to_pydatetime()
                    period_tuple = (f"{oldest:%y/%m/%d}", f"{newest:%y/%m/%d}")
        except Exception:
            period_tuple = (None, None)
        # ---------------------------------------------------------------------

        # Recompute robust masks using thresholds from metadata
        yhat_az = bundle.az_model(az_lin)
        res_az = off_az - yhat_az
        yhat_el = bundle.el_model(az_lin)
        res_el = off_el - yhat_el

        def mad(x):
            med = np.median(x)
            m = np.median(np.abs(x - med))
            return 1.4826 * m if m > 0 else 0.0

        def mk_mask(res, thr):
            s = mad(res)
            if s == 0.0:
                return np.ones_like(res, dtype=bool)
            return np.abs(res) <= thr * s

        thr_az = getattr(bundle.meta, "zscore_az", args.zscore_az)
        thr_el = getattr(bundle.meta, "zscore_el", args.zscore_el)
        m_az = mk_mask(res_az, thr_az)
        m_el = mk_mask(res_el, thr_el)

        # Figures per file
        fig1, ax1 = plt.subplots(figsize=(7, 4))
        _plot_fit(ax1, az_lin, off_az, bundle.az_model, m_az,
                  unit=args.plot_unit, label_y="offset_az", bundle=bundle)
        fig2, ax2 = plt.subplots(figsize=(7, 4))
        _plot_fit(ax2, az_lin, off_el, bundle.el_model, m_el,
                  unit=args.plot_unit, label_y="offset_el", bundle=bundle)

        meta = bundle.meta
        params_line = (
            f"degree={meta.degree}, α={meta.ridge_alpha:g}, "
            f"zA={meta.zscore_az:g}, zE={meta.zscore_el:g}, f={meta.fourier_k:g}"
        )

        # ---- Titles for single-file plots ----
        if period_tuple[0] and period_tuple[1]:
            top_line = f"{stem}: ({period_tuple[0]} -- {period_tuple[1]})"
        else:
            top_line = stem

        f1, f2 = _default_plot_paths_for_stem(stem)
        for fig in (fig1, fig2):
            fig.subplots_adjust(top=0.86, bottom=0.13)
            fig.suptitle(top_line, fontsize=8, y=0.97)
            fig.text(0.5, 0.930, params_line, ha="center", va="top", fontsize=8)

        fig1.savefig(f1, dpi=300, bbox_inches="tight")
        fig2.savefig(f2, dpi=300, bbox_inches="tight")
        plt.close(fig1)
        plt.close(fig2)
        print(f"Saved plots: {f1} , {f2}")

        # Track for combined (include period_tuple)
        bundles.append((stem, bundle, path, az_lin, off_az, off_el, period_tuple))

        # Clean temp, if any
        if work_path.endswith(".deg.tmp.tsv") and os.path.exists(work_path):
            os.remove(work_path)

    # -------------------------------
    # Combined curves-only plot
    # -------------------------------
    if len(bundles) > 1:
        # Sorted by stem for deterministic naming
        bundles_sorted = sorted(bundles, key=lambda t: t[0])
        stems_sorted = [t[0] for t in bundles_sorted]
        name_base = "+".join(stems_sorted)

        base_path = os.path.join(_default_models_dir(), name_base + ".png")
        root, ext = os.path.splitext(base_path)
        f1 = f"{root}_az{ext}"
        f2 = f"{root}_el{ext}"

        # Determine if all periods are identical and non-None
        periods = [t[6] for t in bundles_sorted]
        def _period_equal(p1, p2):
            return (p1[0] is not None and p1[1] is not None and
                    p2[0] is not None and p2[1] is not None and
                    p1[0] == p2[0] and p1[1] == p2[1])

        all_equal = False
        common_period: tuple[str|None, str|None] = (None, None)
        if periods and periods[0][0] and periods[0][1]:
            all_equal = all(_period_equal(periods[0], p) for p in periods)
            if all_equal:
                common_period = periods[0]

        # Title (top): only file names; if periods differ, show period per name.
        if all_equal:
            top_title = "  +  ".join(stems_sorted)
        else:
            decorated = []
            for (stem, _, _, _, _, _, per) in bundles_sorted:
                if per[0] and per[1]:
                    decorated.append(f"{stem}: ({per[0]} -- {per[1]})")
                else:
                    decorated.append(stem)
            top_title = "  +  ".join(decorated)

        # Sub-title (centered line below): always parameters ONCE.
        # If periods are identical, append the common period here.
        meta0 = bundles_sorted[0][1].meta
        params_line = (
            f"degree={meta0.degree}, α={meta0.ridge_alpha:g}, "
            f"zA={meta0.zscore_az:g}, zE={meta0.zscore_el:g}, f={meta0.fourier_k:g}"
        )
        if all_equal and common_period[0] and common_period[1]:
            params_line = f"{params_line}     ({common_period[0]} -- {common_period[1]})"

        # ---- AZ combined ----
        fig1, ax1 = plt.subplots(figsize=(7, 4))
        fac = _axis_factor_for_unit(args.plot_unit)

        for (stem, bundle, path, az_lin, off_az, off_el, _per) in bundles_sorted:
            res_tot = (off_az - bundle.az_model(az_lin)) * fac
            lbl = f"{stem} (MAD={_mad(res_tot):.2g})"
            xs = np.linspace(float(az_lin.min()), float(az_lin.max()), 600)
            ax1.plot(xs, bundle.az_model(xs) * fac, linewidth=2.0, label=lbl)

        ax1.set_xlabel("az_lin (deg)")
        ax1.set_ylabel(f"offset_az ({args.plot_unit})")
        ax1.grid(True, alpha=0.25)
        ax1.legend()

        fig1.subplots_adjust(top=0.86, bottom=0.13)
        fig1.suptitle(top_title, fontsize=8, y=0.97)
        fig1.text(0.5, 0.930, params_line, ha="center", va="top", fontsize=8)
        fig1.savefig(f1, dpi=300, bbox_inches="tight")
        plt.close(fig1)

        # ---- EL combined ----
        fig2, ax2 = plt.subplots(figsize=(7, 4))
        for (stem, bundle, path, az_lin, off_az, off_el, _per) in bundles_sorted:
            res_tot_el = (off_el - bundle.el_model(az_lin)) * fac
            lbl = f"{stem} (MAD={_mad(res_tot_el):.2g})"
            xs = np.linspace(float(az_lin.min()), float(az_lin.max()), 600)
            ax2.plot(xs, bundle.el_model(xs) * fac, linewidth=2.0, label=lbl)

        ax2.set_xlabel("az_lin (deg)")
        ax2.set_ylabel(f"offset_el ({args.plot_unit})")
        ax2.grid(True, alpha=0.25)
        ax2.legend()

        fig2.subplots_adjust(top=0.86, bottom=0.13)
        fig2.suptitle(top_title, fontsize=8, y=0.97)
        fig2.text(0.5, 0.930, params_line, ha="center", va="top", fontsize=8)
        fig2.savefig(f2, dpi=300, bbox_inches="tight")
        plt.close(fig2)

        print(f"Saved combined plots: {f1} , {f2}")

    return 0


def cmd_predict(args: argparse.Namespace) -> int:
    bundle = load_model_bundle(args.model)
    off_az, off_el = predict_offsets_deg(
        bundle,
        az_deg=args.az,
        allow_extrapolation=args.allow_extrapolation
    )
    if args.unit == "arcmin":
        off_az = off_az * 60
        off_el = off_el * 60
    elif args.unit == "arcsec":
        off_az = off_az * 3600
        off_el = off_el * 3600
    else:
        pass

    output = (
        f"az={args.az:.4f}°  ->  offset_az={off_az:.4f} {args.unit}, "
        f"offset_el={off_el:.4f} {args.unit}"
    )
    print(output)
    return 0


def _plot_fit(
    ax,
    az_lin: np.ndarray,
    y: np.ndarray,
    model,
    keep_mask: np.ndarray,
    unit: str,
    label_y: str,
    bundle=None,
):
    """Plot scatter (outliers/inliers) and fitted curve, rendering in the
    requested unit."""
    # conversion factor for rendering (deg -> chosen unit)
    fac = _axis_factor_for_unit(unit)

    az_lin = np.asarray(az_lin)
    y = np.asarray(y)
    keep_mask = np.asarray(keep_mask).astype(bool)
    if keep_mask.shape != y.shape:
        raise ValueError("keep_mask must have same shape as y")

    # inliers / outliers masks
    m_in = keep_mask
    m_out = ~keep_mask

    res = (y - model(az_lin)) * fac

    lbl_in = f"inliers"
    lbl_out = f"outliers"

    # scatter points
    if np.any(m_out):
        ax.scatter(
            az_lin[m_out],
            y[m_out] * fac,
            s=24,
            alpha=0.35,
            label=lbl_out,
        )
    if np.any(m_in):
        ax.scatter(
            az_lin[m_in],
            y[m_in] * fac,
            s=24,
            alpha=0.85,
            label=lbl_in,
        )


    # fitted curve (label short, without params)
    xs = np.linspace(float(az_lin.min()), float(az_lin.max()), 600)
    ax.plot(xs, model(xs) * fac, linewidth=2.0, label=f"fit (MAD={_mad(res):.2g})")

    # axes
    ax.set_xlabel("az_lin (deg)")
    ax.set_ylabel(f"{label_y} ({unit})")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best")


def cmd_summary(args: argparse.Namespace) -> int:
    bundle = load_model_bundle(args.model)
    print(model_summary(bundle))
    return 0


# ----
# Main
# ----

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Azimuth-only pointing model with linearized azimuth",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    sub = p.add_subparsers(dest="command", required=True)

    # fit
    pf = sub.add_parser(
        "fit",
        help="Fit models from a TSV (offsets in degrees by default)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    pf.add_argument(
        "tsv",
        nargs="+",
        help="Input TSV path"
    )
    pf.add_argument(
        "--degree",
        type=int,
        default=3,
        help="Polynomial degree"
    )
    pf.add_argument(
        "--zscore-az",
        type=float,
        default=2.5,
        help="MAD-based z-score threshold for offset_az residuals"
    )
    pf.add_argument(
        "--zscore-el",
        type=float,
        default=2.5,
        help="MAD-based z-score threshold for offset_el residuals"
    )
    pf.add_argument(
        "--ridge-alpha",
        type=float,
        default=0.01,
        help="Ridge regularization strength"
    )
    pf.add_argument(
        "--fourier-k", type=int, default=0,
        help="Number of Fourier harmonics (k=1..K). 0 disables them."
    )
    pf.add_argument(
        "--periods-deg", default="",
        help="Comma-separated custom periods in degrees (e.g., 6,11.25)."
    )
    pf.add_argument(
        "--sector-edges-deg", default="",
        help="Comma-separated sector edges in degrees (e.g., 60,210)."
    )
    pf.add_argument(
        "--input-offset-unit",
        choices=["deg", "arcmin", "arcsec"],
        default="deg",
        help="Unit of offsets in the TSV")
    pf.add_argument(
        "--save-model",
        default=None,
        help="Output .joblib bundle path"
    )
    pf.add_argument("--summary",
        default=None,
        help="Optional path to write a textual summary"
    )
    pf.add_argument(
        "--notes",
        default=None,
        help="Optional free-form note saved into metadata"
    )
    pf.add_argument(
        "--plot-unit",
        choices=["deg", "arcmin", "arcsec"],
        default="arcmin",
        help="Rendering unit for auto-saved plots in `fit` (Y axis)"
    )
    pf.set_defaults(func=cmd_fit)


    # predict
    pp = sub.add_parser(
        "predict",
        help="Predict offsets at a given azimuth (degrees)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    pp.add_argument(
        "model",
        help="Path to .joblib bundle"
    )
    pp.add_argument(
        "--az",
        type=float,
        required=True,
        help="Azimuth in degrees (0..360)"
    )
    pp.add_argument(
        "--unit",
        choices=["deg", "arcmin", "arcsec"],
        default="arcmin",
        help="Offsets unit"
    )
    pp.add_argument(
        "--allow-extrapolation",
        action="store_true",
        help="Allow evaluation outside the observed linear azimuth range"
    )
    pp.set_defaults(func=cmd_predict)

    # summary
    ps = sub.add_parser(
        "summary",
        help="Print model summary",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    ps.add_argument(
        "model",
        help="Path to .joblib bundle"
    )
    ps.set_defaults(func=cmd_summary)

    return p


def main(argv=None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
