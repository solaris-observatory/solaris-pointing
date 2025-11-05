#!/usr/bin/env python3
"""Fit, predict and merge pointing-offset models (per-axis or unified).

This CLI provides a uniform interface to fit, predict, and merge telescope
pointing-offset models from TSV data files or serialized bundles (.joblib).
It supports independent per-axis models (AZ / EL) and automatically produces
a unified bundle combining both when available.

The tool is backend-agnostic (default backend: ``model_1d``). Each fitted model
creates:
- a .joblib model bundle for AZ and/or EL,
- a PNG plot for each axis,
- a human-readable text summary including a ready-to-copy Python function,
- and a JSON metadata sidecar (.meta.json).

-------------------------------------------------------------------------------
Available subcommands
-------------------------------------------------------------------------------
fit       Fit AZ/EL models from one or more TSV files (shared parameters).
predict   Predict AZ/EL offsets at a given azimuth from saved models.
merge     Merge per-axis models into a unified <stem>.joblib bundle.

-------------------------------------------------------------------------------
Command-line usage examples
-------------------------------------------------------------------------------
# --- FIT --------------------------------------------------------------------

1) Minimal fit (both AZ and EL, default parameters):
   python scripts/model_cli.py fit mydata.tsv
   # Outputs: models/mydata_az.joblib, models/mydata_el.joblib,
   #          plots and summaries per axis,
   #          plus models/mydata.joblib (unified bundle).

2) Fit both axes with custom parameters:
   python scripts/model_cli.py fit alpacino.tsv \
       --degree 3 --zscore 2.5 --fourier-k 2 --plot-unit arcmin

3) Fit AZ only:
   python scripts/model_cli.py fit alpacino.tsv --az \
       --degree 3 --zscore 2.5 --fourier-k 2

4) Fit EL only:
   python scripts/model_cli.py fit alpacino.tsv --el \
       --degree 3 --zscore 2.5 --fourier-k 1 --periods-deg 90,45

5) Fit multiple TSV files (combined plots are generated automatically):
   python scripts/model_cli.py fit new.tsv oranges.tsv \
       --degree 2 --zscore 2.0 --plot-unit arcmin
   # Produces combined plots: models/new+oranges_az.png and _el.png

6) Fit with input offsets in arcseconds:
   python scripts/model_cli.py fit data.tsv --input-offset-unit arcsec

# --- PREDICT ---------------------------------------------------------------

7) Predict both axes (no selector flags):
   python scripts/model_cli.py predict alpacino --azimuth 12.0 --unit arcsec
   # Loads: models/alpacino_az.joblib and models/alpacino_el.joblib

8) Predict AZ only:
   python scripts/model_cli.py predict alpacino --az \
       --azimuth 12.0 --unit arcsec

9) Predict EL only:
   python scripts/model_cli.py predict models/alpacino_el.joblib --el \
       --azimuth 12.0 --unit arcmin

10) Predict with extrapolation beyond observed azimuth range:
    python scripts/model_cli.py predict alpacino --az \
        --azimuth 355.0 --allow-extrapolation

# --- MERGE -----------------------------------------------------------------

11) Merge existing per-axis models into a unified bundle:
    python scripts/model_cli.py merge alpacino
    # Reads:  models/alpacino_az.joblib and models/alpacino_el.joblib
    # Writes: models/alpacino.joblib (+ metadata sidecar .meta.json)

-------------------------------------------------------------------------------
Input / output conventions
-------------------------------------------------------------------------------
- TSV inputs are looked up under ``offsets/`` if no directory is given.
- Models, plots, and summaries are written under ``models/``.
- Summaries include: text statistics, MAD_t/MAD_i in chosen unit,
  and a **Python function** defining the offset model.

-------------------------------------------------------------------------------
Unified parameters (shared by both axes)
-------------------------------------------------------------------------------
--degree (int)                  Polynomial degree.
--zscore (float)                Robust outlier threshold (MAD-based).
--ridge-alpha (float)           L2 regularization factor.
--fourier-k (int)               Number of Fourier harmonics (0 disables).
--periods-deg (list[str])       Custom Fourier periods, e.g. "6,11.25".
--sector-edges-deg (list[str])  Sector edges in degrees, e.g. "60,210".
--input-offset-unit (str)       Input unit of offsets (deg|arcmin|arcsec).
--plot-unit (str)               Y-axis unit in saved plots (deg|arcmin|arcsec).
--notes (str)                   Free text stored in metadata for traceability.
"""

from __future__ import annotations

import argparse
import os
import importlib
import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from typing import Any, Callable, Tuple, List


# --- Dynamic model backend loader -------------------------------------------

def _unbound_backend(*args: Any, **kwargs: Any):
    raise RuntimeError(
        "Backend not bound yet. Call _bind_backend(<kind>) before using backend APIs."
    )


# Placeholders so Ruff (F821) and static tools see defined names.
fit_models_from_tsv: Callable[..., Any] = _unbound_backend
load_model: Callable[..., Any] = _unbound_backend
save_model: Callable[..., Any] = _unbound_backend
predict_offsets_deg: Callable[..., Tuple[float, float]] = _unbound_backend
model_summary: Callable[..., str] = _unbound_backend
model_summary_axis: Callable[..., str] = _unbound_backend
read_offsets_tsv: Callable[..., "pd.DataFrame"] = _unbound_backend
unwrap_azimuth: Callable[..., Tuple["np.ndarray", "np.ndarray", float, float]] = (
    _unbound_backend
)
_mad: Callable[["np.ndarray"], float] = _unbound_backend

_REQUIRED_FUNS: List[str] = [
    "fit_models_from_tsv",
    "load_model",
    "save_model",
    "predict_offsets_deg",
    "model_summary",
    "model_summary_axis",
    "read_offsets_tsv",
    "unwrap_azimuth",
    "_mad",
]


def _import_backend(kind: str):
    """
    Import backend module `model_<kind>` either from installed package
    or local file. Validate required API.
    """
    candidates = [
        f"solaris_pointing.offsets.fitting.model_{kind}",
        f"model_{kind}",
    ]
    last_err = None
    for modname in candidates:
        try:
            mod = importlib.import_module(modname)
            missing = [n for n in _REQUIRED_FUNS if not hasattr(mod, n)]
            if missing:
                raise ImportError(
                    f"Backend {modname} missing symbols: {', '.join(missing)}"
                )
            return mod
        except Exception as e:
            last_err = e
    raise ImportError(f"Could not import model_{kind}: {last_err}")


def _bind_backend(kind: str):
    """
    Rebind backend functions into this module's globals.
    Returns the imported module as well (if you need direct access).
    """
    mod = _import_backend(kind)
    g = globals()
    for name in _REQUIRED_FUNS:
        g[name] = getattr(mod, name)
    return mod


# ---- Bundle metadata shims -----------------------------------------------

def _save_bundle_with_meta(bundle, path: str, backend_kind: str) -> None:
    """
    Save the bundle using the backend's native saver, and write the backend kind
    to a sidecar JSON file (<path>.meta.json). This avoids changing the bundle
    format and keeps backward compatibility.
    """
    # 1) write the original bundle exactly as the backend expects
    save_model(bundle, path)

    # 2) write sidecar metadata
    meta_path = path + ".meta.json"
    try:
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump({"backend_kind": backend_kind}, f)
    except Exception as e:
        # do not fail the fit if metadata write fails; just warn
        print(f"[WARN] Could not write metadata sidecar {meta_path}: {e}")


def _load_bundle_with_meta(path: str):
    """
    Load a saved model bundle together with its backend kind.

    Order of operations (fixed):
    1) Read the sidecar JSON (<path>.meta.json) to discover the backend kind.
       If missing (legacy bundles), default to "1d".
    2) Bind the backend (so that load_model is available).
    3) Load the actual bundle payload with the backend's native loader.
    4) Return (backend_kind, payload).

    This avoids calling load_model before binding the backend.
    """
    # 1) Discover backend kind from sidecar (if present)
    meta_path = path + ".meta.json"
    kind = "1d"
    try:
        if os.path.exists(meta_path):
            with open(meta_path, "r", encoding="utf-8") as f:
                meta = json.load(f)
            if isinstance(meta, dict) and "backend_kind" in meta:
                kind = str(meta["backend_kind"])
    except Exception as e:
        print(f"[WARN] Could not read metadata sidecar {meta_path}: {e}")

    # 2) Bind the backend before attempting to load the bundle
    _bind_backend(kind)

    # 3) Load the bundle payload using the backend's loader
    payload = load_model(path)

    # 4) Return (kind, payload)
    return kind, payload

# ------------------------------------------------------------------


def _stem(path: str) -> str:
    return os.path.splitext(os.path.basename(path))[0]


def _with_ext_same_dir(path: str, new_ext: str) -> str:
    d = os.path.dirname(path)
    s = os.path.splitext(os.path.basename(path))[0]
    return os.path.join(d or ".", s + new_ext)


def _ensure_dir(path: str) -> None:
    d = os.path.dirname(path)
    if d:
        os.makedirs(d, exist_ok=True)


def _parse_csv_floats(text: str):
    text = (text or "").strip()
    if not text:
        return None
    out = []
    for tok in text.split(","):
        tok = tok.strip()
        if not tok:
            continue
        try:
            out.append(float(tok))
        except Exception:
            raise ValueError(f"Invalid float in list: {tok!r}")
    return out if out else None


def _resolve_input_tsv(path: str) -> str:
    if os.path.isabs(path) or os.sep in path:
        return path
    return os.path.join("offsets", path)


def _default_models_dir() -> str:
    return "models"


def _model_path_for(stem: str, axis: str) -> str:
    # axis in {"az","el"}
    return os.path.join(_default_models_dir(), f"{stem}_{axis}.joblib")


def _summary_path_for(stem: str, axis: str) -> str:
    return os.path.join(_default_models_dir(), f"{stem}_summary_{axis}.txt")


def _plot_path_for(stem: str, axis: str) -> str:
    return os.path.join(_default_models_dir(), f"{stem}_{axis}.png")


def _axis_factor_for_unit(unit: str) -> float:
    if unit == "deg":
        return 1.0
    elif unit == "arcmin":
        return 60.0
    elif unit == "arcsec":
        return 3600.0
    else:
        raise ValueError(f"Unknown unit: {unit!r}")


def _mk_mask(res, thr):
    s = _mad(res)
    if s == 0.0:
        return np.ones_like(res, dtype=bool)
    return np.abs(res) <= thr * s


def _write_unified_bundle_and_summary(
    *,
    bundle,
    out_models_dir: str,
    stem: str,
    wrote_az: bool,
    wrote_el: bool,
    save_model_func,
    model_summary_func,
    model_summary_axis_func,
):
    """
    Write a 'unified' bundle alongside per-axis files.

    Rules:
    - If only AZ was fitted: unified .joblib == <stem>_az.joblib; summary = AZ only.
    - If only EL was fitted: unified .joblib == <stem>_el.joblib; summary = EL only.
    - If both were fitted:   unified .joblib contains both axes; summary = both axes.
    """
    out_models = Path(out_models_dir)
    out_models.mkdir(parents=True, exist_ok=True)

    unified_joblib = out_models / f"{stem}.joblib"
    unified_summary = out_models / f"{stem}_summary.txt"

    if wrote_az and not wrote_el:
        # Mirror AZ file into unified name
        src = out_models / f"{stem}_az.joblib"
        if src.exists():
            shutil.copy2(src, unified_joblib)
        # Summary: AZ only
        text = model_summary_axis_func(bundle, "az")
        unified_summary.write_text(text)
        return

    if wrote_el and not wrote_az:
        # Mirror EL file into unified name
        src = out_models / f"{stem}_el.joblib"
        if src.exists():
            shutil.copy2(src, unified_joblib)
        # Summary: EL only
        text = model_summary_axis_func(bundle, "el")
        unified_summary.write_text(text)
        return

    # Both axes were produced: save full bundle and full summary
    save_model_func(bundle, str(unified_joblib))
    text = model_summary_func(bundle)
    unified_summary.write_text(text)


# -----------
# Subcommands
# -----------
def cmd_fit(args: argparse.Namespace) -> int:
    """
    Fit AZ/EL models using a uniform-API backend. The backend is selected
    only for training via --model <kind> and stored in the saved bundles'
    metadata for later auto-selection by predict
    """
    # Determine which axis(es) to fit
    axes: list[str]
    if args.az:
        axes = ["az"]
    elif args.el:
        axes = ["el"]
    else:
        axes = ["az", "el"]

    # Bind backend chosen for training
    kind = getattr(args, "model_kind", "1d")
    _bind_backend(kind)

    raw_paths = args.tsv if isinstance(args.tsv, (list, tuple)) else [args.tsv]
    tsv_paths = [_resolve_input_tsv(p) for p in raw_paths]

    # Verify that all input files exist
    missing = [p for p in tsv_paths if not os.path.exists(p)]
    if missing:
        raise FileNotFoundError(
            f"The following input file(s) do not exist or are not accessible: "
            f"{', '.join(missing)}"
        )

    os.makedirs(_default_models_dir(), exist_ok=True)

    # Define directories for outputs
    models_dir = Path("models")
    plots_dir = Path("models")
    models_dir.mkdir(exist_ok=True)
    plots_dir.mkdir(exist_ok=True)

    bundles_info = []  # (stem, bundle, path, az_lin, off_az, off_el, period_info)

    for path in tsv_paths:
        work_path = path

        # Optional unit conversion to degrees
        if args.input_offset_unit != "deg":
            df = read_offsets_tsv(work_path).copy()
            to_deg = {"deg": 1.0, "arcmin": 1.0 / 60.0, "arcsec": 1.0 / 3600.0}[
                args.input_offset_unit
            ]
            df["offset_az"] = df["offset_az"].astype(float) * to_deg
            df["offset_el"] = df["offset_el"].astype(float) * to_deg
            tmp_path = work_path + ".deg.tmp.tsv"
            df.to_csv(tmp_path, sep="\t", index=False)
            work_path = tmp_path

        # Unified params are applied to the selected axis(es)
        bundle = fit_models_from_tsv(
            path=work_path,
            degree=args.degree,
            zscore=args.zscore,
            ridge_alpha=args.ridge_alpha,
            notes=args.notes,
            fourier_k=args.fourier_k,
            periods_deg=_parse_csv_floats(args.periods_deg),
            sector_edges_deg=_parse_csv_floats(args.sector_edges_deg),
            sector_edges_deg_az=_parse_csv_floats(args.sector_edges_deg),
            fourier_k_el=args.fourier_k,
            periods_deg_el=_parse_csv_floats(args.periods_deg),
            sector_edges_deg_el=_parse_csv_floats(args.sector_edges_deg),
        )

        stem = _stem(path)

        # Read back data for plotting & period detection
        dfp = read_offsets_tsv(work_path)
        az = dfp["azimuth"].to_numpy(float) % 360.0
        off_az = dfp["offset_az"].to_numpy(float)
        off_el = dfp["offset_el"].to_numpy(float)
        az_lin, cut, lo, hi = unwrap_azimuth(az)

        # --- Detect overall period info (for titles/summaries)
        period_info: tuple[str | None, str | None, bool, str] = (None, None, False, "")
        try:
            cols = [c.strip() for c in dfp.columns]
            if "timestamp" in cols:
                s = dfp["timestamp"].astype(str).str.strip()
                ts = pd.to_datetime(s, utc=True, errors="coerce")
                if ts.isna().all():
                    ts = pd.to_datetime(
                        s, format="%Y-%m-%dT%H:%M:%S.%fZ", utc=True, errors="coerce"
                    )
                if ts.isna().all():
                    ts = pd.to_datetime(
                        s, format="%Y-%m-%dT%H:%M:%SZ", utc=True, errors="coerce"
                    )
                if ts.isna().all():
                    ts = pd.to_datetime(
                        s.str.replace("Z", "+00:00", regex=False),
                        utc=True,
                        errors="coerce",
                    )
                ts = ts.dropna()
                if len(ts) > 0:
                    days = pd.to_datetime(ts.dt.normalize()).unique()
                    days = np.sort(days)
                    start_day = pd.to_datetime(days[0])
                    end_day = pd.to_datetime(days[-1])
                    expected_count = (end_day - start_day).days + 1
                    has_gaps = len(days) != expected_count
                    start_str = start_day.to_pydatetime().strftime("%y/%m/%d")
                    end_str = end_day.to_pydatetime().strftime("%y/%m/%d")
                    days_list_str = ", ".join(
                        pd.to_datetime(days).strftime("%y/%m/%d").tolist()
                    )
                    period_info = (start_str, end_str, has_gaps, days_list_str)
        except Exception:
            period_info = (None, None, False, "")
        # ---------------------------------------------------------------------

        # Compute robust masks for each axis (using unified zscore)
        yhat_az = bundle.az_model(az_lin)
        res_az = off_az - yhat_az
        yhat_el = bundle.el_model(az_lin)
        res_el = off_el - yhat_el

        m_az = _mk_mask(res_az, args.zscore)
        m_el = _mk_mask(res_el, args.zscore)

        # Titles / params for plots
        ps, pe, has_gaps, _days_str = period_info
        if ps and pe:
            gap_mark = " **" if has_gaps else ""
            top_line = f"{stem}: ({ps} -- {pe}){gap_mark}"
        else:
            top_line = stem

        def _write_axis(axis: str) -> None:
            """
            Write artifacts (model, summary, plot) for a single axis ('az' or 'el').
            """
            axis = axis.lower().strip()
            assert axis in ("az", "el")

            out_model = models_dir / f"{stem}_{axis}.joblib"
            out_summary = models_dir / f"{stem}_summary_{axis}.txt"
            out_plot = plots_dir / f"{stem}_{axis}.png"

            # Guard for model presence
            if axis == "az" and bundle.az_model is None:
                print("[WARN] No AZ model in bundle; skipping AZ outputs.")
                return
            if axis == "el" and bundle.el_model is None:
                print("[WARN] No EL model in bundle; skipping EL outputs.")
                return

            _save_bundle_with_meta(bundle, str(out_model), backend_kind=kind)
            print(f"Saved model: {out_model}")

            # Per-axis summary + MAD_t / MAD_i + Python function
            with open(out_summary, "w", encoding="utf-8") as f:
                txt = model_summary_axis(bundle, axis)

                # Residuals in requested plot-unit
                if axis == "az":
                    res_all = (
                        off_az - bundle.az_model(az_lin)
                    ) * _axis_factor_for_unit(args.plot_unit)
                    keep = m_az
                else:
                    res_all = (
                        off_el - bundle.el_model(az_lin)
                    ) * _axis_factor_for_unit(args.plot_unit)
                    keep = m_el

                # Compute MAD_t (all points) and MAD_i (inliers only)
                mad_t = _mad(res_all)
                mad_i = _mad(res_all[keep])

                # Append to summary
                txt += f"\nMAD_t ({args.plot_unit}): {mad_t:.3f}\n"
                txt += f"MAD_i ({args.plot_unit}): {mad_i:.3f}\n"

                # --- Append Python function representation ----------
                func_txt = ""
                try:
                    model = bundle.az_model if axis == "az" else bundle.el_model
                    meta = bundle.meta  # includes cut_deg, etc.

                    lines = []
                    lines.append(
                        f"\n# Python function for {axis.upper()} "
                        f"offset (degrees in --> degrees out)"
                    )
                    lines.append(f"def {axis}_offset(az):")
                    lines.append(
                        '    """Return %s offset in degrees. '
                        'Input az in degrees [0..360) or any real."""' % axis.upper()
                    )
                    lines.append("    from math import sin, cos, radians")
                    lines.append("")
                    lines.append(
                        "    # Linearize azimuth using the stored cut from the fit"
                    )
                    lines.append(f"    CUT = {getattr(meta, 'cut_deg', 0.0):.6f}")
                    lines.append("    a = az % 360.0")
                    lines.append(
                        "    x = a + 360.0 if a < CUT else a  # x = az_lin (deg)"
                    )
                    lines.append("")
                    lines.append("    # Polynomial (in x = az_lin)")
                    lines.append("    y = (")

                    # ---- polynomial part ----
                    coef = np.asarray(getattr(model, "coef", []), dtype=float)
                    d = int(getattr(model, "poly_degree", len(coef) - 1))
                    base = 0
                    poly_lines = []
                    for i in range(d + 1):
                        c = float(coef[base + i]) if base + i < coef.size else 0.0
                        if abs(c) <= 1e-15:
                            continue
                        if i == 0:
                            poly_lines.append(f"        {c:+.9e}")
                        elif i == 1:
                            poly_lines.append(f"        {c:+.9e} * x")
                        else:
                            poly_lines.append(f"        {c:+.9e} * x**{i}")
                    if not poly_lines:
                        poly_lines.append("        0.0")
                    lines.extend(poly_lines)
                    lines.append("    )")
                    lines.append("")

                    # ---- Fourier terms ----
                    kmax = int(getattr(model, "fourier_k", 0))
                    base = d + 1
                    if kmax > 0:
                        lines.append("    # Fourier k/rev terms (in x = az_lin)")
                        for k in range(1, kmax + 1):
                            Ak = (
                                float(coef[base + 2 * (k - 1) + 0])
                                if base + 2 * (k - 1) + 0 < coef.size
                                else 0.0
                            )
                            Bk = (
                                float(coef[base + 2 * (k - 1) + 1])
                                if base + 2 * (k - 1) + 1 < coef.size
                                else 0.0
                            )
                            if abs(Ak) > 1e-15:
                                lines.append(
                                    f"    y += {Ak:+.9e} * cos(radians({k} * x))"
                                )
                            if abs(Bk) > 1e-15:
                                lines.append(
                                    f"    y += {Bk:+.9e} * sin(radians({k} * x))"
                                )
                        lines.append("")

                    # ---- custom periods ----
                    periods = list(getattr(model, "periods_deg", []) or [])
                    idx = d + 1 + 2 * kmax
                    if periods:
                        lines.append("    # Custom-period terms")
                        for P in periods:
                            C = float(coef[idx]) if idx < coef.size else 0.0
                            idx += 1
                            S = float(coef[idx]) if idx < coef.size else 0.0
                            idx += 1
                            if abs(C) > 1e-15:
                                lines.append(
                                    f"    y += {C:+.9e} * cos((3.141592653589793*2) "
                                    f"* x / {float(P):.6f})"
                                )
                            if abs(S) > 1e-15:
                                lines.append(
                                    f"    y += {S:+.9e} * sin((3.141592653589793*2) "
                                    f"* x / {float(P):.6f})"
                                )
                        lines.append("")

                    lines.append("    return y")
                    func_txt = "\n" + "\n".join(lines) + "\n"

                except Exception as e:
                    func_txt = f"\n# [WARN] Could not generate Python function: {e}\n"

                # Write summary text + nicely formatted Python function
                f.write(txt + func_txt)
                # ----------------------------------------------------------------

            # --- Per-axis param line for the single-plot title
            meta = bundle.meta

            def _axis_meta(m, ax: str, name: str, default=None):
                return getattr(m, f"{name}_{ax}", getattr(m, name, default))

            deg = _axis_meta(meta, axis, "degree", getattr(meta, "degree", None))
            alpha = _axis_meta(
                meta, axis, "ridge_alpha", getattr(meta, "ridge_alpha", None)
            )
            k_ax = _axis_meta(meta, axis, "fourier_k", getattr(meta, "fourier_k", 0))

            def _fmt_alpha(a):
                return f"{a:g}" if isinstance(a, (int, float)) else "n/a"

            def _fmt_int(x):
                return f"{int(x)}" if isinstance(x, (int, float)) else "n/a"

            params_line_axis = (
                f"degree={_fmt_int(deg)}, α={_fmt_alpha(alpha)}, "
                f"z={args.zscore:g}, f={_fmt_int(k_ax)}"
            )
            # ------------------------------------------------------------------

            # Plot
            if args.plot_unit:
                unit = args.plot_unit.lower()
                if unit not in ("deg", "arcmin", "arcsec"):
                    raise ValueError("--plot-unit must be 'deg', 'arcmin', or 'arcsec'")

                fig, axp = plt.subplots(figsize=(7, 4))

                if axis == "az":
                    _plot_fit(
                        ax=axp,
                        az_lin=az_lin,
                        y=off_az,
                        model=bundle.az_model,
                        keep_mask=m_az,
                        unit=unit,
                        label_y="offset_az",
                        bundle=bundle,
                    )
                else:
                    _plot_fit(
                        ax=axp,
                        az_lin=az_lin,
                        y=off_el,
                        model=bundle.el_model,
                        keep_mask=m_el,
                        unit=unit,
                        label_y="offset_el",
                        bundle=bundle,
                    )

                fig.subplots_adjust(top=0.86, bottom=0.13)
                fig.suptitle(top_line, fontsize=8, y=0.97)
                fig.text(
                    0.5, 0.930, params_line_axis, ha="center", va="top", fontsize=8
                )

                fig.savefig(out_plot, dpi=300, bbox_inches="tight")
                plt.close(fig)
                print(f"Saved plot: {out_plot}")

        wrote_az = wrote_el = False
        if "az" in axes:
            _write_axis("az")
            wrote_az = True
        if "el" in axes:
            _write_axis("el")
            wrote_el = True

        bundles_info.append((stem, bundle, path, az_lin, off_az, off_el, period_info))
        # --- create unified artifacts ONLY when both axes are fitted together ---
        if not (args.az or args.el):
            _write_unified_bundle_and_summary(
                bundle=bundle,
                out_models_dir=models_dir,
                stem=stem,
                wrote_az=wrote_az,
                wrote_el=wrote_el,
                save_model_func=save_model,
                model_summary_func=model_summary,
                model_summary_axis_func=model_summary_axis,
            )

        if work_path.endswith(".deg.tmp.tsv") and os.path.exists(work_path):
            os.remove(work_path)

    # ------------------------------------------------------------------
    # Combined curves-only plot (multi-file, per-axis aware)
    # ------------------------------------------------------------------
    if len(tsv_paths) > 1:
        # Use bundles_info accumulated during fitting
        bundles_sorted = sorted(bundles_info, key=lambda t: t[0])  # sort by stem
        stems_sorted = [t[0] for t in bundles_sorted]
        name_base = "+".join(stems_sorted)

        base_path = os.path.join(_default_models_dir(), name_base + ".png")
        root, ext = os.path.splitext(base_path)
        f1 = f"{root}_az{ext}"
        f2 = f"{root}_el{ext}"

        fac = _axis_factor_for_unit(args.plot_unit)
        params_line = (
            f"degree={args.degree}, α={args.ridge_alpha:g}, "
            f"z={args.zscore:g}, f={args.fourier_k:g}"
        )
        top_title = "  +  ".join(stems_sorted)

        # ---- AZ combined ----
        if not args.el:  # plot AZ if --el not exclusive
            fig1, ax1 = plt.subplots(figsize=(7, 4))
            for (
                stem,
                bundle,
                path,
                az_lin,
                off_az,
                off_el,
                _period_info,
            ) in bundles_sorted:
                if bundle.az_model is None:
                    continue
                yhat = bundle.az_model(az_lin)
                res = (off_az - yhat) * fac
                lbl = f"{stem} (MAD={_mad(res):.2g})"
                xs = np.linspace(float(az_lin.min()), float(az_lin.max()), 600)
                ax1.plot(xs, bundle.az_model(xs) * fac, linewidth=2.0, label=lbl)

            if len(ax1.lines) > 0:
                ax1.set_xlabel("az_lin (deg)")
                ax1.set_ylabel(f"offset_az ({args.plot_unit})")
                ax1.grid(True, alpha=0.25)
                ax1.legend()
                fig1.subplots_adjust(top=0.86, bottom=0.13)
                fig1.suptitle(top_title, fontsize=8, y=0.97)
                fig1.text(0.5, 0.930, params_line, ha="center", va="top", fontsize=8)
                fig1.savefig(f1, dpi=300, bbox_inches="tight")
                plt.close(fig1)
                print(f"Saved combined plot: {f1}")
            else:
                plt.close(fig1)

        # ---- EL combined ----
        if not args.az:  # plot EL if --az not exclusive
            fig2, ax2 = plt.subplots(figsize=(7, 4))
            for (
                stem,
                bundle,
                path,
                az_lin,
                off_az,
                off_el,
                _period_info,
            ) in bundles_sorted:
                if bundle.el_model is None:
                    continue
                yhat = bundle.el_model(az_lin)
                res = (off_el - yhat) * fac
                lbl = f"{stem} (MAD={_mad(res):.2g})"
                xs = np.linspace(float(az_lin.min()), float(az_lin.max()), 600)
                ax2.plot(xs, bundle.el_model(xs) * fac, linewidth=2.0, label=lbl)

            if len(ax2.lines) > 0:
                ax2.set_xlabel("az_lin (deg)")
                ax2.set_ylabel(f"offset_el ({args.plot_unit})")
                ax2.grid(True, alpha=0.25)
                ax2.legend()
                fig2.subplots_adjust(top=0.86, bottom=0.13)
                fig2.suptitle(top_title, fontsize=8, y=0.97)
                fig2.text(0.5, 0.930, params_line, ha="center", va="top", fontsize=8)
                fig2.savefig(f2, dpi=300, bbox_inches="tight")
                plt.close(fig2)
                print(f"Saved combined plot: {f2}")
            else:
                plt.close(fig2)

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
    fac = _axis_factor_for_unit(unit)

    az_lin = np.asarray(az_lin)
    y = np.asarray(y)
    keep_mask = np.asarray(keep_mask).astype(bool)
    if keep_mask.shape != y.shape:
        raise ValueError("keep_mask must have same shape as y")

    m_in = keep_mask
    m_out = ~keep_mask

    res = (y - model(az_lin)) * fac

    if np.any(m_out):
        ax.scatter(az_lin[m_out], y[m_out] * fac, s=24, alpha=0.35, label="outliers")
    if np.any(m_in):
        ax.scatter(az_lin[m_in], y[m_in] * fac, s=24, alpha=0.85, label="inliers")

    xs = np.linspace(float(az_lin.min()), float(az_lin.max()), 600)
    ax.plot(
        xs,
        model(xs) * fac,
        linewidth=2.0,
        label=f"fit (MAD={_mad(res):.2g})",  # MAD_t (total -> all points)
    )

    ax.set_xlabel("az_lin (deg)")
    ax.set_ylabel(f"{label_y} ({unit})")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best")


def _resolve_model_path_for_predict(user_path: str, axis: str | None) -> str:
    """
    Resolve the model path for predict, honoring the default 'models/' directory
    when the user does not provide an explicit directory.

    Rules:
    - If 'user_path' contains a directory separator, use it as-is (only ensure
      the proper axis suffix and .joblib when needed).
    - If 'user_path' has no directory, prepend 'models/'.
    - If 'axis' is provided ('az' or 'el'):
        * If basename already ends with '_az'/'_el', keep it and only ensure .joblib.
        * Otherwise, append f'_{axis}' and ensure .joblib.
    - If 'axis' is None, just ensure .joblib if missing (used rarely here).
    """
    base = user_path
    has_dir = (os.sep in base) or (os.altsep and os.altsep in base)

    # Prepend default directory when none is provided
    if not has_dir:
        base = os.path.join("models", base)

    b, e = os.path.splitext(base)
    name = os.path.basename(b)

    def ensure_joblib(p: str) -> str:
        bb, ee = os.path.splitext(p)
        return p if ee else (bb + ".joblib")

    if axis in ("az", "el"):
        if name.endswith("_az") or name.endswith("_el"):
            # User already provided an axis-suffixed basename; just ensure .joblib
            return ensure_joblib(base + e)
        # Append the requested axis and ensure .joblib
        return ensure_joblib(f"{b}_{axis}{e if e else ''}")

    # No axis provided: just ensure .joblib
    return ensure_joblib(base + e)


def cmd_predict(args: argparse.Namespace) -> int:
    """
    Predict offsets at a given azimuth. The correct backend is auto-selected
    by reading the bundle metadata. Backward compatible with legacy bundles
    (assumed '1d').
    """
    unit = args.unit

    if args.az and args.el:
        raise SystemExit("--az and --el are mutually exclusive.")

    # Helper: return (payload, path) choosing the first existing path
    def _load_payload_first_ok(paths: list[str]):
        for p in paths:
            if os.path.exists(p):
                kind, payload = _load_bundle_with_meta(p)
                _bind_backend(kind)
                return payload, p
        # Fallback: attempt the first anyway
        kind, payload = _load_bundle_with_meta(paths[0])
        _bind_backend(kind)
        return payload, paths[0]

    # Helper: derive twin paths if user passes *_az.joblib or *_el.joblib
    def _twins_for(path: str) -> tuple[str, str]:
        """
        Return the twin AZ/EL model paths derived from the given base path.

        Rules:
        - If the user provides a path ending with '_az' or '_el', derive the
          twin by swapping the suffix.
        - If the user provides a bare base name (with or without directory),
          automatically append '_az.joblib' and '_el.joblib'.
        - If no directory is specified, the default search directory is 'models/'.

        Examples:
        _twins_for("models/alpacino")
            -> ("models/alpacino_az.joblib", "models/alpacino_el.joblib")
        _twins_for("alpacino")
            -> ("models/alpacino_az.joblib", "models/alpacino_el.joblib")
        _twins_for("foo_el.joblib")
            -> ("foo_az.joblib", "foo_el.joblib")
        """
        base, ext = os.path.splitext(path)

        # If no directory is provided, assume 'models/' as default search directory
        if os.sep not in base and not base.startswith("models" + os.sep):
            base = os.path.join("models", base)

        def _ensure_joblib(p: str) -> str:
            """Append .joblib extension if missing."""
            b, e = os.path.splitext(p)
            return p if e else (b + ".joblib")

        # If the user passed a model ending in _az/_el, derive the
        # twin by swapping suffixes
        if base.endswith("_az"):
            return _ensure_joblib(base + ext), _ensure_joblib(base[:-3] + "_el" + ext)
        if base.endswith("_el"):
            return _ensure_joblib(base[:-3] + "_az" + ext), _ensure_joblib(base + ext)

        # No selector provided: create both AZ/EL paths, ensuring .joblib extension
        if not ext:
            return base + "_az.joblib", base + "_el.joblib"
        return base + "_az" + ext, base + "_el" + ext

    if args.az:
        # Resolve path to the AZ model under 'models/' when no directory is given
        model_path = _resolve_model_path_for_predict(args.model_path, axis="az")
        kind, payload = _load_bundle_with_meta(model_path)
        off_az, off_el = predict_offsets_deg(
            payload, az_deg=args.azimuth, allow_extrapolation=args.allow_extrapolation
        )
        unit = args.unit
        if unit == "arcmin":
            off_az *= 60
        elif unit == "arcsec":
            off_az *= 3600
        print(f"[AZ] az={args.azimuth:.4f}°  ->  offset_az={off_az:.4f} {unit}")
        return 0

    if args.el:
        # Resolve path to the EL model under 'models/' when no directory is given
        model_path = _resolve_model_path_for_predict(args.model_path, axis="el")
        kind, payload = _load_bundle_with_meta(model_path)
        off_az, off_el = predict_offsets_deg(
            payload, az_deg=args.azimuth, allow_extrapolation=args.allow_extrapolation
        )
        unit = args.unit
        if unit == "arcmin":
            off_el *= 60
        elif unit == "arcsec":
            off_el *= 3600
        print(f"[EL] az={args.azimuth:.4f}°  ->  offset_el={off_el:.4f} {unit}")
        return 0

    # No axis selector: print BOTH. Try *_az and *_el twins first.
    az_path, el_path = _twins_for(args.model_path)

    # Load AZ and EL payloads (auto-bind each one's backend before use)
    payload_az, _ = _load_payload_first_ok([az_path, args.model_path])
    payload_el, _ = _load_payload_first_ok([el_path, args.model_path])

    off_az_AZ, _ = predict_offsets_deg(
        payload_az, az_deg=args.azimuth, allow_extrapolation=args.allow_extrapolation
    )
    _, off_el_EL = predict_offsets_deg(
        payload_el, az_deg=args.azimuth, allow_extrapolation=args.allow_extrapolation
    )

    if unit == "arcmin":
        off_az_AZ *= 60
        off_el_EL *= 60
    elif unit == "arcsec":
        off_az_AZ *= 3600
        off_el_EL *= 3600

    print(
        f"az={args.azimuth:.4f}°  ->  "
        f"[AZ] offset_az={off_az_AZ:.4f} {unit}   "
        f"[EL] offset_el={off_el_EL:.4f} {unit}"
    )
    return 0


def cmd_merge(args: argparse.Namespace) -> int:
    """
    Merge two per-axis bundles (AZ and EL) into a unified bundle saved
    as models/{stem}.joblib.

    Rules:
    - Load models/{stem}_az.joblib and models/{stem}_el.joblib
      (auto-binding backends via sidecar).
    - Require same backend kind.
    - Use AZ payload as base; set its 'el_model' from EL payload.
    - Save unified bundle as models/{stem}.joblib (+ sidecar metadata).
    """
    stem = args.stem.strip()
    az_path = _model_path_for(stem, "az")
    el_path = _model_path_for(stem, "el")

    if not os.path.exists(az_path):
        raise SystemExit(f"Missing AZ model: {az_path}")
    if not os.path.exists(el_path):
        raise SystemExit(f"Missing EL model: {el_path}")

    kind_az, payload_az = _load_bundle_with_meta(az_path)
    kind_el, payload_el = _load_bundle_with_meta(el_path)

    if kind_az != kind_el:
        raise SystemExit(
            f"Backend mismatch between AZ ({kind_az}) and EL ({kind_el}). "
            "Re-fit or re-save with the same backend."
        )

    # Compose unified bundle:
    base = payload_az
    # Ensure attributes exist; fall back to error if missing
    if not hasattr(base, "az_model"):
        raise SystemExit("AZ payload has no 'az_model' attribute.")
    if not hasattr(payload_el, "el_model"):
        raise SystemExit("EL payload has no 'el_model' attribute.")

    setattr(base, "el_model", getattr(payload_el, "el_model"))

    # Save unified
    out_path = os.path.join(_default_models_dir(), f"{stem}.joblib")
    _save_bundle_with_meta(base, out_path, backend_kind=kind_az)
    print(f"Saved unified model: {out_path}")
    return 0


# -------------
# Main / Parser
# -------------
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Pointing model CLI with per-axis selection and unified params",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    sub = p.add_subparsers(dest="command", required=True)

    # fit
    pf = sub.add_parser(
        "fit",
        help="Fit models from TSV (choose --az/--el or both by default)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    pf.add_argument("tsv", nargs="+", help="Input TSV path(s)")
    # Axis selector (mutually exclusive)
    g_axis = pf.add_mutually_exclusive_group()
    g_axis.add_argument("--az", action="store_true", help="Fit azimuth only")
    g_axis.add_argument("--el", action="store_true", help="Fit elevation only")

    # Backend kind ONLY for fit (uniform API across modules)
    pf.add_argument(
        "--model",
        dest="model_kind",
        default="1d",
        help="Backend kind to train with: model_<kind> (e.g., 1d, 2d, foo).",
    )

    # Unified parameters
    pf.add_argument("--degree", type=int, default=3, help="Polynomial degree")
    pf.add_argument(
        "--zscore",
        type=float,
        default=2.5,
        help="MAD-based z-score threshold (unified for both axes)",
    )
    pf.add_argument(
        "--ridge-alpha", type=float, default=0.01, help="Ridge regularization strength"
    )
    pf.add_argument(
        "--fourier-k",
        type=int,
        default=0,
        help="# of Fourier harmonics (k=1..K). 0 disables.",
    )
    pf.add_argument(
        "--periods-deg",
        default="",
        help="Custom periods (deg), comma-separated (e.g., 6,11.25)",
    )
    pf.add_argument(
        "--sector-edges-deg",
        default="",
        help="Sector edges (deg), comma-separated (e.g., 60,210)",
    )

    pf.add_argument(
        "--input-offset-unit",
        choices=["deg", "arcmin", "arcsec"],
        default="deg",
        help="Unit of offsets in the TSV",
    )
    pf.add_argument("--notes", default=None, help="Note saved into metadata")
    pf.add_argument(
        "--plot-unit",
        choices=["deg", "arcmin", "arcsec"],
        default="arcmin",
        help="Rendering unit for saved plots (Y axis)",
    )

    pf.set_defaults(func=cmd_fit)

    # predict
    pp = sub.add_parser(
        "predict",
        help="Predict offsets at a given azimuth (choose --az/--el or both by default)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    pp.add_argument("model_path", help="Path to model base or *_az/_el .joblib")
    g_axis_p = pp.add_mutually_exclusive_group()
    g_axis_p.add_argument("--az", action="store_true", help="Use azimuth model only")
    g_axis_p.add_argument("--el", action="store_true", help="Use elevation model only")
    pp.add_argument(
        "--azimuth",
        dest="azimuth",
        type=float,
        required=True,
        help="Azimuth in degrees (0..360)",
    )
    pp.add_argument(
        "--unit",
        choices=["deg", "arcmin", "arcsec"],
        default="arcmin",
        help="Output offsets unit",
    )
    pp.add_argument(
        "--allow-extrapolation",
        action="store_true",
        help="Allow evaluation outside the observed linear azimuth range",
    )
    pp.set_defaults(func=cmd_predict)

    # merge
    pm = sub.add_parser(
        "merge",
        help="Merge {stem}_az.joblib and {stem}_el.joblib into {stem}.joblib",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    pm.add_argument("stem", help="Base stem of the model files under 'models/'")
    pm.set_defaults(func=cmd_merge)

    return p


def main(argv=None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
