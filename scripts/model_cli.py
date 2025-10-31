#!/usr/bin/env python3
"""Fit pointing offsets with per-axis selection (az or el) and unified params.

This CLI wraps the core modeling library to fit, summarize, and predict telescope
pointing offsets directly from TSV input files or saved model bundles (.joblib).

-------------------------------------------------------------------------------
Available subcommands
-------------------------------------------------------------------------------
fit       Fit az/el models from one or more TSV input files.
predict   Predict az/el offsets for a given azimuth using saved model(s).
summary   Print model diagnostics and metadata.

-------------------------------------------------------------------------------
Command-line usage examples
-------------------------------------------------------------------------------
1) Minimal default fit (both AZ and EL using default parameters):
   python scripts/model_cli.py fit mydata.tsv
   # Output models: models/mydata_az.joblib and models/mydata_el.joblib
   # Plots and summaries are saved in the same directory.

2) Fit both axes (default, no selector), unified params:
   python scripts/model_cli.py fit alpacino.tsv \
       --degree 3 --zscore 2.5 --fourier-k 2 --plot-unit arcmin

3) Fit AZIMUTH only:
   python scripts/model_cli.py fit alpacino.tsv --az \
       --degree 3 --zscore 2.5 --fourier-k 2 --sector-edges-deg 360

4) Fit ELEVATION only:
   python scripts/model_cli.py fit alpacino.tsv --el \
       --degree 3 --zscore 2.5 --fourier-k 1 --periods-deg 90,45

5) Predict (both axes if no selector):
   python scripts/model_cli.py predict models/alpacino \
       --az 12.0 --unit arcsec
   # This will try models/alpacino_az.joblib and models/alpacino_el.joblib.

   Only az:
   python scripts/model_cli.py predict models/alpacino_az.joblib --az \
       --az 12.0 --unit arcsec

-------------------------------------------------------------------------------
Input and output conventions
-------------------------------------------------------------------------------
- Input TSV files are searched under `offsets/` if no directory component is
  given in the path. If a directory is included, the file is taken as-is.
- Default output directory for models and plots is `models/`.

-------------------------------------------------------------------------------
Parameters and their physical meaning (unified)
-------------------------------------------------------------------------------
- `degree` (int): polynomial degree for the fit (1=linear, 2=quadratic, ...).
- `zscore` (float): robust MAD-based threshold for outlier rejection.
- `ridge-alpha` (float): L2 regularization strength (default 0.01).
- `fourier-k` (int): number of Fourier harmonics (0 disables).
- `periods-deg` (str): comma-separated list of custom periods (degrees).
- `sector-edges-deg` (str): comma-separated list of sector edges (degrees).
- `input-offset-unit` (deg|arcmin|arcsec): unit for offset columns in TSV.
- `plot-unit` (deg|arcmin|arcsec): rendering unit for plots (Y axis).
- `notes` (str): free text stored in the model metadata for traceability.

"""

from __future__ import annotations

import argparse
import os
import importlib
import json
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
load_model_bundle: Callable[..., Any] = _unbound_backend
save_model_bundle: Callable[..., Any] = _unbound_backend
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
    "load_model_bundle",
    "save_model_bundle",
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
    save_model_bundle(bundle, path)

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
    Load the bundle using the backend's native loader. Then read the sidecar
    JSON (<path>.meta.json) to get the backend kind. If the sidecar is missing
    (legacy bundles), default to '1d'.
    Returns: (backend_kind, payload)
    """
    # 1) load the real bundle (payload) first
    payload = load_model_bundle(path)

    # 2) read sidecar metadata if available
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


def _mad_local(x: np.ndarray) -> float:
    med = np.median(x)
    m = np.median(np.abs(x - med))
    return 1.4826 * m if m > 0 else 0.0


def _mk_mask(res, thr):
    s = _mad_local(res)
    if s == 0.0:
        return np.ones_like(res, dtype=bool)
    return np.abs(res) <= thr * s


# -----------
# Subcommands
# -----------
def cmd_fit(args: argparse.Namespace) -> int:
    """
    Fit AZ/EL models using a uniform-API backend. The backend is selected
    only for training via --model <kind> and stored in the saved bundles'
    metadata for later auto-selection by predict/summary.
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
            out_summary = models_dir / f"{stem}_{axis}_summary.txt"
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

            # Per-axis summary + MAD
            with open(out_summary, "w", encoding="utf-8") as f:
                txt = model_summary_axis(bundle, axis)
                if axis == "az":
                    res = (off_az - bundle.az_model(az_lin)) * _axis_factor_for_unit(
                        args.plot_unit
                    )
                else:
                    res = (off_el - bundle.el_model(az_lin)) * _axis_factor_for_unit(
                        args.plot_unit
                    )
                mad_val = _mad(res)
                txt += f"\nMAD ({args.plot_unit}): {mad_val:.3f}\n"
                f.write(txt)

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

        if "az" in axes:
            _write_axis("az")
        if "el" in axes:
            _write_axis("el")

        bundles_info.append((stem, bundle, path, az_lin, off_az, off_el, period_info))

        if work_path.endswith(".deg.tmp.tsv") and os.path.exists(work_path):
            os.remove(work_path)

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
    ax.plot(xs, model(xs) * fac, linewidth=2.0, label=f"fit (MAD={_mad(res):.2g})")

    ax.set_xlabel("az_lin (deg)")
    ax.set_ylabel(f"{label_y} ({unit})")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best")


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
        base, ext = os.path.splitext(path)
        if base.endswith("_az"):
            return path, base[:-3] + "_el" + ext
        if base.endswith("_el"):
            return base[:-3] + "_az" + ext, path
        return base + "_az" + ext, base + "_el" + ext

    if args.az:
        # Prefer exact path; else try suffix _az
        model_path = args.model_path
        if not (os.path.exists(model_path) and model_path.endswith(".joblib")):
            base, ext = os.path.splitext(model_path)
            cand = base + "_az" + (ext if ext else ".joblib")
            if os.path.exists(cand):
                model_path = cand
        kind, payload = _load_bundle_with_meta(model_path)
        _bind_backend(kind)
        off_az, off_el = predict_offsets_deg(
            payload, az_deg=args.azimuth, allow_extrapolation=args.allow_extrapolation
        )
        if unit == "arcmin":
            off_az *= 60
        elif unit == "arcsec":
            off_az *= 3600
        print(f"[AZ] az={args.azimuth:.4f}°  ->  offset_az={off_az:.4f} {unit}")
        return 0

    if args.el:
        # Prefer exact path; else try suffix _el
        model_path = args.model_path
        if not (os.path.exists(model_path) and model_path.endswith(".joblib")):
            base, ext = os.path.splitext(model_path)
            cand = base + "_el" + (ext if ext else ".joblib")
            if os.path.exists(cand):
                model_path = cand
        kind, payload = _load_bundle_with_meta(model_path)
        _bind_backend(kind)
        off_az, off_el = predict_offsets_deg(
            payload, az_deg=args.azimuth, allow_extrapolation=args.allow_extrapolation
        )
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


def cmd_summary(args: argparse.Namespace) -> int:
    """
    Print model summary. Auto-select the proper backend from bundle metadata.
    Backward compatible with legacy bundles (assumed '1d').
    """
    kind, payload = _load_bundle_with_meta(args.model_path)
    _bind_backend(kind)
    print(model_summary(payload))
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
        help="Fit models from TSV (unified params; choose --az/--el or both by default)",
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

    # summary
    ps = sub.add_parser(
        "summary",
        help="Print model summary",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ps.add_argument("model_path", help="Path to .joblib bundle")
    ps.set_defaults(func=cmd_summary)

    return p


def main(argv=None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
