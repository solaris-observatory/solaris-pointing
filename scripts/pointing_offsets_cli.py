
from __future__ import annotations

import argparse
import os
from typing import Dict, Any, Tuple, Iterable
from datetime import datetime, timezone
from types import SimpleNamespace

from solaris_pointing.offset_core.config_loader import (
    load_run_config,
    dump_effective_config,
)
from solaris_pointing.offset_core.discovery import discover_maps
from solaris_pointing.offset_core.model import MapInput, Site, Config
from solaris_pointing.offset_core.environment_reader import read_environment_tsv
from solaris_pointing.offset_core.offset_io import (
    Metadata,
    Measurement,
    write_offsets_tsv,
)
import importlib


def _load_algo_module(cfg: Dict[str, Any]):
    name = cfg.get("algo", {}).get("name", "gauss2d")
    modname = f"solaris_pointing.offset_algorithms.algo_{name}"
    try:
        return importlib.import_module(modname)
    except Exception as e:
        raise RuntimeError(f"Cannot import algorithm module: {modname}")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="pointing_offsets_cli",
        description="Compute per-map offsets.",
    )
    p.add_argument("--run", help="Run name, resolves to config/runs/<name>.toml")
    p.add_argument("--run-config", help="Explicit run file path (TOML)")
    p.add_argument(
        "--set",
        action="append",
        default=[],
        help="Override config key=value (repeatable).",
    )
    p.add_argument(
        "--dump-effective-config",
        action="store_true",
        help="Print final merged config and exit.",
    )
    p.add_argument(
        "--log-dir",
        default="logs",
        help="Directory where the run log will be created.",
    )
    return p


def _init_logger(project_root: str, log_dir: str) -> Tuple[str, Any]:
    os.makedirs(os.path.join(project_root, log_dir), exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%SZ")
    path = os.path.join(project_root, log_dir, f"run_{stamp}.log")

    def _log(msg: str) -> None:
        with open(path, "a", encoding="utf-8") as f:
            f.write(msg.rstrip() + "\n")

    return path, _log


def _log_header(log, project_root: str, paths: Dict[str, Any], cfg: Dict[str, Any]):
    now = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    log(f"[{now}] Run started")
    log(f"Project root: {project_root}")
    if paths.get("run_path"):
        rel = os.path.relpath(paths["run_path"], project_root)
        log(f"Run config: {rel}")
    if paths.get("site_path"):
        rel = os.path.relpath(paths["site_path"], project_root)
        log(f"Site config: {rel}")
    if paths.get("algo_path"):
        rel = os.path.relpath(paths["algo_path"], project_root)
        log(f"Algo config: {rel}")
    log("")
    log("----- Effective configuration -----")
    log(dump_effective_config(cfg).rstrip())
    log("-----------------------------------")
    log("")


def _load_env_lookup(cfg: Dict[str, Any], project_root: str) -> Dict[str, Dict[str, Any]]:
    inp = cfg.get("input", {})
    data_dir = inp.get("data_dir", "data")
    env_file = inp.get("environment_file", "environment.tsv")
    env_path = os.path.join(project_root, data_dir, env_file)
    env_cfg = cfg.get("environment", {})
    if not os.path.exists(env_path):
        return {}
    env_map = read_environment_tsv(
        env_path,
        ts_col=env_cfg.get("ts_column", "timestamp"),
        p_col=env_cfg.get("pressure_column", "pressure_hPa"),
        t_col=env_cfg.get("temperature_column", "temperature_C"),
        tau_col=env_cfg.get("tau_column", "tau_225"),
        missing=env_cfg.get("missing_token", "none"),
    )
    return env_map


def _site_slug(cfg_site: Dict[str, Any]) -> str:
    import re
    name = str(cfg_site.get("name", "Unknown")).strip()
    m = re.search(r"\(([^)]+)\)", name)
    if m:
        tok = re.sub(r"[^A-Za-z0-9]+", "", m.group(1)).lower()
        if tok:
            return tok
    tok = re.sub(r"[^A-Za-z0-9]+", "_", name).strip("_").lower()
    return tok or "site"


def _make_writer(cfg: Dict[str, Any], project_root: str, algo_name: str, log):
    out_cfg = cfg.get("output", {})
    out_path = out_cfg.get("out_tsv", "")
    if not out_path:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%SZ")
        site_code = _site_slug(cfg.get("site", {}))
        template = out_cfg.get(
            "filename_template",
            "offsets_{site}_{algo}_{stamp}.tsv",
        )
        base = template.format(site=site_code, algo=algo_name, stamp=stamp)
        out_dir = out_cfg.get("out_dir", os.path.join("output", "offsets"))
        out_path = os.path.join(out_dir, base)
    if not os.path.isabs(out_path):
        out_path = os.path.join(project_root, out_path)
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    print(f"Output TSV: {os.path.relpath(out_path, project_root)}")
    site = cfg.get("site", {})
    md = Metadata(
        location=site.get("name", "Unknown site"),
        antenna_diameter_m=site.get("antenna_diameter_m", 1.0),
        frequency_hz=site.get("frequency_hz", 1.0e9),
        software_version="dev",
        created_at_iso=None,
    )
    env_map = _load_env_lookup(cfg, project_root)

    # Pre-create the TSV with header so the file is never empty
    try:
        write_offsets_tsv(out_path, md, [], append=False)
    except Exception:
        pass

    first_write = True

    def write_row(timestamp_iso, az, el, daz, del_):
        nonlocal first_write
        env = env_map.get(timestamp_iso, {})
        row = Measurement(
            timestamp_iso=timestamp_iso,
            azimuth_deg=float(az),
            elevation_deg=float(el),
            offset_az_deg=float(daz),
            offset_el_deg=float(del_),
            temperature_c=env.get("temperature_C"),
            pressure_hpa=env.get("pressure_hPa"),
            humidity_frac=env.get("humidity_frac"),
        )
        append = not first_write
        write_offsets_tsv(out_path, md, [row], append=append)
        if first_write:
            first_write = False
        try:
            msg = (
                f"  -> az={float(az):.2f}, el={float(el):.2f}, "
                f"offset_az={float(daz):.4f}, offset_el={float(del_):.4f} (deg)"
            )
            print(msg); log(msg)
        except Exception:
            pass

    return write_row


def _build_algo_config(cfg: Dict[str, Any]) -> SimpleNamespace:
    # Base algorithm knobs (compatible with the frozen Config dataclass)
    name = cfg.get("algo", {}).get("name", "gauss2d")
    ga = cfg.get("algo", {}).get(name, {})

    base = Config(
        signal_min=float(ga.get("signal_min", 0.0)),
        grid_step_deg=float(ga.get("grid_step_deg", 0.05)),
        smooth_sigma_deg=float(ga.get("smooth_sigma_deg", 0.05)),
        fwhm_deg=float(ga.get("fwhm_deg", 0.5)),
        refraction=str(cfg.get("ephemeris", {}).get("refraction", "apparent")),
    )
    # Extras carried in a SimpleNamespace to avoid mutating the frozen dataclass
    calib = cfg.get("calibration", {})
    az_bias_deg = float(calib.get("az_bias_deg", 0.0))
    max_az_rate_dps = float(ga.get("max_az_rate_dps", 3.0))
    max_el_rate_dps = float(ga.get("max_el_rate_dps", 3.0))
    eph_backend = str(cfg.get("ephemeris", {}).get("backend", "astropy"))

    return SimpleNamespace(
        # fields from Config
        signal_min=base.signal_min,
        grid_step_deg=base.grid_step_deg,
        smooth_sigma_deg=base.smooth_sigma_deg,
        fwhm_deg=base.fwhm_deg,
        refraction=base.refraction,
        # extras
        az_bias_deg=az_bias_deg,
        max_az_rate_dps=max_az_rate_dps,
        max_el_rate_dps=max_el_rate_dps,
        ephemeris_backend=eph_backend,
    )


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    project_root = os.getcwd()
    cfg, paths = load_run_config(
        project_root=project_root,
        run_name=args.run,
        run_path=args.run_config,
        set_overrides=args._get_kwargs() and args.__dict__.get("set", []),
    )

    print("----- Effective configuration -----")
    print(dump_effective_config(cfg).rstrip())
    print("-----------------------------------")
    algo = _load_algo_module(cfg)

    if args.dump_effective_config:
        return 0

    log_path, log = _init_logger(project_root, args.log_dir)
    _log_header(log, project_root, paths, cfg)
    print(f"Log file: {os.path.relpath(log_path, project_root)}")

    inp = cfg.get("input", {})
    data_dir = os.path.join(project_root, inp.get("data_dir", "data"))
    maps: Iterable[MapInput] = discover_maps(
        data_dir=data_dir,
        start_iso=None,
        end_iso=None,
        recursive=True,
    )
    maps = list(maps)
    print(f"Discovered {len(maps)} maps")

    def _iter_maps_with_print(ms):
        total = len(ms)
        for i, mp in enumerate(ms, 1):
            msg = f"Processing map {i:02d}/{total}: {mp.map_id}"
            print(msg); log(msg)
            yield mp

    if not maps:
        msg = "No maps were discovered. Check input.data_dir, file naming, and discovery"
        print(f"ERROR: {msg}"); log(f"ERROR: {msg}")
        return 2

    writer = _make_writer(cfg, project_root, cfg.get('algo', {}).get('name', 'gauss2d'), log)

    site_cfg = cfg.get("site", {})
    site = Site(
        name=str(site_cfg.get("name", "Unknown")),
        latitude_deg=float(site_cfg.get("latitude_deg", 0.0)),
        longitude_deg=float(site_cfg.get("longitude_deg", 0.0)),
        elevation_m=float(site_cfg.get("elevation_m", 0.0)),
    )
    algo_cfg = _build_algo_config(cfg)

    algo.compute_offsets(
        maps=_iter_maps_with_print(maps),
        site=site,
        cfg=algo_cfg,
        write_row=writer,
    )
    print("Done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
