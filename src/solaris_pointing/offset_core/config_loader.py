from __future__ import annotations

import os
from typing import Dict, Any, Iterable, Tuple
try:
    import tomllib as toml  # py311+
except Exception:
    import tomli as toml  # fallback for older envs


def load_toml(path: str) -> Dict[str, Any]:
    with open(path, "rb") as f:
        return toml.load(f)


def merge_dicts(a: Dict[str, Any], b: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(a)
    for k, v in b.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = merge_dicts(out[k], v)
        else:
            out[k] = v
    return out


def apply_sets(cfg: Dict[str, Any], sets: Iterable[str]) -> Dict[str, Any]:
    for item in sets:
        if "=" not in item:
            raise ValueError(f"--set requires key=value, got: {item}")
        key, val = item.split("=", 1)
        path = key.strip().split(".")
        cursor = cfg
        for p in path[:-1]:
            if p not in cursor or not isinstance(cursor[p], dict):
                cursor[p] = {}
            cursor = cursor[p]
        # parse bool, int, float, or keep string
        v = parse_scalar(val.strip())
        cursor[path[-1]] = v
    return cfg


def parse_scalar(s: str):
    sl = s.lower()
    if sl in ("true", "false"):
        return sl == "true"
    try:
        if "." in s:
            return float(s)
        return int(s)
    except Exception:
        return s


def load_run_config(
    project_root: str,
    run_name: str | None,
    run_path: str | None,
    set_overrides: Iterable[str] = (),
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Load config by composing site, algo, and run, then apply --set.
    Returns (effective_cfg, summary_paths).
    summary_paths contains keys: run_path, site_path, algo_path.
    """
    summary = {"run_path": None, "site_path": None, "algo_path": None}

    if run_name and run_path:
        raise ValueError("Use either --run or --run-config, not both.")

    if run_name:
        run_path = os.path.join(project_root, "config", "runs", f"{run_name}.toml")

    if not run_path:
        raise ValueError("Missing --run or --run-config")

    if not os.path.isabs(run_path):
        run_path = os.path.normpath(os.path.join(project_root, run_path))

    if not os.path.exists(run_path):
        raise FileNotFoundError(
            f"Run file not found: {run_path}. Expected in config/runs for --run."
        )
    run_cfg = load_toml(run_path)
    summary["run_path"] = run_path

    site_cfg = {}
    algo_cfg = {}
    site_ref = run_cfg.get("include_site")
    algo_ref = run_cfg.get("include_algo")

    if site_ref:
        site_path = site_ref
        if not os.path.isabs(site_path):
            site_path = os.path.join(project_root, site_path)
        if not os.path.exists(site_path):
            raise FileNotFoundError(f"Site file not found: {site_path}")
        site_cfg = load_toml(site_path)
        summary["site_path"] = site_path

    if algo_ref:
        algo_path = algo_ref
        if not os.path.isabs(algo_path):
            algo_path = os.path.join(project_root, algo_path)
        if not os.path.exists(algo_path):
            raise FileNotFoundError(f"Algo file not found: {algo_path}")
        algo_cfg = load_toml(algo_path)
        summary["algo_path"] = algo_path

    # Merge order: site -> algo -> run
    cfg = merge_dicts(site_cfg, algo_cfg)
    cfg = merge_dicts(cfg, run_cfg)

    # Apply --set overrides last
    cfg = apply_sets(cfg, set_overrides)

    # Basic defaults
    cfg.setdefault("processing", {})
    cfg.setdefault("ephemeris", {})
    cfg.setdefault("calibration", {})
    cfg.setdefault("output", {})

    return cfg, summary


def dump_effective_config(cfg: Dict[str, Any]) -> str:
    try:
        import tomli_w  # type: ignore
    except Exception:
        # Manual minimal TOML emitter
        return _emit_kv(cfg, prefix="")
    return tomli_w.dumps(cfg)  # type: ignore


def _emit_kv(d: Dict[str, Any], prefix: str) -> str:
    lines = []
    scalars = {}
    subtables = {}
    for k, v in d.items():
        if isinstance(v, dict):
            subtables[k] = v
        else:
            scalars[k] = v
    for k, v in scalars.items():
        lines.append(f"{k} = {toml_value(v)}")
    for k, sub in subtables.items():
        lines.append("")
        lines.append(f"[{k}]")
        lines.append(_emit_kv(sub, prefix + k + "."))
    return "\n".join(lines)


def toml_value(v):
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, (int, float)):
        return str(v)
    if isinstance(v, list):
        inner = ", ".join([toml_value(x) for x in v])
        return f"[{inner}]"
    s = str(v).replace('"', '\"')
    return f'"{s}"'
