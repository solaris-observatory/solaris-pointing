
from __future__ import annotations

"""
env_loader.py
=============

Loader and validator for per-map environment data in a single CSV file.

Format (header is **required** and case-sensitive):
    map_id,temperature_c,pressure_hpa,humidity_frac

Example:
    250106T010421,-28.5,690.2,0.55
    250106T025443,-27.9,689.7,0.52

Semantics
---------
- `map_id` must match the prefix derived from *.path/*.sky files.
- Values may be missing or non-numeric; such entries are converted to `None`.
- The validator reports three classes of issues:
    1) Environment rows whose `map_id` is **not** present among discovered maps.
    2) Maps found in the data dir **without** a corresponding environment row.
    3) Duplicate `map_id` rows in the CSV (later rows override earlier ones; a warning is emitted).

The runner determines whether these issues are fatal (via --env-strict).
"""

from typing import Dict, Iterable, List, Optional, Tuple
from .model import MapInput

EnvTuple = Tuple[Optional[float], Optional[float], Optional[float]]
# (temperature_c, pressure_hpa, humidity_frac)

def load_environment_csv(csv_path: str) -> Dict[str, EnvTuple]:
    """
    Load an environment CSV.

    Returns
    -------
    dict
        { map_id : (temperature_c, pressure_hpa, humidity_frac) }

    Notes
    -----
    - Missing/invalid numeric values are returned as None.
    - Duplicate map_id rows: later rows **override** earlier ones.
      The validator can still report duplicates for user awareness.
    """
    env: Dict[str, EnvTuple] = {}
    seen_counts: Dict[str, int] = {}

    with open(csv_path, "r", encoding="utf-8") as f:
        header = f.readline().strip().split(",")
        expected = ["map_id", "temperature_c", "pressure_hpa", "humidity_frac"]
        if [h.strip() for h in header] != expected:
            raise ValueError(f"Unexpected header in {csv_path}. Expected {expected}, got {header}")
        for ln in f:
            if not ln.strip():
                continue
            parts = [p.strip() for p in ln.rstrip("\n").split(",")]
            if len(parts) < 4:
                continue
            mid = parts[0]
            def to_float(s: str) -> Optional[float]:
                try:
                    return float(s) if s != "" else None
                except Exception:
                    return None
            t = to_float(parts[1])
            p = to_float(parts[2])
            h = to_float(parts[3])
            env[mid] = (t, p, h)
            seen_counts[mid] = seen_counts.get(mid, 0) + 1

    # Attach duplicate counts for validator via a hidden attribute (optional)
    env.__dict__["_duplicates"] = {k: c for k, c in seen_counts.items() if c > 1}  # type: ignore[attr-defined]
    return env


def validate_environment(env_by_map: Dict[str, EnvTuple],
                         maps: Iterable[MapInput]) -> List[str]:
    """
    Perform validation and return human-readable warnings:
    - env rows with map_id not present among discovered maps
    - discovered maps missing env data
    - duplicate map_id rows encountered in CSV (informational)

    The runner decides whether to treat these as warnings or errors.
    """
    warnings: List[str] = []
    set_maps = {m.map_id for m in maps}
    set_env = set(env_by_map.keys())

    # Unknown env rows (orphans)
    extra = sorted(set_env - set_maps)
    if extra:
        warnings.append(f"Environment rows with unknown map_id (ignored during write): {', '.join(extra)}")

    # Maps missing environment
    missing = sorted(set_maps - set_env)
    if missing:
        warnings.append(f"Maps without environment data: {', '.join(missing)}")

    # Duplicates info
    dups = getattr(env_by_map, "_duplicates", {})  # type: ignore[attr-defined]
    if dups:
        formatted = ", ".join(f"{k}×{v}" for k, v in sorted(dups.items()))
        warnings.append(f"Duplicate environment rows (last occurrence used): {formatted}")

    return warnings
