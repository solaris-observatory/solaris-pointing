from __future__ import annotations

import csv
from typing import Dict, Any, Optional


def read_environment_tsv(
    path: str,
    ts_col: str,
    p_col: Optional[str],
    t_col: Optional[str],
    tau_col: Optional[str],
    missing: str = "none",
) -> Dict[str, Dict[str, Any]]:
    env: Dict[str, Dict[str, Any]] = {}
    with open(path, "r", encoding="utf-8") as f:
        rd = csv.DictReader(f, delimiter="\t")
        for row in rd:
            ts = row.get(ts_col)
            if not ts:
                continue
            e: Dict[str, Any] = {}

            if (
                p_col
                and p_col in row
                and row[p_col]
                and row[p_col].lower() != missing
            ):
                e["pressure_hPa"] = _num(row[p_col])

            if (
                t_col
                and t_col in row
                and row[t_col]
                and row[t_col].lower() != missing
            ):
                e["temperature_C"] = _num(row[t_col])

            if (
                tau_col
                and tau_col in row
                and row[tau_col]
                and row[tau_col].lower() != missing
            ):
                e["tau_225"] = _num(row[tau_col])

            env[ts] = e
    return env


def _num(s: str):
    try:
        if "." in s:
            return float(s)
        return int(s)
    except Exception:
        return None
