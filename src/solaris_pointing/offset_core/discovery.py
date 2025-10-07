
from __future__ import annotations

"""
discovery.py
============

Utilities to discover input maps from a data directory. We pair *.path and
*.sky files by a common `map_id` prefix **before the first underscore**.

Example filenames:
  - 250106T010421_OASI.path  → map_id = "250106T010421"
  - 250106T010421_OASI.sky   → map_id = "250106T010421"

We also convert `map_id` → canonical UTC timestamp (ISO with trailing 'Z'):
  YYMMDDTHHMMSS  →  20YY-MM-DDTHH:MM:SSZ
"""

import os
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple

from .model import MapInput


def parse_map_id_timestamp(map_id: str) -> datetime:
    """
    Convert a map_id 'YYMMDDTHHMMSS' into a UTC datetime.

    Raises
    ------
    ValueError
        If the format is not as expected.
    """
    if len(map_id) != 13 or map_id[6] != "T":
        raise ValueError(f"Unexpected map_id format: {map_id}")
    yy = int(map_id[0:2]); year = 2000 + yy
    month = int(map_id[2:4]); day = int(map_id[4:6])
    hour = int(map_id[7:9]); minute = int(map_id[9:11]); second = int(map_id[11:13])
    return datetime(year, month, day, hour, minute, second, tzinfo=timezone.utc)


def _find_pairs(data_dir: str) -> Dict[str, Tuple[str, str]]:
    """
    Discover *.path and *.sky files and pair them by common map_id prefix.

    Returns
    -------
    dict
        { map_id : (path_file, sky_file) } for all pairs found.
    """
    entries: Dict[str, Dict[str, str]] = {}
    for fname in os.listdir(data_dir):
        if not (fname.endswith(".path") or fname.endswith(".sky")):
            continue
        root = os.path.splitext(fname)[0]
        map_id = root.split("_")[0]
        kind = "path" if fname.endswith(".path") else "sky"
        entries.setdefault(map_id, {})[kind] = os.path.join(data_dir, fname)

    pairs: Dict[str, Tuple[str, str]] = {}
    for map_id, d in entries.items():
        if "path" in d and "sky" in d:
            pairs[map_id] = (d["path"], d["sky"])
    return pairs


def discover_maps(data_dir: str,
                  start_iso: Optional[str],
                  end_iso: Optional[str]) -> List[MapInput]:
    """
    Return a list of MapInput found in `data_dir`, optionally filtered by a time
    window [start_iso, end_iso]. The list is **sorted by timestamp**.

    Parameters
    ----------
    data_dir : str
        Directory containing *.path and *.sky files.
    start_iso : str or None
        Inclusive lower bound ISO timestamp (e.g., "2025-01-01T00:00:00Z"), or None.
    end_iso : str or None
        Inclusive upper bound ISO timestamp (e.g., "2025-01-31T23:59:59Z"), or None.
    """
    pairs = _find_pairs(data_dir)
    start_dt = datetime.fromisoformat(start_iso.replace("Z", "+00:00")).astimezone(timezone.utc) if start_iso else None
    end_dt = datetime.fromisoformat(end_iso.replace("Z", "+00:00")).astimezone(timezone.utc) if end_iso else None

    items: List[Tuple[datetime, MapInput]] = []
    for map_id, (path_file, sky_file) in pairs.items():
        ts = parse_map_id_timestamp(map_id)
        if (start_dt and ts < start_dt) or (end_dt and ts > end_dt):
            continue
        items.append((
            ts,
            MapInput(
                map_id=map_id,
                path_file=path_file,
                sky_file=sky_file,
                map_timestamp_iso=ts.strftime("%Y-%m-%dT%H:%M:%SZ"),
            )
        ))

    items.sort(key=lambda x: x[0])
    return [mp for _, mp in items]
