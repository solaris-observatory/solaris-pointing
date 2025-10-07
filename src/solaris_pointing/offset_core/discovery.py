
from __future__ import annotations

"""
discovery.py
============

Utilities to discover input maps from a data directory. We pair *.path and
*.sky files and derive a canonical `map_id` as the first token matching
`^\d{6}T\d{6}` at the **start** of the basename. Filenames may contain suffixes
(e.g., "250103T202803_OASI", "250103T202803bOASI"): those are considered
**segments of the same map** and will be **merged** into one logical map.

Examples:
  "250103T202803_OASI.path"   → token = "250103T202803"
  "250103T202803bOASI.sky"    → token = "250103T202803"
Both segments are concatenated (header kept once) into a temporary
`.combined` folder so that downstream algorithms see a single map.

Supports **recursive** discovery through subdirectories.
"""

import os
import re
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple

from .model import MapInput

_TS_RE = re.compile(r'^(\d{6}T\d{6})')

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


def _extract_token(root: str) -> Optional[str]:
    """
    Extract the leading timestamp token YYMMDDTHHMMSS from a basename (without extension).
    Return None if not found.
    """
    m = _TS_RE.match(root)
    return m.group(1) if m else None



def _is_binary_variant(root: str, token: str) -> bool:
    """
    Heuristic: treat files whose basename has a 'b' **immediately after**
    the YYMMDDTHHMMSS token as **binary variants** to be ignored, e.g.:
        250103T202803bOASI.path  → ignored
    """
    if not token:
        return False
    if len(root) > len(token) and (root[len(token)] in ("b","B")):
        return True
    return False
def _scan_files(data_dir: str, recursive: bool) -> List[Tuple[str, str]]:
    """
    Return list of (fullpath, ext) for *.path and *.sky under data_dir.
    """
    out: List[Tuple[str, str]] = []
    if recursive:
        for dirpath, _dirs, files in os.walk(data_dir):
            for fname in files:
                if fname.endswith(".path") or fname.endswith(".sky"):
                    out.append((os.path.join(dirpath, fname), os.path.splitext(fname)[1]))
    else:
        for fname in os.listdir(data_dir):
            if fname.endswith(".path") or fname.endswith(".sky"):
                out.append((os.path.join(data_dir, fname), os.path.splitext(fname)[1]))
    return out


def _group_segments(files: List[Tuple[str, str]]) -> Dict[str, Dict[str, List[str]]]:
    """
    Group files by timestamp token:
      { token : { 'path': [paths...], 'sky': [paths...] } }
    Ignore files whose basename doesn't start with the token.
    """
    groups: Dict[str, Dict[str, List[str]]] = {}
    for full, ext in files:
        root = os.path.splitext(os.path.basename(full))[0]
        token = _extract_token(root)
        if not token:
            continue
        if _is_binary_variant(root, token):
            # Skip binary variants (suffix 'b' right after token)
            continue
        kind = 'path' if ext == '.path' else 'sky'
        groups.setdefault(token, {}).setdefault(kind, []).append(full)

    # Sort each list for deterministic ordering (base first, then suffix variants)
    for token, kinds in groups.items():
        for k in ('path','sky'):
            if k in kinds:
                kinds[k].sort()
    return groups


def _read_data_lines(file_path: str) -> Tuple[str, List[str]]:
    """
    Read a tab-separated file returning (header_line, data_lines).
    Keeps the first line as header, returns the remaining non-empty lines.
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.read().splitlines()
    if not lines:
        return "", []
    header = lines[0]
    data = [ln for ln in lines[1:] if ln.strip()]
    return header, data


def _ensure_combined_dir(root_dir: str) -> str:
    out = os.path.join(root_dir, ".combined")
    os.makedirs(out, exist_ok=True)
    return out


def _combine_segments(token: str, seg_paths: List[str], combined_dir: str, ext: str) -> str:
    """
    Concatenate multiple segment files (same schema) into a single file:
    - keep header from the first segment
    - append only data lines from subsequent segments
    Returns the path to the combined file.
    """
    out_path = os.path.join(combined_dir, f"{token}_COMBINED{ext}")
    header_written = False
    with open(out_path, 'w', encoding='utf-8') as out:
        for i, seg in enumerate(seg_paths):
            hdr, data = _read_data_lines(seg)
            if not header_written and hdr:
                out.write(hdr.rstrip() + "\n")
                header_written = True
            for ln in data:
                out.write(ln.rstrip() + "\n")
    return out_path


def _prepare_pairs(groups: Dict[str, Dict[str, List[str]]], root_dir: str) -> Dict[str, Tuple[str, str]]:
    """
    For each token, if both 'path' and 'sky' exist, produce a single pair:
    - If there is a single segment per kind → use that file directly.
    - If there are multiple segments → concatenate them into .combined files.
    Returns:
      { token : (path_file, sky_file) }
    """
    pairs: Dict[str, Tuple[str, str]] = {}
    combined_dir = _ensure_combined_dir(root_dir)

    for token, kinds in groups.items():
        if 'path' not in kinds or 'sky' not in kinds:
            continue

        # Decide path file
        path_files = kinds['path']
        if len(path_files) == 1:
            path_out = path_files[0]
        else:
            path_out = _combine_segments(token, path_files, combined_dir, ".path")

        # Decide sky file
        sky_files = kinds['sky']
        if len(sky_files) == 1:
            sky_out = sky_files[0]
        else:
            sky_out = _combine_segments(token, sky_files, combined_dir, ".sky")

        pairs[token] = (path_out, sky_out)

    return pairs


def discover_maps(data_dir: str,
                  start_iso: Optional[str],
                  end_iso: Optional[str],
                  recursive: bool = False) -> List[MapInput]:
    """
    Return a list of MapInput found in `data_dir`, optionally filtered by a time
    window [start_iso, end_iso]. The list is **sorted by timestamp** and **merges
    same-token segments** into a single logical map by concatenation.

    Parameters
    ----------
    data_dir : str
        Directory containing *.path and *.sky files (or subdirectories).
    start_iso : str or None
        Inclusive lower bound ISO timestamp (e.g., "2025-01-01T00:00:00Z"), or None.
    end_iso : str or None
        Inclusive upper bound ISO timestamp (e.g., "2025-01-31T23:59:59Z"), or None.
    recursive : bool
        If True, search recursively below `data_dir`.
    """
    files = _scan_files(data_dir, recursive=recursive)
    groups = _group_segments(files)
    pairs = _prepare_pairs(groups, root_dir=data_dir)

    start_dt = datetime.fromisoformat(start_iso.replace("Z", "+00:00")).astimezone(timezone.utc) if start_iso else None
    end_dt = datetime.fromisoformat(end_iso.replace("Z", "+00:00")).astimezone(timezone.utc) if end_iso else None

    items: List[Tuple[datetime, MapInput]] = []
    for token, (path_file, sky_file) in pairs.items():
        ts = parse_map_id_timestamp(token)
        if (start_dt and ts < start_dt) or (end_dt and ts > end_dt):
            continue
        items.append((
            ts,
            MapInput(
                map_id=token,
                path_file=path_file,
                sky_file=sky_file,
                map_timestamp_iso=ts.strftime("%Y-%m-%dT%H:%M:%SZ"),
            )
        ))

    items.sort(key=lambda x: x[0])
    return [mp for _, mp in items]
