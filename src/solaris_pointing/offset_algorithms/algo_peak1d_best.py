
from __future__ import annotations

"""
algo_peak1d_best.py — linear (1D) offsets using the single *best* sub-scan

Overview
--------
- Parse .sky/.path using POSIX times.
- Segment PATH rows where F0==1 into contiguous sub-scans.
- For each row, compute:
    * row_max = max(signal in [t0,t1])
    * centroid time t_c over samples >= 0.75*row_max (fallback: time of absolute max)
- Select the *best* row as the one with largest row_max.
- Compute Sun at t_c (ISO UTC), read encoder (RAW) at nearest PATH time to t_c.
- Apply az_bias_deg only inside the az delta: dAz = signed_delta(az_raw + az_bias, az_sun).
- Write a single TSV row per map with azimuth_deg RAW and the offsets from the best row.

Ephemeris
---------
- backend via cfg.ephemeris_backend ("pysolar"|"astropy"), default "astropy".
- refraction via cfg.refraction ("none"|"apparent").
"""

from typing import Iterable, List, Tuple, Dict, Any
from datetime import datetime, timezone
import os
import math
import numpy as np


# --------------------------- utilities ---------------------------

def _iso_utc_from_posix(t: float) -> str:
    return datetime.fromtimestamp(t, tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _signed_delta_deg(a: float, b: float) -> float:
    """Minimal signed difference a - b in degrees in (-180, 180]."""
    d = (a - b + 180.0) % 360.0 - 180.0
    return 180.0 if d == -180.0 else d


def _nearest_index(arr: np.ndarray, x: float) -> int:
    return int(np.nanargmin(np.abs(arr - x)))


def _wrap0_360(a: float) -> float:
    x = a % 360.0
    return x if x >= 0.0 else x + 360.0


# ----------------------- direct ephemeris backends -----------------------

def _sun_astropy(timestamp_iso: str, site: Dict[str, Any], refraction: str) -> Tuple[float, float]:
    try:
        from astropy.time import Time
        from astropy.coordinates import EarthLocation, AltAz, get_sun
        import astropy.units as u
    except Exception as e:
        raise NotImplementedError("Astropy backend requested but astropy is not available.") from e

    lat = float(site.get("latitude_deg", 0.0))
    lon = float(site.get("longitude_deg", 0.0))
    elev = float(site.get("elevation_m", 0.0))

    obstime = Time(timestamp_iso, format="isot", scale="utc")
    loc = EarthLocation(lon=lon * u.deg, lat=lat * u.deg, height=elev * u.m)

    if str(refraction).lower() == "none":
        pressure = 0 * u.hPa
    else:
        pressure = 1010 * u.hPa

    frame = AltAz(obstime=obstime, location=loc, pressure=pressure, temperature=0 * u.deg_C)
    sun = get_sun(obstime).transform_to(frame)
    az = float(sun.az.to(u.deg).value)
    el = float(sun.alt.to(u.deg).value)
    return _wrap0_360(az), el


def _sun_pysolar(timestamp_iso: str, site: Dict[str, Any], refraction: str) -> Tuple[float, float]:
    try:
        from pysolar.solar import get_azimuth, get_altitude
    except Exception as e:
        raise NotImplementedError("PySolar backend requested but pysolar is not available.") from e

    lat = float(site.get("latitude_deg", 0.0))
    lon = float(site.get("longitude_deg", 0.0))
    dt = datetime.fromisoformat(timestamp_iso.replace("Z", "+00:00"))
    az = float(get_azimuth(lat, lon, dt))
    el = float(get_altitude(lat, lon, dt))
    return _wrap0_360(az), el


def _get_sun_altaz(timestamp_iso: str, site: Dict[str, Any], backend: str, refraction: str) -> Tuple[float, float]:
    be = (backend or "astropy").lower()
    if be == "pysolar":
        return _sun_pysolar(timestamp_iso, site, refraction)
    elif be == "astropy":
        return _sun_astropy(timestamp_iso, site, refraction)
    else:
        raise NotImplementedError(f"Unsupported ephemeris backend: {backend}")


# ----------------------- file readers (.sky/.path) -----------------------

def _read_sky(files: List[str]) -> Tuple[np.ndarray, np.ndarray]:
    """Return (t_posix, signal)."""
    ts_list: List[float] = []
    s_list: List[float] = []
    for fp in sorted(files):
        if not os.path.exists(fp):
            continue
        with open(fp, "r", encoding="utf-8", errors="ignore") as f:
            header = f.readline().strip().split("\t")
            col_t = header.index("Posix_time")
            col_ms = header.index("ms")
            col_sig = header.index("Signal")
            for line in f:
                if not line.strip():
                    continue
                parts = line.rstrip("\n").split("\t")
                try:
                    t = float(parts[col_t]) + float(parts[col_ms]) / 1000.0
                    s = float(parts[col_sig])
                except Exception:
                    continue
                ts_list.append(t); s_list.append(s)
    return np.asarray(ts_list, float), np.asarray(s_list, float)


def _read_path(files: List[str]) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return (t_posix, az_deg, el_deg, F0)."""
    ts_list: List[float] = []; az_list: List[float] = []; el_list: List[float] = []; f0_list: List[float] = []
    for fp in sorted(files):
        if not os.path.exists(fp):
            continue
        with open(fp, "r", encoding="utf-8", errors="ignore") as f:
            header = f.readline().strip().split("\t")
            col_t = header.index("Posix_time"); col_ms = header.index("ms")
            col_az = header.index("Azimuth"); col_el = header.index("Elevation")
            col_f0 = header.index("F0")
            for line in f:
                if not line.strip():
                    continue
                parts = line.rstrip("\n").split("\t")
                try:
                    t = float(parts[col_t]) + float(parts[col_ms]) / 1000.0
                    az = float(parts[col_az]); el = float(parts[col_el])
                    f0 = float(parts[col_f0])
                except Exception:
                    continue
                ts_list.append(t); az_list.append(az); el_list.append(el); f0_list.append(f0)
    return (np.asarray(ts_list, float), np.asarray(az_list, float),
            np.asarray(el_list, float), np.asarray(f0_list, float))


# ----------------------- best-row selection -----------------------

def _rows_from_f0(ts_path: np.ndarray, f0: np.ndarray) -> List[Tuple[int, int]]:
    """Return list of (start_idx, end_idx) inclusive for contiguous segments where F0>0.5."""
    on = f0 > 0.5
    if not np.any(on):
        return []
    idx = np.nonzero(on)[0]
    splits = np.where(np.diff(idx) > 1)[0]
    starts = [idx[0], *[idx[i+1] for i in splits]]
    ends = [idx[i] for i in splits] + [idx[-1]]
    return list(zip(starts, ends))


def _best_row_time(ts_sky: np.ndarray, s: np.ndarray,
                   ts_path: np.ndarray, rows: List[Tuple[int, int]],
                   thr_rel: float = 0.75) -> float:
    """Pick the row with highest peak power and return its centroid time."""
    if not rows:
        # fallback: absolute max time over entire sky
        k = int(np.nanargmax(s))
        return float(ts_sky[k])
    best_peak = -np.inf
    best_cent = None
    for a, b in rows:
        t0, t1 = ts_path[a], ts_path[b]
        m = (ts_sky >= t0) & (ts_sky <= t1)
        if not np.any(m):
            continue
        s_row = s[m]; t_row = ts_sky[m]
        if s_row.size < 3:
            continue
        row_max = float(np.nanmax(s_row))
        if not np.isfinite(row_max) or row_max <= 0:
            continue
        thr = thr_rel * row_max
        keep = s_row >= thr
        if np.any(keep):
            w = s_row[keep]; tk = t_row[keep]
            t_c = float(np.nansum(w * tk) / np.nansum(w))
        else:
            kmax = int(np.nanargmax(s_row))
            t_c = float(t_row[kmax])
        if row_max > best_peak:
            best_peak = row_max
            best_cent = t_c
    if best_cent is None:
        # last fallback
        k = int(np.nanargmax(s))
        return float(ts_sky[k])
    return float(best_cent)


# ----------------------------- main API -----------------------------

def compute_offsets(
    maps: Iterable,
    site,
    cfg,
    write_row,
) -> None:
    """
    Compute a single offset per map selecting the *best* sub-scan.
    """
    az_bias = float(getattr(cfg, "az_bias_deg", 0.0))
    refraction = getattr(cfg, "refraction", "none")
    eph_backend = getattr(cfg, "ephemeris_backend", "astropy")

    for mp in maps:
        # Collect file paths from various MapInput shapes
        sky_files = list(getattr(mp, "sky_files", []) or [])
        path_files = list(getattr(mp, "path_files", []) or [])

        # Accept single-file attributes as well
        sky_single = None
        path_single = None
        for attr in ("sky_file", "sky"):
            v = getattr(mp, attr, None)
            if isinstance(v, str) and v.endswith(".sky"):
                sky_single = v; break
        for attr in ("path_file", "path"):
            v = getattr(mp, attr, None)
            if isinstance(v, str) and v.endswith(".path"):
                path_single = v; break
        if sky_single and sky_single not in sky_files:
            sky_files.append(sky_single)
        if path_single and path_single not in path_files:
            path_files.append(path_single)

        # Fallback discovery by map_id in base_dir
        if (not sky_files or not path_files) and getattr(mp, "map_id", None):
            base_dir = getattr(mp, "base_dir", None)
            if not base_dir:
                any_file = sky_single or path_single
                if isinstance(any_file, str) and os.path.exists(any_file):
                    base_dir = os.path.dirname(any_file)
            if base_dir and os.path.isdir(base_dir):
                token = str(getattr(mp, "map_id"))
                for name in os.listdir(base_dir):
                    if not name.startswith(token):
                        continue
                    fp = os.path.join(base_dir, name)
                    if name.endswith(".sky"): sky_files.append(fp)
                    elif name.endswith(".path"): path_files.append(fp)

        ts_sky, s = _read_sky(sky_files)
        ts_path, az, el, f0 = _read_path(path_files)

        if ts_sky.size == 0 or ts_path.size == 0:
            continue

        # Build rows and select best
        rows = _rows_from_f0(ts_path, f0)
        t_best = _best_row_time(ts_sky, s, ts_path, rows, thr_rel=0.75)
        ts_iso = _iso_utc_from_posix(t_best)

        # Sun position at t_best (direct backend call)
        site_dict: Dict[str, Any] = {
            "name": getattr(site, "name", "Unknown"),
            "latitude_deg": getattr(site, "latitude_deg", 0.0),
            "longitude_deg": getattr(site, "longitude_deg", 0.0),
            "elevation_m": getattr(site, "elevation_m", 0.0),
        }
        az_sun, el_sun = _get_sun_altaz(ts_iso, site_dict, eph_backend, refraction)

        # Read encoder at that time (RAW in TSV; bias only inside delta)
        k = _nearest_index(ts_path, t_best)
        az_raw = float(az[k])
        el_raw = float(el[k])

        if az_raw < 0:
            az_raw += 360.0

        d_az = _signed_delta_deg(az_raw + az_bias, az_sun)
        d_el = el_raw - el_sun

        write_row(ts_iso, az_raw, el_raw, d_az, d_el)
