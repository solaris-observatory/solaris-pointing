from __future__ import annotations

def _wrap0_360(a: float) -> float:
    """Wrap angle in degrees to [0, 360)."""
    x = a % 360.0
    return x if x >= 0.0 else x + 360.0

def _signed_delta_deg(a: float, b: float) -> float:
    """Minimal signed difference a - b in degrees in (-180, 180]."""
    d = (a - b + 180.0) % 360.0 - 180.0
    # map -180 to +180 for consistency (optional)
    return 180.0 if d == -180.0 else d


"""
algo_gauss2d.py (clean)
=======================
Two–dimensional peak algorithm with robust I/O for TSV .path/.sky files.

- Parse tabular files with header (Posix_time, ms, UTC, Azimuth/Elevation/Signal).
- Synchronize power to pointing using time interpolation.
- Build a small 2D grid around the maximum power and refine by weighted centroid.
- Determine the timestamp at the 2D peak by nearest neighbor in (az, el).
- Compute solar az,el with astropy.get_sun; honor cfg.refraction == "none".
- Keep algorithm pure: no bias application here.
"""

from typing import Iterable
import numpy as np
from dataclasses import dataclass

from astropy.coordinates import AltAz, EarthLocation, get_sun
import astropy.units as u
from astropy.time import Time

from solaris_pointing.offset_core.model import Site, MapInput, Config, WriterFn


@dataclass
class _Samples:
    t: np.ndarray      # seconds relative to map start
    az: np.ndarray     # degrees
    el: np.ndarray     # degrees
    pw: np.ndarray     # power
    on: np.ndarray     # bool mask of valid (finite) power


def _load_path(path_file: str):
    t, az, el = [], [], []
    posix0 = None
    ms0 = None
    with open(path_file, "r", encoding="utf-8") as f:
        for ln in f:
            if not ln.strip() or ln.startswith("#"):
                continue
            parts = ln.rstrip().split("\t")
            if len(parts) < 5 or parts[0] == "Posix_time":
                continue
            try:
                posix = float(parts[0])
                ms = float(parts[1])
                a0 = float(parts[3])
                e0 = float(parts[4])
            except ValueError:
                continue
            if posix0 is None:
                posix0, ms0 = posix, ms
            t_rel = (posix - posix0) + (ms - ms0) / 1000.0
            t.append(t_rel)
            az.append(a0)
            el.append(e0)
    return np.asarray(t), np.asarray(az), np.asarray(el)


def _load_sky(sky_file: str):
    t, pw = [], []
    posix0 = None
    ms0 = None
    with open(sky_file, "r", encoding="utf-8") as f:
        for ln in f:
            if not ln.strip() or ln.startswith("#"):
                continue
            parts = ln.rstrip().split("\t")
            if len(parts) < 4 or parts[0] == "Posix_time":
                continue
            try:
                posix = float(parts[0])
                ms = float(parts[1])
                sig = float(parts[3])
            except ValueError:
                continue
            if posix0 is None:
                posix0, ms0 = posix, ms
            t_rel = (posix - posix0) + (ms - ms0) / 1000.0
            t.append(t_rel)
            pw.append(sig)
    return np.asarray(t), np.asarray(pw)


def _sync_power_to_pointing(path_file: str, sky_file: str) -> _Samples:
    t_p, az, el = _load_path(path_file)
    t_s, pw_s = _load_sky(sky_file)
    if t_p.size == 0:
        return _Samples(t_p, az, el, np.zeros_like(t_p), np.zeros_like(t_p, bool))
    if t_s.size == 0:
        return _Samples(t_p, az, el, np.zeros_like(t_p), np.zeros_like(t_p, bool))
    order = np.argsort(t_s)
    t_s = t_s[order]
    pw_s = pw_s[order]
    pw = np.interp(t_p, t_s, pw_s, left=np.nan, right=np.nan)
    on = np.isfinite(pw)
    return _Samples(t_p, az, el, pw, on)


def _sun_altaz(timestamp_iso: str, site: Site, cfg: Config) -> tuple[float, float]:
    loc = EarthLocation(
        lat=site.latitude_deg * u.deg,
        lon=site.longitude_deg * u.deg,
        height=site.elevation_m * u.m,
    )
    obstime = Time(timestamp_iso, scale="utc")
    if getattr(cfg, "refraction", "none") == "none":
        frame = AltAz(location=loc, obstime=obstime, pressure=0 * u.hPa)
    else:
        frame = AltAz(location=loc, obstime=obstime)
    altaz = get_sun(obstime).transform_to(frame)
    return float(altaz.az.deg), float(altaz.alt.deg)


def _grid_peak(az: np.ndarray, el: np.ndarray, pw: np.ndarray,
               grid_step_deg: float = 0.05) -> tuple[float, float]:
    # Build a small grid around the max-power sample and compute weighted centroid.
    k = int(np.nanargmax(pw))
    a0, e0 = float(az[k]), float(el[k])
    # Select neighbors within +/- 3 grid steps
    da = np.abs(az - a0) <= 3 * grid_step_deg
    de = np.abs(el - e0) <= 3 * grid_step_deg
    mask = np.isfinite(pw) & da & de
    if not np.any(mask):
        return a0, e0
    w = np.maximum(pw[mask] - np.nanmin(pw[mask]), 0.0) + 1e-9
    az_c = float(np.sum(az[mask] * w) / np.sum(w))
    el_c = float(np.sum(el[mask] * w) / np.sum(w))
    return az_c, el_c


def compute_offsets(
    maps: Iterable[MapInput],
    site: Site,
    cfg: Config,
    write_row: WriterFn,
) -> None:
    from datetime import datetime, timezone, timedelta

    for mp in maps:
        S = _sync_power_to_pointing(mp.path_file, mp.sky_file)
        if S.t.size == 0:
            continue
        # 2D peak (weighted centroid around local max)
        az_meas, el_meas = _grid_peak(S.az, S.el, S.pw, cfg.grid_step_deg)
        az_meas = _wrap0_360(float(az_meas) + float(getattr(cfg, "az_bias_deg", 0.0)))
        el_meas = float(el_meas) + float(getattr(cfg, "el_bias_deg", 0.0))

        # Timestamp at the 2D peak via nearest neighbor in (az, el)
        d2 = (S.az - az_meas) ** 2 + (S.el - el_meas) ** 2
        k = int(np.nanargmin(d2))
        t_peak = float(S.t[k])
        t0 = datetime.fromisoformat(mp.map_timestamp_iso.replace("Z", "+00:00"))
        best_time_iso = (
            t0 + timedelta(seconds=t_peak)
        ).astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

        # Ideal Sun position (refraction as per cfg)
        az_sun, el_sun = _sun_altaz(best_time_iso, site, cfg)

        d_az = _signed_delta_deg(az_meas, az_sun)
        d_el = el_meas - el_sun

        write_row(mp.map_timestamp_iso, az_meas, el_meas, d_az, d_el)
