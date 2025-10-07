
from __future__ import annotations

"""
algo_gauss2d.py
===============
Reference implementation of the plugin API for computing telescope pointing offsets.

Strategy
--------
1) Load *.path (az, el vs time) and *.sky (power vs time).
2) Synchronize sky (power) samples to the nearest pointing samples (az/el).
3) Keep only "on-scan" samples above a power threshold (configurable).
4) Project samples onto a regular 2D grid (az, el → power), apply a mild Gaussian
   smoothing, find the peak, and fit a small **elliptical 2D Gaussian** window
   around the peak to estimate the centroid (az_meas, el_meas).
   - If the fit fails, fallback to a simple power-weighted centroid (center of mass)
     within the same window.
5) Compute the Sun apparent AltAz using Astropy at the map timestamp (from map_id).
6) Offsets are Δaz = az_meas − az_sun, Δel = el_meas − el_sun.
7) Immediately call `write_row(timestamp_iso, az_meas, el_meas, d_az, d_el)`.

This module is deliberately independent from file-output concerns; the runner
provides the writer that uses `offset_io`.
"""

import numpy as np
from typing import Iterable, Tuple

from scipy.interpolate import griddata
from scipy.ndimage import gaussian_filter
from scipy.optimize import curve_fit

from astropy.coordinates import AltAz, EarthLocation, get_sun
from astropy.time import Time
import astropy.units as u

from solaris_pointing.offset_core.model import Site, MapInput, Config, WriterFn

# ----------------------------
# I/O helpers
# ----------------------------

def _load_path(path_file: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Load the .path file.

    Expected columns (tab-separated, skip first line header):
      - col 0: seconds
      - col 1: milliseconds
      - col 3: azimuth [deg]
      - col 4: elevation [deg]
      - col 7: flag (1 for "on-scan", 0 otherwise)
    """
    t_s, t_ms, az, el = np.loadtxt(path_file, dtype=float, delimiter='\t',
                                   skiprows=1, usecols=(0, 1, 3, 4), unpack=True)
    flag = np.loadtxt(path_file, dtype=float, delimiter='\t',
                      skiprows=1, usecols=(7,), unpack=True)
    t = t_s + t_ms / 1000.0

    # Normalize az into [0, 360)
    az = np.where(az < 0.0, az + 360.0, az)
    az = np.where(az >= 360.0, az - 360.0, az)

    return t, az, el, flag


def _load_sky(sky_file: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load the .sky file.

    Expected columns (tab-separated, skip first line header):
      - col 0: seconds
      - col 1: milliseconds
      - col 3: signal (power units)
    """
    t_s, t_ms, power = np.loadtxt(sky_file, dtype=float, delimiter='\t',
                                  skiprows=1, usecols=(0, 1, 3), unpack=True)
    t = t_s + t_ms / 1000.0
    return t, power


def _nearest_index(x: np.ndarray, val: float) -> int:
    """Return the index of `x` with value closest to `val` (absolute difference)."""
    return int(np.argmin(np.abs(x - val)))


def _sync_power_to_pointing(t_path: np.ndarray,
                            az_path: np.ndarray,
                            el_path: np.ndarray,
                            flag_path: np.ndarray,
                            t_sky: np.ndarray,
                            power: np.ndarray,
                            cfg: Config) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Associate each sky (power) sample with the nearest pointing (az, el) sample.
    Keep only samples where the .path flag indicates we're "on-scan", and where
    the signal exceeds the configured threshold.

    Returns:
      az_s : (K,) float64  azimuth for selected samples
      el_s : (K,) float64  elevation for selected samples
      pw_s : (K,) float64  power for selected samples
    """
    idxs = np.array([_nearest_index(t_path, ts) for ts in t_sky], dtype=int)
    az_k = az_path[idxs]
    el_k = el_path[idxs]
    fl_k = flag_path[idxs]
    pw_k = power

    # Keep only "on-scan" and above signal threshold
    mask = (fl_k >= 1.0) & (pw_k >= cfg.signal_min)
    return az_k[mask], el_k[mask], pw_k[mask]


# ----------------------------
# Grid and fitting
# ----------------------------

def _make_grid(az: np.ndarray,
               el: np.ndarray,
               pw: np.ndarray,
               grid_step_deg: float):
    """
    Build a regular 2D grid (az, el) and interpolate power onto it using griddata.
    For large datasets, consider replacing griddata with a faster custom binning.
    """
    az_min, az_max = np.min(az), np.max(az)
    el_min, el_max = np.min(el), np.max(el)

    pad_az = grid_step_deg
    pad_el = grid_step_deg
    az_lin = np.arange(az_min - pad_az, az_max + pad_az, grid_step_deg)
    el_lin = np.arange(el_min - pad_el, el_max + pad_el, grid_step_deg)
    AZ, EL = np.meshgrid(az_lin, el_lin)

    points = np.column_stack((az, el))
    PW = griddata(points, pw, (AZ, EL), method="linear", fill_value=np.nan)
    PW = np.nan_to_num(PW, copy=False, nan=0.0)
    return AZ, EL, PW


def _gauss2d(coords, A, x0, y0, sx, sy, theta, C):
    """
    Elliptical 2D Gaussian with rotation.
    coords: (2, N) array; coords[0] = X.ravel(), coords[1] = Y.ravel()
    Returns flattened model of shape (N,).
    """
    x, y = coords
    cos_t = np.cos(theta); sin_t = np.sin(theta)
    x_ = (x - x0) * cos_t + (y - y0) * sin_t
    y_ = -(x - x0) * sin_t + (y - y0) * cos_t
    g = A * np.exp(-0.5 * ((x_/sx)**2 + (y_/sy)**2)) + C
    return g.ravel()


def _fit_gaussian2d(AZ, EL, PW, peak_i: int, peak_j: int, window: int = 5):
    """
    Fit an elliptical 2D Gaussian in a (2*window+1)^2 neighborhood around (peak_i, peak_j).
    Returns (az_c, el_c, ok).
    """
    i0, j0 = peak_i, peak_j
    i1, j1 = max(0, i0 - window), max(0, j0 - window)
    i2, j2 = min(PW.shape[0], i0 + window + 1), min(PW.shape[1], j0 + window + 1)

    subZ = AZ[i1:i2, j1:j2]
    subE = EL[i1:i2, j1:j2]
    subP = PW[i1:i2, j1:j2]

    if subP.size < 9:
        return float(AZ[i0, j0]), float(EL[i0, j0]), False

    X = subZ.ravel()
    Y = subE.ravel()
    Z = subP.ravel()

    A0 = float(np.max(Z) - np.median(Z))
    x0 = float(AZ[i0, j0])
    y0 = float(EL[i0, j0])
    sx0 = sy0 = max((AZ[0,1]-AZ[0,0])*2, (EL[1,0]-EL[0,0])*2)
    theta0 = 0.0
    C0 = float(np.median(Z))

    p0 = [A0, x0, y0, sx0, sy0, theta0, C0]
    bounds = (
        [0.0, x0-2*sx0, y0-2*sy0, 1e-3, 1e-3, -np.pi, -np.inf],
        [np.inf, x0+2*sx0, y0+2*sy0, 10.0, 10.0,  np.pi,  np.inf],
    )

    try:
        popt, _ = curve_fit(_gauss2d, (X, Y), Z, p0=p0, bounds=bounds, maxfev=5000)
        _, x0, y0, _, _, _, _ = popt
        return float(x0), float(y0), True
    except Exception:
        total = np.sum(subP)
        if total <= 0:
            return float(AZ[i0, j0]), float(EL[i0, j0]), False
        az_c = float(np.sum(subZ * subP) / total)
        el_c = float(np.sum(subE * subP) / total)
        return az_c, el_c, False


# ----------------------------
# Sun position (Astropy)
# ----------------------------

def _sun_altaz(map_timestamp_iso: str, site: Site, cfg: Config):
    """
    Compute the Sun's apparent AltAz at the map timestamp for the given site.
    Atmospheric refraction is OFF by default. If cfg.refraction == 'simple',
    you may adapt this function to include pressure/temperature in the AltAz frame.
    """
    t = Time(map_timestamp_iso)
    loc = EarthLocation(lat=site.latitude_deg*u.deg,
                        lon=site.longitude_deg*u.deg,
                        height=site.elevation_m*u.m)
    frame = AltAz(obstime=t, location=loc)  # no pressure/temp → refraction off
    sun = get_sun(t).transform_to(frame)
    return float(sun.az.degree), float(sun.alt.degree)


# ----------------------------
# Public API
# ----------------------------

def compute_offsets(maps: Iterable[MapInput], site: Site, cfg: Config, write_row: WriterFn) -> None:
    """
    Compute offsets for all maps and append results via `write_row`.
    """
    grid_step = cfg.grid_step_deg if cfg.grid_step_deg is not None else max(0.01, cfg.fwhm_deg / 3.0)
    smooth_sig = cfg.smooth_sigma_deg if cfg.smooth_sigma_deg is not None else max(0.005, cfg.fwhm_deg / 4.0)

    for mp in maps:
        t_path, az_path, el_path, fl_path = _load_path(mp.path_file)
        t_sky, power = _load_sky(mp.sky_file)

        az_s, el_s, pw_s = _sync_power_to_pointing(
            t_path, az_path, el_path, fl_path, t_sky, power, cfg
        )
        if len(pw_s) == 0:
            continue

        AZ, EL, PW = _make_grid(az_s, el_s, pw_s, grid_step_deg=grid_step)
        if smooth_sig > 0:
            sigma_pix = smooth_sig / grid_step
            PW_s = gaussian_filter(PW, sigma=sigma_pix, mode="nearest")
        else:
            PW_s = PW

        peak_idx_flat = int(np.argmax(PW_s))
        peak_i, peak_j = np.unravel_index(peak_idx_flat, PW_s.shape)

        az_meas, el_meas, _ok = _fit_gaussian2d(AZ, EL, PW_s, peak_i, peak_j, window=5)

        az_sun, el_sun = _sun_altaz(mp.map_timestamp_iso, site, cfg)
        d_az = az_meas - az_sun
        d_el = el_meas - el_sun

        write_row(mp.map_timestamp_iso, az_meas, el_meas, d_az, d_el)
