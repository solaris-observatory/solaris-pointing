#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Solar pointing offsets — compatibility algorithm (A–D)
=====================================================

This file implements a **compatibility mode** to mirror the legacy pipeline
you’re comparing against, applying exactly the points **A–D** discussed:

A) **No interpolation** for observed coordinates:
   use the **nearest** `.path` sample to the chosen centroid time (instead of
   interpolating az/el).
B) **Centroid time per scan**: simple **mean of times** whose **signal ≥ 0.75 ×
   (scan max)**, computed from `.sky` samples **within that scan’s time range**.
C) **Which scan to use**: among the scans (F0==1 segments) whose *central
   power* (mean of the above-threshold samples) is **≥ 0.75 × MaxCentralPower**
   (MaxCentralPower = max of those “central powers” across scans), pick the
   **middle one by index** (i.e., the median scan among the selected ones).
D) **Ephemerides via PySolar** (instead of Astropy).

All other choices are kept minimal and explicit to reduce degrees of freedom.
This lets you perform **apples-to-apples** comparisons with your legacy
`main_new.py` flow.

-------------------------------------------------------------------------------
(Original problem statement and data description)
-------------------------------------------------------------------------------

Purpose
-------
Compute telescope pointing offsets in azimuth and elevation (Δaz, Δel) when
mapping the Sun. Each map consists of a pair of files with the same base name
(`map_id`) and different extensions: `<map_id>.path` and `<map_id>.sky`. The
goal is to estimate the observed centroid time/position of the Sun from the
scan data, compare it to the ideal ephemeris position at that time and site,
and report the offsets:

    Δaz = az_observed_centroid  −  az_ephemeris_centroid
    Δel = el_observed_centroid  −  el_ephemeris_centroid

Context and key idea
--------------------
Because the telescope has a non-zero pointing offset, the Sun is not centered
in the scans. Even in the scan that crosses the true solar center, the signal
peak does not fall at the geometric center of the scan timeline. We therefore
(1) identify the scan that most likely passes nearest the solar center and
(2) estimate the centroid time along that scan using a simple rule.

Input files
-----------
Files come in pairs, one pair per map. The file name is the `map_id` and only
the extension differs.

1) Path file: `<map_id>.path`
   Columns (tab-separated):

       Posix_time   ms   UTC                         Azimuth   Elevation
       Azimuth_raw  Elevation_raw   F0   F1

   Example first data row:

       1735761254   310  2025-01-01T19:54:14.310     83.92568  26.04125
       7822440      2427220          0    0

   Fields used:
   - `UTC` (3rd column): timestamp of the pointing coordinates.
   - `Azimuth` (4th) and `Elevation` (5th): telescope coordinates at `UTC`.
   - `F0` (8th): scan flag. `F0 = 1` means the telescope is actively scanning
     (in RA for this dataset). `F0 = 0` means idle/transition. Only rows with
     `F0 = 1` are used to form scans.

2) Sky file: `<map_id>.sky`
   Columns (tab-separated):

       Posix_time   ms   UTC                         Signal

   Example first data row:

       1735761254   334  2025-01-01T19:54:14.334     23166

   Fields used:
   - `UTC` (3rd): timestamp of signal measurement.
   - `Signal` (4th): measured signal.

Temporal alignment between files
--------------------------------
Timestamps in `.path` and `.sky` do not generally coincide. In this compatibility
mode we do **not** do a pointwise regridding; instead, for each **scan** (a run
of `F0 == 1` in the `.path` timeline), we collect `.sky` samples whose `UTC`
falls **within the time range** of that scan and analyze them.

Scan segmentation
-----------------
The observation consists of scans in RA. Use `F0` to segment the sequence:
consecutive rows with `F0 = 1` define a single scan. Between scans, `F0` goes
to 0; then another `F0 = 1` segment marks the next scan.

Compatibility details (A–D)
---------------------------
A) Observed az/el at the centroid time are taken from the **nearest** `.path`
   row in time (no interpolation).
B) The **centroid time** of a scan is the **mean** of the `.sky` timestamps in
   that scan where `Signal ≥ 0.75 × (scan max)`. Before this, you may apply an
   **absolute threshold** on `.sky` samples (default `Signal > 30000`) to ignore
   obvious background.
C) **Scan choice**: compute a “central power” per scan as the **mean** of the
   `.sky` values used in (B). Let `MaxCentralPower` be the maximum of these
   central powers across scans. Keep only scans with
   `central_power ≥ 0.75 × MaxCentralPower`, and among them pick the **middle
   one by index** (median in time order).
D) Ephemerides are computed with **PySolar** (`get_azimuth`, `get_altitude`).

Azimuth bias and wrapping
-------------------------
A fixed **azimuth bias of +0.75°** is applied to the **observed** azimuth, then
azimuth values are **wrapped to [0, 360)**. Offsets are raw differences:

    Δaz = az_observed_centroid  −  az_ephemeris_centroid
    Δel = el_observed_centroid  −  el_ephemeris_centroid

Minimal outputs
---------------
For each map (i.e., `<map_id>.path` + `<map_id>.sky`), append one TSV line with:

- `map_id`
- `centroid_utc`
- `azimuth_deg_observed`
- `elevation_deg_observed`
- `delta_az_deg`
- `delta_el_deg`

Assumptions and conventions
---------------------------
- `.path` az/el are telescope coordinates at `UTC` (no extra transforms).
- Only rows with `F0 = 1` represent valid scan data.
- `.sky` and `.path` timestamps are both UTC in ISO-8601 format.
- Site coordinates used for ephemerides: longitude = **164.1000°**,
  latitude = **−74.6933°**, height not used by PySolar here.

Note on the choice of Astropy over PySolar
------------------------------------------
Astropy is used for solar ephemerides instead of PySolar because it provides
astronomical accuracy consistent with IAU standards. Astropy’s solar position
is computed through ERFA (the official C implementation of the IAU
astrometry algorithms), including effects such as precession, nutation,
aberration of light, relativistic corrections, polar motion, and UT1–UTC
offsets. This results in sub-arcsecond precision in both azimuth and elevation.

By contrast, PySolar relies on empirical NOAA/NREL formulas originally designed
for photovoltaic and meteorological applications. These simplified models treat
the Earth as a perfect sphere, ignore site altitude and Earth orientation
parameters, and apply a fixed atmospheric refraction correction corresponding
to standard sea-level conditions (≈1013 hPa, 10 °C). Consequently, PySolar
introduces systematic errors of up to several arcminutes, especially at low
solar elevations.

Astropy allows specifying pressure, temperature, humidity, and observing
wavelength, enabling a physically consistent and site-specific treatment of
refraction (or its full exclusion if desired). When configured with realistic
meteorological parameters, Astropy reproduces the apparent solar position to
better than a few arcseconds and ensures internal consistency with professional
astronomical frameworks.


Roadmap (if you want to extend this file)
-----------------------------------------
- Make absolute and relative thresholds configurable.
- Add optional pre-cleaning of `.sky` glitches (simple MAD) without altering the
  A–D behavior by default.
- Add a switch to choose “median” instead of “mean” for the central power.
"""

import os
import glob
import csv
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import List, Tuple, Optional

import numpy as np

from solaris_pointing.offset_core.offset_io import (
    Metadata,
    Measurement,
    write_offsets_tsv,
)

# --- Site coordinates (fixed as requested) ---
SITE_LON_DEG = 164.1000
SITE_LAT_DEG = -74.6933

# --- Azimuth bias and wrapping ---
AZIMUTH_BIAS_DEG = 0.75

# --- Sky thresholds (compat mode) ---
ABS_SIGNAL_MIN = 30000.0           # absolute filter on sky samples (legacy-like)
REL_PEAK_FRAC = 0.75               # keep samples >= 75% of scan max power
REL_CENTRAL_POWER_FRAC = 0.75      # keep scans with central_power >= 75% of MaxCentralPower


@dataclass
class PathRow:
    t_utc: datetime
    az_deg: float
    el_deg: float
    f0: int


@dataclass
class SkyRow:
    t_utc: datetime
    signal: float


def parse_iso_utc(s: str) -> datetime:
    # Example: "2025-01-01T19:54:14.310"
    dt = datetime.fromisoformat(s)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    else:
        dt = dt.astimezone(timezone.utc)
    return dt


def read_path_file(path_fname: str) -> List[PathRow]:
    out: List[PathRow] = []
    with open(path_fname, "r", newline="") as f:
        r = csv.reader(f, delimiter="\t")
        header = next(f, None)  # raw header line
        if header is None:
            return out
        # Use csv reader again to parse after header
        f.seek(len(header))
        r = csv.reader(f, delimiter="\t")
        # robust: locate columns by name
        header_cols = header.strip().split("\t")
        name_to_idx = {name: i for i, name in enumerate(header_cols)}
        idx_utc = name_to_idx["UTC"]
        idx_az = name_to_idx["Azimuth"]
        idx_el = name_to_idx["Elevation"]
        idx_f0 = name_to_idx["F0"]
        for row in r:
            if not row:
                continue
            try:
                t = parse_iso_utc(row[idx_utc])
                az = float(row[idx_az])
                el = float(row[idx_el])
                f0 = int(row[idx_f0])
                out.append(PathRow(t, az, el, f0))
            except Exception:
                continue
    return out


def read_sky_file(sky_fname: str) -> List[SkyRow]:
    out: List[SkyRow] = []
    with open(sky_fname, "r", newline="") as f:
        r = csv.reader(f, delimiter="\t")
        header = next(f, None)
        if header is None:
            return out
        f.seek(len(header))
        r = csv.reader(f, delimiter="\t")
        header_cols = header.strip().split("\t")
        name_to_idx = {name: i for i, name in enumerate(header_cols)}
        idx_utc = name_to_idx["UTC"]
        idx_sig = name_to_idx["Signal"]
        for row in r:
            if not row:
                continue
            try:
                t = parse_iso_utc(row[idx_utc])
                s = float(row[idx_sig])
                out.append(SkyRow(t, s))
            except Exception:
                continue
    return out


def find_scan_segments(f0_array: np.ndarray) -> List[Tuple[int, int]]:
    """Return [start,end) index ranges for contiguous runs with F0 == 1."""
    segs = []
    in_run = False
    start = 0
    for i, v in enumerate(f0_array):
        if v == 1 and not in_run:
            in_run = True
            start = i
        elif v != 1 and in_run:
            in_run = False
            segs.append((start, i))
    if in_run:
        segs.append((start, len(f0_array)))
    return segs


def to_unix_s(dt: datetime) -> float:
    return dt.timestamp()


def choose_scan_and_centroid_time(
    t_path: np.ndarray,
    f0: np.ndarray,
    sky_times: np.ndarray,
    sky_vals: np.ndarray,
) -> Optional[Tuple[Tuple[int, int], float, float]]:
    """
    Return (segment, t_centroid, central_power) where:
      - segment = (a,b) indices into path arrays for the chosen scan
      - t_centroid = mean time of sky samples >= REL_PEAK_FRAC * scan_max within [t[a], t[b-1]]
      - central_power = mean( sky >= REL_PEAK_FRAC * scan_max ) for that scan
    Selection:
      - compute central_power per scan;
      - keep scans with central_power >= REL_CENTRAL_POWER_FRAC * MaxCentralPower;
      - choose the **middle one by index** among the kept scans.
    """
    segs = find_scan_segments(f0)
    if not segs:
        return None

    # Pre-filter sky by absolute threshold (legacy-like)
    sky_mask_abs = sky_vals > ABS_SIGNAL_MIN
    sky_times_abs = sky_times[sky_mask_abs]
    sky_vals_abs = sky_vals[sky_mask_abs]

    if sky_times_abs.size == 0:
        return None

    chosen_list = []  # (seg_index, (a,b), t_centroid, central_power)

    # Compute per-scan central power and centroid time
    for seg_idx, (a, b) in enumerate(segs):
        if b <= a:
            continue
        t0 = t_path[a]
        t1 = t_path[b - 1]

        # select sky within scan time window
        i0 = np.searchsorted(sky_times_abs, t0, side="left")
        i1 = np.searchsorted(sky_times_abs, t1, side="right")
        if i1 <= i0:
            continue

        st = sky_times_abs[i0:i1]
        sv = sky_vals_abs[i0:i1]
        if sv.size == 0:
            continue

        scan_max = np.max(sv)
        if scan_max <= 0:
            continue

        keep = sv >= (REL_PEAK_FRAC * scan_max)
        if not np.any(keep):
            continue

        st_sel = st[keep]
        sv_sel = sv[keep]

        # centroid time = mean of times above relative threshold
        t_centroid = float(np.mean(st_sel))
        # central power = mean of powers above relative threshold
        central_power = float(np.mean(sv_sel))

        chosen_list.append((seg_idx, (a, b), t_centroid, central_power))

    if not chosen_list:
        return None

    # Keep scans with central_power ≥ REL_CENTRAL_POWER_FRAC × MaxCentralPower
    central_powers = np.array([cp for _, _, _, cp in chosen_list], dtype=float)
    max_cp = float(np.max(central_powers))
    thr_cp = REL_CENTRAL_POWER_FRAC * max_cp

    kept = [(i, seg, tc, cp) for (i, seg, tc, cp) in chosen_list if cp >= thr_cp]
    if not kept:
        return None

    # Pick the **middle** by index among the kept scans (median scan index)
    kept_sorted = sorted(kept, key=lambda x: x[0])
    mid_idx = len(kept_sorted) // 2
    _, seg_chosen, t_centroid_chosen, central_power_chosen = kept_sorted[mid_idx]

    return seg_chosen, t_centroid_chosen, central_power_chosen


def wrap_az(az: float) -> float:
    return az % 360.0


def nearest_path_observed(
    t_centroid_s: float, t_path: np.ndarray, az_path: np.ndarray, el_path: np.ndarray
) -> Tuple[float, float]:
    """Observed az/el at centroid time using the NEAREST .path sample (no interpolation)."""
    idx = int(np.searchsorted(t_path, t_centroid_s))
    # pick nearer between idx-1 and idx
    if idx <= 0:
        k = 0
    elif idx >= len(t_path):
        k = len(t_path) - 1
    else:
        left = idx - 1
        k = left if (t_centroid_s - t_path[left]) <= (t_path[idx] - t_path[left]) else idx
    az_obs = wrap_az(float(az_path[k]) + AZIMUTH_BIAS_DEG)
    el_obs = float(el_path[k])
    return az_obs, el_obs


def compute_ephem(dt_utc: datetime, lat_deg: float, lon_deg: float) -> Tuple[float, float]:
    """Return (az_deg, el_deg) using Astropy. Azimuth wrapped to [0, 360)."""
    # --- Ephemerides via Astropy ---
    try:
        from astropy.time import Time
        from astropy.coordinates import EarthLocation, AltAz, get_sun
        import astropy.units as u
    except Exception as e:
        raise SystemExit("This mode requires Astropy. Please `pip install astropy`.") from e

    loc = EarthLocation(lat=lat_deg * u.deg, lon=lon_deg * u.deg, height= 50 * u.m)
    t = Time(dt_utc, scale="utc")
    altaz = AltAz(
        obstime=t,
        location=loc,
        pressure=950 * u.hPa,
        temperature=-5 * u.deg_C,
        relative_humidity=0.2,
        obswl=3 * u.mm
    )
    sun = get_sun(t).transform_to(altaz)
    az = (sun.az.to(u.deg).value) % 360.0
    el = sun.alt.to(u.deg).value
    return az, el


def process_map(map_id: str, path_fname: str, sky_fname: str) -> Optional[Tuple[str, str, float, float, float, float]]:
    """
    Compatibility pipeline for one map_id:
      - segment scans via F0 from .path
      - for each scan, collect .sky samples in its time window, apply ABS_SIGNAL_MIN
      - compute t_centroid (mean times with power >= 0.75*scan_max)
      - compute central_power (mean power of those samples)
      - keep scans with central_power >= 0.75*MaxCentralPower
      - choose middle-by-index scan among kept; use nearest .path sample at t_centroid
      - compute ephemerides with PySolar; return offsets
    """
    path_rows = read_path_file(path_fname)
    sky_rows = read_sky_file(sky_fname)
    if not path_rows or not sky_rows:
        return None

    t_path = np.array([to_unix_s(r.t_utc) for r in path_rows], dtype=float)
    az_path = np.array([r.az_deg for r in path_rows], dtype=float)
    el_path = np.array([r.el_deg for r in path_rows], dtype=float)
    f0 = np.array([r.f0 for r in path_rows], dtype=int)

    sky_times = np.array([to_unix_s(r.t_utc) for r in sky_rows], dtype=float)
    sky_vals = np.array([r.signal for r in sky_rows], dtype=float)

    choice = choose_scan_and_centroid_time(t_path, f0, sky_times, sky_vals)
    if choice is None:
        return None
    (a, b), t_centroid_s, _central_power = choice

    # Observed coordinates: NEAREST .path sample
    az_obs, el_obs = nearest_path_observed(t_centroid_s, t_path, az_path, el_path)

    # Ephemerides from PySolar
    dt_centroid = datetime.fromtimestamp(t_centroid_s, tz=timezone.utc)
    az_eph, el_eph = compute_ephem(dt_centroid, SITE_LAT_DEG, SITE_LON_DEG)

    delta_az = az_obs - az_eph
    delta_el = el_obs - el_eph

    centroid_iso = dt_centroid.isoformat(timespec="milliseconds").replace("+00:00", "Z")
    return (map_id, centroid_iso, az_obs, el_obs, delta_az, delta_el)


def find_map_pairs() -> List[Tuple[str, str, str]]:
    """Return (map_id, path_fname, sky_fname) pairs in the data directory."""
    path_files = glob.glob("data2/*.path")
    sky_set = set(glob.glob("data2/*.sky"))
    pairs = []
    for p in path_files:
        base = os.path.splitext(p)[0]
        s = base + ".sky"
        if s in sky_set:
            pairs.append((base, p, s))
    return sorted(pairs)


def append_result_tsv(out_fname: str, row: Tuple[str, str, float, float, float, float]) -> None:
    # Metadata will be added to the header of the file
    md = Metadata(
        location="MZS, Antarctica",
        antenna_diameter_m=2.0,
        frequency_ghz=100,
        software_version="2025.08.05",
    )

    map_id, timestamp, az, el, offset_az, offset_el = row
    record = [
        Measurement(
            map_id=map_id,
            timestamp_iso=timestamp,
            azimuth_deg=az,
            elevation_deg=el,
            offset_az_deg=offset_az,
            offset_el_deg=offset_el,
            temperature_c=None,
            pressure_hpa=None,
            humidity_frac=None,
        )
    ]
    write_offsets_tsv(out_fname, md, record, append=True)


def main():
    out_fname = "buttu_1.tsv"
    pairs = find_map_pairs()
    if not pairs:
        print("No <map_id>.path / <map_id>.sky pairs found in current directory.")
        return
    for map_id, path_fname, sky_fname in pairs:
        try:
            res = process_map(map_id, path_fname, sky_fname)
            if res is None:
                print(f"[WARN] Skipping {map_id}: could not compute offsets (compat).")
                continue
            append_result_tsv(out_fname, res)
            print(f"[OK] {map_id} -> appended to {out_fname}")
        except Exception as e:
            print(f"[ERROR] {map_id}: {e}")


if __name__ == "__main__":
    main()

