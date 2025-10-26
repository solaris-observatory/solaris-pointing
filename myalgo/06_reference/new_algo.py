#!/usr/bin/env python3
"""
Solar pointing offsets
======================

Compute telescope pointing offsets (daz, del) between the observed and
apparent solar positions using Astropy ephemerides. The algorithm identifies
the scan that passes nearest the solar center, estimates its centroid time,
and derives the pointing offsets in azimuth and elevation for each map.

Overview of algorithm
---------------------
The procedure follows four main rules:

A) Select scans whose signal peak exceeds 75% of the global maximum and whose
   total power is above 75% of the maximum total power.

B) Among these scans, compute the centroid time as the mean of timestamps
   corresponding to samples with signal >= 0.75 x (scan max).

C) Choose the scan whose "central power" (mean of those high-signal samples)
   is >= 0.75 x MaxCentralPower and which lies in the median time position
   among the selected scans.

D) Compute the solar apparent position via Astropy (`get_sun`, `AltAz`)
   including atmospheric refraction, and calculate daz, del between observed
   and ephemeris coordinates.

Input data
----------
Each map is defined by two tab-separated files sharing the same base name
(`map_id`):

1) <map_id>.path
   Columns used:
   - UTC (ISO-8601): timestamp of telescope coordinates
   - Azimuth, Elevation: telescope pointing angles in degrees
   - F0: scan flag; 1 = active scanning, 0 = idle

2) <map_id>.sky
   Columns used:
   - UTC (ISO-8601): timestamp of signal samples
   - Signal: measured power or brightness temperature

Timestamps from the two files are not synchronized sample-by-sample.
For each scan (F0 == 1 in the .path file), .sky samples within the scan time
range are associated and analyzed together.

Scan analysis and centroid determination
----------------------------------------
- Consecutive F0 == 1 rows in .path define a scan.
- For each scan, compute the maximum signal in .sky.
- Select .sky samples where Signal >= 0.75 x (scan max).
- The centroid time is the mean of these timestamps.
- The observed azimuth/elevation are taken from the nearest .path row
  (no interpolation) corresponding to that centroid time.
- Each scan's "central power" is the mean of the high-signal samples.
- The scan whose central power >= 0.75 x MaxCentralPower and is temporally
  median among the candidates is used to compute the offsets.

Computation of offsets
----------------------
Offsets are computed as:

    daz = az_observed_centroid - az_ephemeris_centroid
    del = el_observed_centroid - el_ephemeris_centroid

Azimuth bias and wrapping
-------------------------
- A fixed azimuth bias of +0.75 deg is applied to the observed azimuths
  (instrumental encoder convention).
- Azimuths are wrapped to the [0, 360) deg range so that offsets correspond
  to the shortest angular distance.

Coordinate and atmospheric model
--------------------------------
All positions are expressed in the apparent AltAz frame as computed by
Astropy. Atmospheric refraction is included through the following parameters,
typical for MZS summer conditions and a 2 m antenna operating near 100 GHz:

- pressure = 990 hPa
- temperature = -5 C
- relative_humidity = 0.2
- obswl = 3 mm  (approx. 100 GHz)

The observing site is defined as:
latitude = -74.694 deg, longitude = 164.120 deg, height = 50 m.
The height is used by EarthLocation to compute the local geocentric position.

Output format
-------------
One line per map is appended to the output TSV file with columns:

- map_id
- centroid_utc
- azimuth_deg_observed
- elevation_deg_observed
- delta_az_deg
- delta_el_deg

Metadata stored in the file header include:
- Location: MZS, Antarctica
- Antenna diameter: 2.0 m
- Observing frequency: 100 GHz
- Software version: current repository revision

Physical meaning and rationale for Astropy
------------------------------------------
Astropy's ephemeris engine provides a physically consistent solar position
through ERFA, implementing the official IAU astrometry algorithms and
accounting for precession, nutation, aberration, relativistic effects,
polar motion, and UT1-UTC corrections. It also models refraction using the
actual observer pressure, temperature, humidity, and wavelength, which is
essential for low-elevation solar observations typical in Antarctica.

By contrast, PySolar employs simplified empirical formulas derived from
NOAA/NREL solar-position models intended for photovoltaic applications.
These neglect site altitude, polar motion, and most atmospheric dependencies,
assuming standard sea-level conditions (~1013 hPa, 10 C). Consequently,
PySolar can introduce systematic errors of several arcminutes near the
horizon.

For these reasons, Astropy ensures sub-arcminute consistency and is preferred
for scientific pointing calibration.

Possible extensions
-------------------
- Make thresholds and atmospheric parameters configurable via CLI options.
- Add optional pre-filtering of .sky glitches (e.g., using MAD).
- Provide uncertainty estimates on daz, del using scan-to-scan variance.
- Support median-based centroid estimation as an alternative to the mean.
"""

import os
import glob
import csv
from pathlib import Path
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
REL_PEAK_FRAC = 0.75               # keep samples >= 80% of scan max power
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
        # i0 = np.searchsorted(sky_times_abs, t0, side="left")
        # i1 = np.searchsorted(sky_times_abs, t1, side="right")
        i0 = np.searchsorted(sky_times_abs, t0, side="left")   # include t0
        i1 = np.searchsorted(sky_times_abs, t1, side="left")   # exclude t1

        if i1 <= i0:
            continue

        st = sky_times_abs[i0:i1]
        sv = sky_vals_abs[i0:i1]
        if sv.size == 0:
            continue

        # More robust than simple max, less sensitive to single-sample spikes
        scan_max = np.percentile(sv, 99.9)
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
        left = idx - 1
        dist_left = abs(t_centroid_s - t_path[left])
        dist_right = abs(t_path[idx] - t_centroid_s)
        k = left if dist_left <= dist_right else idx
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
        pressure=990 * u.hPa,
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
      - compute ephemerides with Astropy; return offsets
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

    dt_centroid = datetime.fromtimestamp(t_centroid_s, tz=timezone.utc)
    az_eph, el_eph = compute_ephem(dt_centroid, SITE_LAT_DEG, SITE_LON_DEG)

    delta_az = az_obs - az_eph
    delta_az = (delta_az + 180.0) % 360.0 - 180.0
    delta_el = el_obs - el_eph

    centroid_iso = dt_centroid.isoformat(timespec="milliseconds").replace("+00:00", "Z")
    return (map_id, centroid_iso, az_obs, el_obs, delta_az, delta_el)


def find_map_pairs() -> List[Tuple[str, str, str]]:
    """Return (map_id, path_fname, sky_fname) pairs in the data directory."""
    path_files = glob.glob("data1/*.path")
    sky_set = set(glob.glob("data1/*.sky"))
    pairs = []
    for p in path_files:
        base = os.path.splitext(p)[0]
        s = base + ".sky"
        map_id = Path(base).name
        if s in sky_set:
            pairs.append((map_id, p, s))
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
    script_path = Path(__file__)
    suffix = '.tsv'
    stem = script_path.stem + suffix
    out_fname = script_path.with_suffix(suffix)
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
            print(f"[OK] {map_id} -> appended to {stem}")
        except Exception as e:
            print(f"[ERROR] {map_id}: {e}")


if __name__ == "__main__":
    main()

