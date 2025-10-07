
from __future__ import annotations

"""
model.py
========
Shared data models and type aliases used by the pointing offsets runner and
by algorithm plugins.

Keep this module minimal and stable so that multiple algorithm implementations
can rely on a consistent API.
"""

from dataclasses import dataclass
from typing import Callable, Optional


@dataclass
class Site:
    """
    Observer site information needed to compute solar ephemeris.

    Attributes
    ----------
    name : str
        Human-readable site name (e.g., "OASI, Antarctica"). Also written
        to the offset_io Metadata header.
    latitude_deg : float
        Geographic latitude in degrees (north positive).
    longitude_deg : float
        Geographic longitude in degrees (east positive).
    elevation_m : float
        Elevation above mean sea level, in meters.
    """
    name: str
    latitude_deg: float
    longitude_deg: float
    elevation_m: float = 0.0


@dataclass
class MapInput:
    """
    One input map described by a paired .path and .sky file.

    Notes
    -----
    - `map_id` is derived from filenames as the prefix before the first underscore
      (e.g., "250106T010421_OASI.path" → map_id = "250106T010421").
    - `map_timestamp_iso` is the canonical UTC timestamp derived from map_id,
      formatted as ISO 8601 with trailing 'Z' (e.g., "2025-01-06T01:04:21Z").
    """
    map_id: str                 # e.g., '250106T010421'
    path_file: str              # absolute or relative path to *.path
    sky_file: str               # absolute or relative path to *.sky
    map_timestamp_iso: str      # 'YYYY-MM-DDTHH:MM:SSZ'


@dataclass
class Config:
    """
    Algorithm configuration container.

    Plugin modules may ignore fields they do not use. The runner will pass this
    object with values coming from defaults, config files, and CLI overrides.

    Parameters are intentionally general:
      - The recommended method is 'auto' (2D Gaussian with robust fallback).
      - FWHM/grid/smoothing tune the 2D reconstruction.
      - Power threshold and subscan duration filter noisy/short data.
    """
    method: str = "auto"               # 'auto' | 'gauss2d' | 'boresight1d'
    fwhm_deg: float = 0.2
    grid_step_deg: Optional[float] = None
    smooth_sigma_deg: Optional[float] = None
    power_thresh_frac: float = 0.75
    subscan_min_sec: float = 1.0
    signal_min: float = 30000.0
    refraction: str = "off"            # 'off' | 'simple'
    make_plots: bool = True
    verbose: bool = False


# WriterFn: callback provided by the runner. Algorithms MUST call this exactly
# once per processed map. The runner will implement it using offset_io.
WriterFn = Callable[[str, float, float, float, float], None]
# Signature:
#   write_row(timestamp_iso, azimuth_deg, elevation_deg, offset_az_deg, offset_el_deg)
