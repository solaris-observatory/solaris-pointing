from __future__ import annotations

"""
model.py
========
Minimal data models shared across the pointing offsets pipeline.

This module is intentionally small and stable. Multiple algorithms can rely
on a consistent API without importing runners or I/O details.

All identifiers and comments are in English, and lines are <= 88 chars.
"""

from dataclasses import dataclass
from typing import Protocol, Any, Dict, Optional


# A site where observations are made.
@dataclass(frozen=True)
class Site:
    # Human readable site name.
    name: str
    # Latitude in decimal degrees (south negative).
    latitude_deg: float
    # Longitude in decimal degrees (east positive).
    longitude_deg: float
    # Elevation above mean sea level in meters.
    elevation_m: float


# A single map input record discovered by the pipeline.
@dataclass(frozen=True)
class MapInput:
    # Unique identifier of the map (e.g., file stem or timestamp).
    map_id: str
    # ISO8601 timestamp (e.g., 2025-01-03T00:02:35Z).
    map_timestamp_iso: str
    # Full path to the paired .path file.
    path_file: str
    # Full path to the paired .sky file.
    sky_file: str
    # Optional extra metadata parsed during discovery.
    meta: Optional[Dict[str, Any]] = None


class WriterFn(Protocol):
    """Callable that writes one output row to the offsets TSV."""

    def __call__(
        self,
        timestamp_iso: str,
        az_meas_deg: float,
        el_meas_deg: float,
        d_az_deg: float,
        d_el_deg: float,
    ) -> None: ...


# Minimal algorithm configuration expected by legacy algorithms.
@dataclass(frozen=True)
class Config:
    # Minimum signal threshold for sample selection (power units).
    signal_min: float = 0.0
    # Grid step in degrees for building the 2D map.
    grid_step_deg: float = 0.05
    # Smoothing sigma in degrees applied before peak finding.
    smooth_sigma_deg: float = 0.05
    # Beam full width at half maximum in degrees (used for windows).
    fwhm_deg: float = 0.5
    # Refraction mode: "none" or "apparent".
    refraction: str = "apparent"


__all__ = [
    "Site",
    "MapInput",
    "WriterFn",
    "Config",
]
