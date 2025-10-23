from __future__ import annotations

"""
io_writer.py
============

A thin adapter around ``offset_io`` that produces a ``write_row(...)`` callback
for algorithm plugins.

Design
------
- Algorithm plugins NEVER write files; they call ``write_row`` once per map.
- The runner creates this writer using metadata and a map->environment lookup.
- This module hides the on-disk format so plugins do not import ``offset_io``.

Schema
------
Aligned with the ``offset_io`` schema for ``Measurement`` fields (degrees).
"""

from typing import Dict, Optional, Tuple

from .offset_io import Metadata, Measurement, write_offsets_tsv


def writer_factory(
    output_path: str,
    md: Metadata,
    env_by_map: Dict[str, Tuple[Optional[float], Optional[float], Optional[float]]],
    ts_to_map: Dict[str, str],
):
    """
    Build and return a ``write_row(ts, az, el, daz, del_)`` function that appends
    a single ``Measurement`` to the TSV via ``offset_io``.

    Parameters
    ----------
    output_path : str
        Path to the TSV file. Created if missing; rows get appended. Header is
        written once by ``offset_io``.
    md : Metadata
        Static header metadata (location, antenna, frequency, version).
    env_by_map : dict
        Mapping ``map_id -> (temperature_c, pressure_hpa, humidity_frac)``.
    ts_to_map : dict
        Mapping ``timestamp_iso -> map_id`` to attach environment by map.
    """

    def write_row(ts: str, az: float, el: float, daz: float, del_: float) -> None:
        # Resolve map_id from timestamp and attach environment if present.
        map_id = ts_to_map.get(ts)
        t_c = p_hpa = h_frac = None
        if map_id is not None and map_id in env_by_map:
            t_c, p_hpa, h_frac = env_by_map[map_id]

        rows = [
            Measurement(
                timestamp_iso=ts,
                azimuth_deg=float(az),
                elevation_deg=float(el),
                offset_az_deg=float(daz),
                offset_el_deg=float(del_),
                temperature_c=t_c,
                pressure_hpa=p_hpa,
                humidity_frac=h_frac,
            )
        ]
        write_offsets_tsv(output_path, md, rows, append=True)

    return write_row
