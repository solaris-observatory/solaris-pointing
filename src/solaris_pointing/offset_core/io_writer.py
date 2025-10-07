
from __future__ import annotations

"""
io_writer.py
============

A thin adapter around `offset_io` (relocated under
`solaris_pointing.offset_core.offset_io`) that produces a `write_row(...)`
callback for algorithm plugins.

Design
------
- Algorithm plugins NEVER write files; they call `write_row(...)` once per map.
- The runner creates this writer using metadata and a map to environment lookup.
- We keep the `offset_io` dependency contained here so plugins do not have to
  import it or care about the on-disk format.

Field names
-----------
Aligned with the official offset_io schema:
  Measurement(
    timestamp_iso=...,           # str
    azimuth_deg=...,             # float
    elevation_deg=...,           # float
    offset_az_deg=...,           # float
    offset_el_deg=...,           # float
    temperature_c=...,           # Optional[float]
    pressure_hpa=...,            # Optional[float]
    humidity_frac=...,           # Optional[float]
  )
"""

from typing import Dict, Optional, Tuple

# NOTE: offset_io is expected under offset_core, as requested.
from .offset_io import Metadata, Measurement, write_offsets_tsv


def writer_factory(output_path: str,
                   md: Metadata,
                   env_by_map: Dict[str, Tuple[Optional[float], Optional[float], Optional[float]]],
                   ts_to_map: Dict[str, str]):
    """
    Build and return a `write_row(timestamp_iso, az, el, daz, del_)` function that
    appends a single `Measurement` to the TSV via `offset_io`.

    Parameters
    ----------
    output_path : str
        Path to the TSV file. The file will be created if missing; subsequent
        calls will append only data rows (header is written once by offset_io).
    md : Metadata
        Header metadata to be written once (location, dish, frequency, version).
    env_by_map : dict
        Mapping `map_id -> (temperature_c, pressure_hpa, humidity_frac)`.
    ts_to_map : dict
        Mapping `timestamp_iso -> map_id`, used to attach environment by map id.

    Returns
    -------
    Callable[[str, float, float, float, float], None]
        The writer function to give to algorithm plugins.
    """
    def write_row(timestamp_iso: str, az: float, el: float, daz: float, del_: float) -> None:
        # Resolve map_id from timestamp and attach environment if present
        map_id = ts_to_map.get(timestamp_iso)
        t_c = p_hpa = h_frac = None
        if map_id is not None and map_id in env_by_map:
            t_c, p_hpa, h_frac = env_by_map[map_id]

        rows = [Measurement(
            timestamp_iso=timestamp_iso,
            azimuth_deg=float(az),
            elevation_deg=float(el),
            offset_az_deg=float(daz),
            offset_el_deg=float(del_),
            temperature_c=t_c,
            pressure_hpa=p_hpa,
            humidity_frac=h_frac,
        )]
        write_offsets_tsv(output_path, md, rows, append=True)

    return write_row
