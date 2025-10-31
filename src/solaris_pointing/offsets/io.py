"""
Offset TSV writer for solar pointing maps (with `map_id` as first column).

This module writes a TSV file containing telescope pointing offsets derived
from solar maps. The output format is a text file with:
  1) A *commented* metadata block in English (lines starting with '#').
  2) A single *header* line with the column names.
  3) One data line per measurement.

Compared to the previous format, the following changes apply:
  - A new first column, `map_id`, has been added.
  - `timestamp` is now the *second* column.
  - The semantics of `timestamp` are clarified in the metadata block as the
    instant when the telescope pointed at the Sun's centroid.
  - Angles remain in *degrees*. Offsets are serialized with four decimals.

Important on separators
-----------------------
All other field separators are tabs. Concretely, the header is:

  map_id\ttimestamp\tazimuth\televation\toffset_az\toffset_el\t
  temperature\tpressure\thumidity

Units and conventions
---------------------
- azimuth_deg, elevation_deg: observed angles [deg]
- offset_az_deg, offset_el_deg: (solar − observed) angles [deg]
- temperature_c: degrees Celsius [°C]
- pressure_hpa: hectoPascal [hPa]
- humidity_frac: relative humidity in [0, 1]

Robustness and append mode
--------------------------
When appending, the module validates that the on-disk header *exactly*
matches the expected one (including the two-space special delimiter). If
there is any mismatch, a SchemaMismatchError is raised to avoid corrupting
the dataset. Overwrites use an atomic write via a temporary file.

Quickstart
----------
>>> md = Metadata(
...     location="Dome C, Antarctica",
...     antenna_diameter_m=1.2,
...     frequency_ghz=100,
...     software_version="1.4.0",
... )
>>> rows = [
...     Measurement(
...         map_id="250101T185415bOASI",
...         timestamp_iso="2025-01-03T00:02:35Z",
...         azimuth_deg=14.6622,
...         elevation_deg=37.7847,
...         offset_az_deg=-0.0829,
...         offset_el_deg=0.0700,
...         temperature_c=None,
...         pressure_hpa=None,
...         humidity_frac=None,
...     )
... ]
>>> write_offsets_tsv("example.tsv", md, rows, append=False)

"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Optional, TextIO, List


__all__ = [
    "Metadata",
    "Measurement",
    "SchemaMismatchError",
    "write_offsets_tsv",
]


class SchemaMismatchError(RuntimeError):
    """Raised when the on-disk header does not match the expected schema."""


@dataclass
class Metadata:
    """
    File-level metadata written as a commented block at the top of the file.

    Parameters
    ----------
    location : str
        Site name / location string.
    antenna_diameter_m : float
        Antenna diameter in meters.
    frequency_ghz : float
        Observing frequency in GHz. Serialized as GHz in the metadata block.
    software_version : str
        Free-form software version string (e.g., '1.4.0').
    created_at_iso : Optional[str]
        ISO-8601 UTC timestamp string for file creation. If None, the current
        time in UTC is used.
    """

    location: str
    antenna_diameter_m: float
    frequency_ghz: float
    software_version: str
    created_at_iso: Optional[str] = None

    def created_iso_or_now(self) -> str:
        """Return `created_at_iso` if provided, else now in UTC as ISO-8601."""
        if self.created_at_iso:
            return self.created_at_iso
        return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


@dataclass
class Measurement:
    """
    One pointing-offset measurement row.

    Notes
    -----
    - All angles are in **degrees**.
    - `map_id` identifies the solar map from which this row was derived.
    - `timestamp_iso` is the instant when the telescope pointed at the Sun's
      centroid (ISO-8601, UTC recommended).

    Parameters
    ----------
    map_id : str
        Map identifier (non-empty).
    timestamp_iso : str
        ISO-8601 instant, e.g. '2025-01-03T00:02:35Z'.
    azimuth_deg : float
        Azimuth of the Sun's centroid from ephemerides [deg].
    elevation_deg : float
        Elevation of the Sun's centroid from ephemerides [deg].
    offset_az_deg : float
        (observed azimuth − solar azimuth) [deg].
    offset_el_deg : float
        (observed elevation − solar elevation) [deg].
    temperature_c : Optional[float]
        Ambient temperature [°C]. If None, serialized as "NaN".
    pressure_hpa : Optional[float]
        Ambient pressure [hPa]. If None, serialized as "NaN".
    humidity_frac : Optional[float]
        Relative humidity in [0, 1]. If None, serialized as "NaN".
    """

    map_id: str
    timestamp_iso: str
    azimuth_deg: float
    elevation_deg: float
    offset_az_deg: float
    offset_el_deg: float
    temperature_c: Optional[float] = None
    pressure_hpa: Optional[float] = None
    humidity_frac: Optional[float] = None

    def __post_init__(self) -> None:
        if not self.map_id or not isinstance(self.map_id, str):
            raise ValueError("map_id must be a non-empty string")
        # Minimal ISO-8601 shape check to catch obvious mistakes
        if "T" not in self.timestamp_iso:
            raise ValueError(
                "timestamp_iso should look like ISO 8601 (e.g., 'YYYY-MM-DDTHH:MM:SSZ')"
            )


def _write_metadata_block(f: TextIO, md: Metadata) -> None:
    """
    Write the commented metadata block and column dictionary.

    Sections:
      - Telescope
      - File columns
      - Additional information
    """
    created = md.created_iso_or_now()

    # Telescope
    f.write("# ---------\n")
    f.write("# Telescope\n")
    f.write("# ---------\n")
    f.write(f"# Location: {md.location}\n")
    f.write(f"# Antenna diameter: {md.antenna_diameter_m} m\n")
    f.write(f"# Frequency: {md.frequency_ghz} GHz\n")
    f.write("#\n")

    # File columns
    f.write("# ------------\n")
    f.write("# File columns\n")
    f.write("# ------------\n")
    f.write("# map_id: map identifier\n")
    f.write(
        "# timestamp: time when the telescope pointed at the Sun's "
        "centroid [ISO 8601]\n"
    )
    f.write("# azimuth: of the Sun's centroid from ephemerides [deg]\n")
    f.write("# elevation: of the Sun's centroid from ephemerides [deg]\n")
    f.write("# offset_az: observed azimuth - solar azimuth [deg]\n")
    f.write("# offset_el: observed elevation - solar elevation [deg]\n")
    f.write("# temperature: °C\n")
    f.write("# pressure: hPa\n")
    f.write("# humidity: relative humidity, fraction (0..1)\n")
    f.write("#\n")

    # Additional information
    f.write("# ----------------------\n")
    f.write("# Additional information\n")
    f.write("# ----------------------\n")
    f.write(f"# Generated with software version: {md.software_version}\n")
    f.write(f"# Created at: {created}\n")
    f.write("\n")


def _expected_columns() -> List[str]:
    """Return the expected header token list."""
    return [
        "map_id",
        "timestamp",
        "azimuth",
        "elevation",
        "offset_az",
        "offset_el",
        "temperature",
        "pressure",
        "humidity",
    ]


def _write_column_header(f: TextIO) -> None:
    """
    Write the canonical header line.
    """
    f.write(
        "map_id\ttimestamp\tazimuth\televation\toffset_az\toffset_el\t"
        "temperature\tpressure\thumidity\n"
    )


def _fmt_4dec_or_nan(x: Optional[float]) -> str:
    """Format a float with 4 decimals, or 'NaN' if None."""
    if x is None:
        return "NaN"
    return f"{x:.4f}"


def _fmt_default_or_nan(x: Optional[float | str]) -> str:
    """Format a scalar with str() or 'NaN' if None."""
    if x is None:
        return "NaN"
    return str(x)


def _row_to_line(m: Measurement) -> str:
    """Serialize a measurement into a single line."""
    fields = [
        _fmt_default_or_nan(m.map_id),
        _fmt_default_or_nan(m.timestamp_iso),
        _fmt_4dec_or_nan(m.azimuth_deg),
        _fmt_4dec_or_nan(m.elevation_deg),
        _fmt_4dec_or_nan(m.offset_az_deg),
        _fmt_4dec_or_nan(m.offset_el_deg),
        _fmt_default_or_nan(m.temperature_c),
        _fmt_default_or_nan(m.pressure_hpa),
        _fmt_default_or_nan(m.humidity_frac),
    ]
    return "\t".join(fields) + "\n"


def _read_header_tokens(path: Path) -> List[str]:
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.lstrip("\ufeff")

            if not line.strip():
                continue

            if line.lstrip().startswith("#"):
                continue

            return line.rstrip("\n").split("\t")

    raise SchemaMismatchError("No header line found in existing file")


def write_offsets_tsv(
    path: str | Path,
    metadata: Metadata,
    rows: Iterable[Measurement],
    append: bool = True,
) -> None:
    """
    Write (or append) a TSV file of pointing offsets with metadata and header.

    Parameters
    ----------
    path : str or Path
        Output path.
    metadata : Metadata
        File-level metadata (written only when creating/overwriting the file).
    rows : Iterable[Measurement]
        Sequence of measurements to write.
    append : bool, default True
        If True and the file exists, validate schema and append rows.
        If False, overwrite the file atomically (write to temp, then replace).

    Raises
    ------
    SchemaMismatchError
        If in append mode the existing header does not match the expected one.
    """
    p = Path(path)

    if append and p.exists():
        try:
            on_disk = _read_header_tokens(p)
        except SchemaMismatchError:
            # File exists but has no valid header (e.g. empty or only comments)
            append = False
        else:
            expected = _expected_columns()
            if on_disk != expected:
                raise SchemaMismatchError(
                    f"Header mismatch when appending. On disk: {on_disk} ; expected: {expected}"
                )
            with p.open("a", encoding="utf-8") as f:
                for m in rows:
                    f.write(_row_to_line(m))
            return

    # Overwrite (or create new) atomically
    tmp = p.with_suffix(p.suffix + ".tmp")
    try:
        with tmp.open("w", encoding="utf-8", newline="\n") as f:
            _write_metadata_block(f, metadata)
            _write_column_header(f)
            for m in rows:
                f.write(_row_to_line(m))
        tmp.replace(p)
    finally:
        if tmp.exists() and p.exists():
            try:
                tmp.unlink()
            except OSError:
                pass


if __name__ == "__main__":
    # Minimal smoke test / example writer. Adjust values as needed.
    md = Metadata(
        location="Dome C, Antarctica",
        antenna_diameter_m=1.2,
        frequency_ghz=100,
        software_version="1.4.0",
    )
    demo = [
        Measurement(
            map_id="250101T185415bOASI",
            timestamp_iso="2025-01-03T00:02:35Z",
            azimuth_deg=14.6622,
            elevation_deg=37.7847,
            offset_az_deg=-0.0829,
            offset_el_deg=0.0700,
            temperature_c=None,
            pressure_hpa=None,
            humidity_frac=None,
        )
    ]
    write_offsets_tsv("example.tsv", md, demo, append=False)
