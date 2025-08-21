"""
offset_io.tsv
=============

Utilities for writing telescope pointing-offset measurements to a
tab-separated values (TSV) file with a standardized schema.

What this module provides
-------------------------
- `Metadata`: a dataclass describing file-level metadata (location, antenna
  diameter, observing frequency, software version, creation timestamp).
- `Measurement`: a dataclass describing one measurement (timestamp, observed
  azimuth/elevation, offsets in **degrees**, and optional environment fields).
- `write_offsets_tsv(path, metadata, rows, append=True)`: write or append to
  a TSV file using a fixed column schema and an English commented header.
- `SchemaMismatchError`: a dedicated exception raised when appending to a file
  whose existing column header does not match the expected schema.

File format
-----------
The generated TSV file has three parts:

1) A commented metadata block (lines starting with '#'), in English:
   - Key metadata (location, antenna diameter, frequency)
   - Column dictionary with units
   - Provenance (software version, creation time)

2) A single header line with fixed column names:
   timestamp, azimuth, elevation, offset_az, offset_el, temperature, pressure, humidity

3) Data rows (one per measurement). Values are separated by tab ('\\t').
   - Offsets are **in degrees** (not arcseconds).
   - `azimuth`, `elevation`, `offset_az`, `offset_el` are serialized with
     **exactly four decimal places** (e.g., `123.4000`).
   - Missing values (`None`) are written as the string "NaN".

Robustness and guarantees
-------------------------
- When appending (`append=True`) to an existing file, the function validates
  the on-disk column header. If it differs from the expected schema, a
  `SchemaMismatchError` is raised with a clear message (path, expected, found).
- The header parser is robust to blank lines between the metadata block and
  the header (it skips comment lines and blank/whitespace-only lines).
- When overwriting (`append=False`), writing is atomic on POSIX: the file is
  written to `path + ".tmp"` and then replaced via `os.replace`.

Example
-------
>>> from offset_io.tsv import Metadata, Measurement, write_offsets_tsv
>>>
>>> md = Metadata(
...     location="MZS, Antarctica",
...     antenna_diameter_m=2.0,
...     frequency_hz=100e9,
...     software_version="2025.08.05",
... )
>>>
>>> rows = [
...     Measurement(
...         timestamp_iso="2025-08-01T10:00:00Z",
...         azimuth_deg=123.456,
...         elevation_deg=45.789,
...         offset_az_deg=0.0034,   # degrees
...         offset_el_deg=-0.0023,  # degrees
...         temperature_c=None,     # will be written as "NaN"
...         pressure_hpa=None,      # will be written as "NaN"
...         humidity_frac=None,     # will be written as "NaN"
...     )
... ]
>>>
>>> # First run will create the file with header and metadata:
>>> write_offsets_tsv("output_example.tsv", md, rows, append=True)
>>>
>>> # Second run with the same schema will append rows without errors:
>>> write_offsets_tsv("output_example.tsv", md, rows, append=True)

Design notes
------------
- Offsets are consistently expressed in **degrees** (inputs and file schema).
- `azimuth`, `elevation`, `offset_az`, `offset_el` are formatted with four
  decimal places for consistent downstream parsing.
- Optional fields (temperature, pressure, humidity) default to `None` and are
  serialized as the string "NaN".
- The column schema is fixed and validated on append to avoid mixing formats.
- A future extension could provide a streaming writer class if long-running,
  high-throughput logging is needed.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Iterable, Optional, TextIO
import math
import os

__all__ = [
    "Metadata",
    "Measurement",
    "SchemaMismatchError",
    "write_offsets_tsv",
]


# =============================================================================
# Exceptions
# =============================================================================


class SchemaMismatchError(ValueError):
    """
    Raised when appending to an existing file whose column header line does not
    match the expected TSV schema.

    This typically indicates the target file was produced by a different
    version of the schema or by a different tool.

    The exception message includes:
      - The file path
      - The expected header (tab-separated)
      - The found header (tab-separated)
    """

    pass


# =============================================================================
# Data models
# =============================================================================


@dataclass(frozen=True)
class Metadata:
    """
    Container for file-level metadata written as commented header lines.

    Attributes
    ----------
    location : str
        Free-form location string (e.g., "MZS, Antarctica").
    antenna_diameter_m : float
        Antenna diameter in meters. Must be > 0.
    frequency_hz : float
        Observing frequency in hertz. Must be > 0.
    software_version : str
        Software version string used to generate the file.
    created_at_iso : Optional[str], default None
        ISO 8601 instant for file creation time. If None, current UTC time
        is used. Example: "2025-08-05T11:00:00Z".
    """

    location: str
    antenna_diameter_m: float
    frequency_hz: float
    software_version: str
    created_at_iso: Optional[str] = None

    def __post_init__(self) -> None:
        # Minimal validation to catch obvious mistakes early.
        if self.antenna_diameter_m <= 0:
            raise ValueError("antenna_diameter_m must be > 0")
        if self.frequency_hz <= 0:
            raise ValueError("frequency_hz must be > 0")


@dataclass(frozen=True)
class Measurement:
    """
    One pointing-offset measurement row.

    Notes
    -----
    - All angles are in **degrees**.
    - Offsets are in **degrees** everywhere (both as inputs and in the file).

    Attributes
    ----------
    timestamp_iso : str
        ISO 8601 timestamp of the observation (e.g., "2025-08-01T10:00:00Z").
        Provide an explicit timezone ("Z" or ±HH:MM) for clarity.
    azimuth_deg : float
        Observed azimuth [deg].
    elevation_deg : float
        Observed elevation [deg].
    offset_az_deg : float
        Solar azimuth - observed azimuth [deg].
    offset_el_deg : float
        Solar elevation - observed elevation [deg].
    temperature_c : Optional[float], default None
        Ambient temperature [°C]. If None, written as "NaN".
    pressure_hpa : Optional[float], default None
        Ambient pressure [hPa]. If None, written as "NaN".
    humidity_frac : Optional[float], default None
        Relative humidity as a fraction in [0, 1]. If None, written as "NaN".
    """

    timestamp_iso: str
    azimuth_deg: float
    elevation_deg: float
    offset_az_deg: float
    offset_el_deg: float
    temperature_c: Optional[float] = None
    pressure_hpa: Optional[float] = None
    humidity_frac: Optional[float] = None

    def __post_init__(self) -> None:
        # Light but informative validation; keep permissive to avoid blocking
        # real-world data flows with slightly out-of-range values.
        if not (-360.0 <= self.azimuth_deg <= 720.0):
            raise ValueError("azimuth_deg out of a reasonable range [-360, 720]")
        if not (-90.0 <= self.elevation_deg <= 120.0):
            # Allow some margin around [0, 90] if raw readings overshoot slightly.
            raise ValueError("elevation_deg out of a reasonable range [-90, 120]")

        # Offsets must be finite floats (they can be outside 'small' ranges).
        for name, val in (
            ("offset_az_deg", self.offset_az_deg),
            ("offset_el_deg", self.offset_el_deg),
        ):
            if not math.isfinite(val):
                raise ValueError(f"{name} must be a finite float (degrees)")

        # Humidity, if present, must be in [0, 1].
        if self.humidity_frac is not None:
            if not (0.0 <= self.humidity_frac <= 1.0):
                raise ValueError("humidity_frac must be in [0, 1] when provided")


# =============================================================================
# Public API
# =============================================================================


def write_offsets_tsv(
    path: str,
    metadata: Metadata,
    rows: Iterable[Measurement],
    append: bool = True,
) -> None:
    """
    Write (or append) a TSV file with a commented metadata block and a fixed
    column header.

    Behavior
    --------
    - If the file does not exist, this writes:
        (1) a commented metadata/header block (English),
        (2) one column header line,
        (3) one line per `Measurement`.
    - If the file exists and `append=True`, this validates the on-disk header
      against the expected schema and then appends new rows.
    - If `append=False`, the file is overwritten atomically (write to `.tmp`,
      then `os.replace`).

    Parameters
    ----------
    path : str
        Output file path (e.g., "output_example.tsv").
    metadata : Metadata
        File-level metadata written in the commented header.
    rows : Iterable[Measurement]
        Sequence of measurement rows to write.
    append : bool, default True
        If True and the file exists, append to it (after header validation).
        If False, create/overwrite the file atomically.

    Raises
    ------
    SchemaMismatchError
        When appending to an existing file whose header does not match the
        expected TSV schema.
    ValueError
        If the file is present but appears to contain no header line.
    """
    creating_new = not os.path.exists(path)

    if not append:
        # Overwrite: write to a temporary file and atomically replace the target.
        tmp_path = f"{path}.tmp"
        with open(tmp_path, "w", newline="") as f:
            _write_metadata_block(f, metadata)
            _write_column_header(f)
            for m in rows:
                f.write(_row_to_tsv(m))
        os.replace(tmp_path, path)
        return

    # Append mode (or first creation).
    mode = "a" if not creating_new else "w"
    with open(path, mode, newline="") as f:
        if creating_new:
            _write_metadata_block(f, metadata)
            _write_column_header(f)
        else:
            _check_header_or_raise(path)
        for m in rows:
            f.write(_row_to_tsv(m))


# =============================================================================
# Internal helpers
# =============================================================================


def _write_metadata_block(f: TextIO, md: Metadata) -> None:
    """
    Write the commented metadata block (English-only) and standard field descriptions.

    This includes:
      - Location, antenna diameter, frequency
      - Column dictionary (name + unit)
      - Provenance (software version, creation timestamp)
    """
    created = md.created_at_iso or datetime.now(timezone.utc).strftime(
        "%Y-%m-%dT%H:%M:%SZ"
    )

    # Top-level metadata
    f.write(f"# Location: {md.location}\n")
    f.write(f"# Antenna diameter: {md.antenna_diameter_m} m\n")
    f.write(f"# Frequency: {md.frequency_hz} Hz\n")

    # Column dictionary (all English, with units)
    f.write("# timestamp: ISO 8601\n")
    f.write("# azimuth: observed azimuth, deg\n")
    f.write("# elevation: observed elevation, deg\n")
    f.write("# offset_az: solar azimuth - observed azimuth, deg\n")
    f.write("# offset_el: solar elevation - observed elevation, deg\n")
    f.write("# temperature: °C\n")
    f.write("# pressure: hPa\n")
    f.write("# humidity: relative humidity, fraction (0..1)\n")

    # Provenance
    f.write(f"# Generated with software version: {md.software_version}\n")
    f.write(f"# Created at: {created}\n")
    f.write("\n")  # trailing blank line for readability


def _write_column_header(f: TextIO) -> None:
    """
    Write the canonical TSV column header line (tab-separated).
    """
    cols = _expected_columns()
    f.write("\t".join(cols) + "\n")


def _fmt_4dec_or_nan(x: Optional[float]) -> str:
    """
    Format angles with exactly four decimal places, or 'NaN' if None/non-finite.

    This is used for azimuth/elevation and offset fields.
    """
    if x is None:
        return "NaN"
    if not math.isfinite(x):
        return "NaN"
    return f"{x:.4f}"  # fixed four decimals


def _fmt_default_or_nan(x: Optional[float | str]) -> str:
    """
    Format general fields: return 'NaN' for None/non-finite floats, else str(x).
    This is used for temperature, pressure, humidity and timestamp.
    """
    if x is None:
        return "NaN"
    if isinstance(x, float) and not math.isfinite(x):
        return "NaN"
    return str(x)


def _row_to_tsv(m: Measurement) -> str:
    """
    Convert a Measurement to a TSV line.

    Formatting rules:
      - timestamp: raw string
      - azimuth/elevation/offsets: four decimal places
      - temperature/pressure/humidity: default string or "NaN"

    Returns
    -------
    str
        The tab-separated row ending with a newline.
    """
    fields = [
        _fmt_default_or_nan(m.timestamp_iso),
        _fmt_4dec_or_nan(m.azimuth_deg),
        _fmt_4dec_or_nan(m.elevation_deg),
        _fmt_4dec_or_nan(m.offset_az_deg),  # degrees
        _fmt_4dec_or_nan(m.offset_el_deg),  # degrees
        _fmt_default_or_nan(m.temperature_c),
        _fmt_default_or_nan(m.pressure_hpa),
        _fmt_default_or_nan(m.humidity_frac),
    ]
    return "\t".join(fields) + "\n"


def _expected_columns() -> list[str]:
    """
    The canonical list of column names for the TSV schema.

    Returns
    -------
    list[str]
        [
          "timestamp",
          "azimuth",
          "elevation",
          "offset_az",
          "offset_el",
          "temperature",
          "pressure",
          "humidity"
        ]
    """
    return [
        "timestamp",
        "azimuth",
        "elevation",
        "offset_az",
        "offset_el",
        "temperature",
        "pressure",
        "humidity",
    ]


def _check_header_or_raise(path: str) -> None:
    """
    Ensure the existing file at `path` has the expected column schema.

    This function:
      - Skips comment lines (starting with '#') and blank/whitespace-only lines.
      - Reads the first non-comment, non-blank line as the header.
      - Compares it (as a list of columns split on tab) to the expected schema.

    Parameters
    ----------
    path : str
        Existing TSV file path.

    Raises
    ------
    ValueError
        If the file appears to contain no column header at all.
    SchemaMismatchError
        If the column header does not match the expected schema.
    """
    expected_cols = _expected_columns()
    expected_header = "\t".join(expected_cols)

    header_line: Optional[str] = None
    with open(path, "r", newline="") as f:
        for raw in f:
            line = raw.strip()
            # Skip comment lines and blank/whitespace-only lines.
            if not line or line.startswith("#"):
                continue
            header_line = line
            break

    if header_line is None:
        # No header found after skipping comments/blank lines.
        raise ValueError(f"File '{path}' appears to contain no column header")

    found_cols = header_line.split("\t")
    if found_cols != expected_cols:
        # Raise a dedicated exception with actionable context.
        raise SchemaMismatchError(
            "Existing file schema does not match expected header.\n"
            f"Path:     {path}\n"
            f"Expected: {expected_header}\n"
            f"Found:    {header_line}\n"
            "Hint: If you intend to replace the file, "
            "call write_offsets_tsv(..., append=False)."
        )
