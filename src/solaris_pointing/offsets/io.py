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
  - Software metadata now includes a URL pointing to the repository tree at
    the current short commit and may include the full commit SHA.

Metadata block
--------------
The commented metadata block includes the full set of context information
passed through the `Metadata` dataclass. This covers:

  - Site information (location string, site code, site data code,
    latitude/longitude, height).
  - Telescope parameters (antenna diameter, observing frequency).
  - Bias parameters for azimuth and elevation.
  - Refraction mode and, when applicable, relevant atmospheric parameters.
  - Algorithm name used to generate the offsets.
  - Software information (URL to the repository tree at the short commit
    and, optionally, the full commit SHA).
  - Creation timestamp for the metadata block.

For the exhaustive list of fields and their units, refer to the `Metadata`
dataclass definition in this module.

Important on separators
-----------------------
All other field separators are tabs. Concretely, the header is:

  map_id\ttimestamp\tazimuth\televation\toffset_az\toffset_el\t
  temperature\tpressure\thumidity

Units and conventions
---------------------
- azimuth_deg, elevation_deg: observed angles [deg]
- offset_az_deg, offset_el_deg: (observed − solar) angles [deg]
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
...     location="Mario Zucchelli Station, Antarctica",
...     antenna_diameter_m=1.2,
...     frequency_ghz=100,
...     # and several optional arguments
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
    antenna_diameter_m : float
        Antenna diameter in meters.
    frequency_ghz : float
        Observing frequency in GHz.

    site_location : Optional[str]
        General site location (e.g. "Antarctica").
    site_code : Optional[str]
        Short site identifier code (e.g. "MZS").
    data_code: Optional[str]
        Code extrapolated from data (e.g. OASI).
    site_lat : Optional[float]
        Site latitude in degrees.
    site_lon : Optional[float]
        Site longitude in degrees.
    site_height : Optional[float]
        Site height in meters.
    az_offset_bias : Optional[float]
        Applied azimuth offset bias in degrees.
    el_offset_bias : Optional[float]
        Applied elevation offset bias in degrees.
    refraction : Optional[str]
        "enabled" or "disabled".
    algo : Optional[str]
        Name of the algorithm used to generate the offsets.
    software_url : Optional[str]
        URL pointing to the exact commit used.
    software_commit: Optional[str]
        Commit of the software.
    created_at_iso : Optional[str]
        ISO-8601 UTC timestamp string for file creation. If None, current UTC is used.
    """

    config_file: Optional[str] = None
    algo: Optional[str] = None
    # Telescope
    antenna_diameter_m: Optional[float] = None
    frequency_ghz: Optional[float] = None
    az_offset_bias: Optional[float] = None
    el_offset_bias: Optional[float] = None
    # Site information
    site_location: Optional[str] = None
    site_code: Optional[str] = None
    data_code: Optional[str] = None
    site_lat: Optional[float] = None
    site_lon: Optional[float] = None
    site_height: Optional[float] = None
    # Refraction parameters
    refraction: Optional[str] = None
    pressure_hpa: Optional[float] = None
    temperature_c: Optional[float] = None
    humidity_frac: Optional[float] = None
    obswl_mm: Optional[float] = None
    # Algo parameters
    peak_frac: Optional[float] = None
    central_power_frac: Optional[float] = None
    # Additional information
    software_url: Optional[str] = None
    software_commit: Optional[str] = None
    created_at_iso: Optional[str] = None

    def created_iso_or_now(self) -> str:
        """Return created_at_iso if provided, else now in UTC as ISO-8601."""
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
    """Write the commented metadata block."""
    metadata_title = "# === Metadata " + 70 * "=" + "\n"
    f.write(metadata_title)

    # ---------------------------
    # Site information
    # ---------------------------
    f.write("# [Site]\n")
    if md.site_location:
        f.write(f"#  Location: {md.site_location}\n")
    if md.site_code:
        f.write(f"#  Code: {md.site_code}\n")
    if md.data_code:
        f.write(f"#  Code from data: {md.data_code}\n")
    if md.site_lat is not None:
        f.write(f"#  Latitude (deg): {md.site_lat}\n")
    if md.site_lon is not None:
        f.write(f"#  Longitude (deg): {md.site_lon}\n")
    if md.site_height is not None:
        f.write(f"#  Height (m): {md.site_height}\n")
    f.write("#\n")

    # ---------------------------
    # Telescope parameters
    # ---------------------------
    f.write("# [Telescope]\n")
    if md.antenna_diameter_m is not None:
        f.write(f"#   Diameter (m)    : {md.antenna_diameter_m}\n")
    if md.frequency_ghz is not None:
        f.write(f"#   Frequency (GHz) : {md.frequency_ghz}\n")
    f.write("#\n")

    # ---------------------------
    # Algorithm parameters
    # ---------------------------
    f.write("# [Algorithm]\n")
    if md.algo:
        f.write(f"#  Name: {md.algo}\n")
    if md.az_offset_bias is not None:
        f.write(f"#  AZ offset bias (deg): {md.az_offset_bias}\n")
    if md.el_offset_bias is not None:
        f.write(f"#  EL offset bias (deg): {md.el_offset_bias}\n")
    if md.refraction is not None:
        f.write(f"#  Refraction: {md.refraction}\n")
    if md.pressure_hpa is not None:
        f.write(f"#  Pressure (hPa): {md.pressure_hpa}\n")
    if md.temperature_c is not None:
        f.write(f"#  Temperature (C): {md.temperature_c}\n")
    if md.humidity_frac is not None:
        f.write(f"#  Humidity (frac): {md.humidity_frac}\n")
    if md.obswl_mm is not None:
        f.write(f"#  Wavelength (mm): {md.obswl_mm}\n")
    if md.peak_frac is not None:
        f.write(f"#  Peak fraction: {md.peak_frac}\n")
    if md.central_power_frac is not None:
        f.write(f"#  Central power frac: {md.central_power_frac}\n")
    f.write("#\n")

    # ---------------------------
    # Software information
    # ---------------------------
    f.write("# [Software]\n")
    if md.software_url:
        f.write(f"#  Repository: {md.software_url}\n")
    if md.software_commit:
        f.write(f"#  Commit SHA: {md.software_commit}\n")
    f.write("#\n")

    # ---------------------------
    # Run information
    # ---------------------------
    f.write("# [Run]\n")
    if md.config_file:
        f.write(f"#  Config file: {md.config_file}\n")
    if md.created_at_iso:
        f.write(f"#  Created at (UTC): {md.created_at_iso}\n")
    f.write("# " + (len(metadata_title) - 2) * "=" + "\n")


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
                    f"Header mismatch when appending. On disk: {on_disk} ; "
                    f"expected: {expected}"
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
        site_location="Mario Zucchelli Station, Antarctica",
        antenna_diameter_m=1.2,
        frequency_ghz=100,
        software_url="https://...",
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
