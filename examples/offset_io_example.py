"""
offset_io_example.py
====================

Purpose
-------
This script demonstrates how to use the `offset_io` library from the
`solaris-pointing` project to produce a standard TSV output file containing
telescope pointing offsets. It is meant as a minimal, end‑to‑end example that
you can adapt to your own offset‑computation pipeline.

Requirements
------------
- Install the `offset_io` package from the `solaris-pointing` repository.
  See that project's README.md for installation instructions.
- No prior knowledge of the internal codebase is required.

What this example does
----------------------
1. Builds a `Metadata` object with general information about the observation
   (e.g., site/location, antenna diameter, observing frequency, and software
   version). This metadata will be written once in the header of the TSV file.
2. Creates one or more `Measurement` rows, each representing the results for a
   single map/scan (timestamp, azimuth, elevation, and the computed offsets).
   Optional environmental fields (temperature, pressure, humidity) can be left
   as `None`; they will be written as "NaN" in the file.
3. Calls `write_offsets_tsv()` to write the output file:
   - On the **first** call, the file is created and the header (with metadata)
     is written, followed by any provided rows of data.
   - On **subsequent** calls using the same schema, only the data rows are
     appended; the header is not written again.

Output
------
- A file named ``output_offset_io_example.tsv`` is produced in the working directory.
- The file contains:
  - A header section populated from `Metadata` (written only once).
  - One data row per `Measurement` you provide (appended on each call).

Usage patterns
--------------
- Batch append: pass a list of `Measurement` objects to append multiple rows
  in one call.
- Per‑line append: you may also call `write_offsets_tsv()` once per line you
  want to add; since the header is no longer added after the first run,
  each call will simply append a new data row.

How to run
----------
If you want to run the script, navigate to the directory where it is located and run:

    python offset_io_example.py

Adjust the example metadata and measurements to match your setup (e.g., site
name, timestamps, offsets). The values shown here are placeholders to help you
get started quickly.
"""

from solaris_pointing.offset_io import Metadata, Measurement, write_offsets_tsv

# Metadata will be added to the header of the file
md = Metadata(
    location="MZS, Antarctica",
    antenna_diameter_m=2.0,
    frequency_hz=100e9,
    software_version="2025.08.05",
)

rows = []  # Rows of data, one for each map, that you want to append to the file.

# For every map of that location, append the corresponding raw data.
rows.append(
    Measurement(
        timestamp_iso="2025-08-01T10:00:00Z",
        azimuth_deg=123.456,
        elevation_deg=45.789,
        offset_az_deg=0.0034,
        offset_el_deg=-0.0023,
        temperature_c=None,  # will be written as "NaN"
        pressure_hpa=None,  # will be written as "NaN"
        humidity_frac=None,  # will be written as "NaN"
    )
)

# First run will create a file with both metadata and, if present, also data:
write_offsets_tsv("output_offset_io_example.tsv", md, rows, append=True)

# Second run with the same schema will append only rows of data (no metadata):
write_offsets_tsv("output_offset_io_example.tsv", md, rows, append=True)

# That's all. Alternatively, you can make one call for each line you append, since the
# header is no longer added. In other words, you create one row for each line of data:
#
# row = [
#    Measurement(
#        timestamp_iso="2025-08-01T10:00:00Z",
#        azimuth_deg=123.456,
#        elevation_deg=45.789,
#        offset_az_deg=0.0034,
#        offset_el_deg=-0.0023,
#        temperature_c=None,     # will be written as "NaN"
#        pressure_hpa=None,      # will be written as "NaN"
#        humidity_frac=None,     # will be written as "NaN"
#    )
# ]
# write_offsets_tsv("output_offset_io_example.tsv", md, row, append=True)
