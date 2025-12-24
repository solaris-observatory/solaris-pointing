# Solaris Pointing

[![CI](https://github.com/solaris-observatory/solaris-pointing/actions/workflows/ci.yml/badge.svg)](https://github.com/solaris-observatory/solaris-pointing/actions/workflows/ci.yml)
[![Coverage Status](https://coveralls.io/repos/github/solaris-observatory/solaris-pointing/badge.svg?branch=main)](https://coveralls.io/github/solaris-observatory/solaris-pointing?branch=main)
[![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

---

## Overview

**Solaris Pointing** is a comprehensive Python toolkit for telescope pointing analysis and correction. It provides command-line tools and a Python API to:

- **Compute solar pointing offsets** from Sun scan maps (`.path` + `.sky` pairs)
- **Fit azimuth/elevation pointing models** (polynomial + Fourier harmonics)
- **Generate production-ready correction models** for telescope operations
- **Use pointing models in telescope control software** (copy/paste summary functions or load `.joblib` bundles)
- **Support multiple algorithms** for offset computation with pluggable architecture

The package is designed for operational use in solar radio astronomy, particularly for 100 GHz observations requiring precise pointing calibration.

---

## Table of Contents

- [Features](#features)
- [Requirements](#requirements)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Command-Line Tools](#command-line-tools)
  - [generate_offsets.py](#1-generate_offsetspy)
  - [generate_model.py](#2-generate_modelpy)
- [Configuration Profiles](#configuration-profiles)
- [Input Data Format](#input-data-format)
- [Output Files](#output-files)
- [Python API Usage](#python-api-usage)
- [Testing](#testing)
- [Project Structure](#project-structure)
- [Contributing](#contributing)
- [Troubleshooting](#troubleshooting)
- [License](#license)
---

## Features

### Offset Computation (`generate_offsets.py`)

- ✅ **Recursive scan discovery** driven by `.sky`, pairing `.path` by filename prefix (`<base>*.path`) with `<base>_offset.path` preferred when two candidates exist
- ✅ **Date-based filtering** via stem timestamps (`YYMMDDTHHMMSS`)
- ✅ **Atmospheric refraction correction** (optional, via `pysolar`)
- ✅ **Multi-signal support** for `.sky` files with multiple detector channels
- ✅ **Site and telescope metadata** integration
- ✅ **Configurable thresholds** for peak/power selection
- ✅ **Fixed bias application** for systematic offset correction
- ✅ **Configuration profiles** via TOML files

### Model Fitting (`generate_model.py`)

- ✅ **Polynomial models** with adjustable degree
- ✅ **Fourier harmonics** with custom periods
- ✅ **MAD-based outlier rejection** with configurable z-score threshold
- ✅ **Ridge regularization** (L2) for stable fits
- ✅ **Per-axis or unified fits** (AZ, EL, or both)
- ✅ **Model prediction** at arbitrary azimuths
- ✅ **Model merging** for unified bundles
- ✅ **Auto-generated Python functions** for each fitted model
- ✅ **Comprehensive plots and summaries** with multiple unit support

### Data Pipeline Support

- ✅ **TSV format** for portable, human-readable offset tables
- ✅ **Joblib serialization** for efficient model storage
- ✅ **Metadata tracking** via JSON sidecars
- ✅ **Incremental processing** with progress indicators
- ✅ **Reproducible workflows** via configuration management

---

## Requirements

### Python Version

- **Python 3.12 or higher** (tested on 3.12 and 3.13)

---

## Installation

### Method 1: Poetry (Recommended)

```bash
# Clone the repository
git clone https://github.com/solaris-observatory/solaris-pointing.git
cd solaris-pointing

# Install with Poetry
poetry install

# Activate the virtual environment
poetry shell
```

### Method 2: pip (Runtime Only)

```bash
# Clone the repository
git clone https://github.com/solaris-observatory/solaris-pointing.git
cd solaris-pointing

# Install runtime dependencies
pip install -r requirements.txt

# Install the package in development mode
pip install -e .
```

### Method 3: pip (With Development Tools)

```bash
# Install all dependencies including testing/docs
pip install -r requirements-dev.txt
pip install -e .
```

### Verify Installation

```bash
# Test the CLI tools
python scripts/generate_offsets.py --examples
python scripts/generate_model.py --examples

# Run the test suite
pytest
```

---

## Quick Start

### 1. Compute Offsets from Sun Scans

```bash
python scripts/generate_offsets.py --data scans/ --algo sun_maps
```

**Output:** `offsets/sun_maps.tsv`

Each row contains: `azimuth`, `offset_az`, `offset_el`, plus algorithm-specific metadata.

Important: `generate_offsets.py` appends results to the output TSV. If you run it twice with
the same `--outdir` and `--algo`, the second run will add new rows to the existing file.
Delete/rename the TSV (or use a different `--outdir`) if you want a clean file.


### 2. Fit Pointing Models

```bash
python scripts/generate_model.py fit offsets/sun_maps.tsv
```

**Output:**
```
models/
├── sun_maps_az.joblib         # Azimuth model
├── sun_maps_el.joblib         # Elevation model
├── sun_maps_summary_az.txt    # AZ fit summary + Python code
├── sun_maps_summary_el.txt    # EL fit summary + Python code
├── sun_maps_summary.txt       # General summary
├── sun_maps_az.png            # AZ residual plot
├── sun_maps_el.png            # EL residual plot
└── sun_maps.joblib            # Unified bundle
```

### 3. Use the Model in Telescope Code

After you fit a model, you can integrate it in the pointing software in two ways:

1. **Copy/paste the generated Python functions** (`az_offset(...)` / `el_offset(...)`) from the
   per-axis summary files.
2. **Load the serialized `.joblib` bundle** and call the model API at runtime.

See: [Using the Pointing Model in Telescope Code](#using-the-pointing-model-in-telescope-code)

---

### 4. Predict Offsets

```bash
python scripts/generate_model.py predict sun_maps --azimuth 45.0 --unit arcmin
```

**Output:** Predicted azimuth and elevation offsets at 45° azimuth.

---

## Command-Line Tools

### 1. `generate_offsets.py`

**Purpose:** Discover Sun scan pairs, compute pointing offsets, and write TSV results.

#### Basic Usage

```bash
python scripts/generate_offsets.py --data <scan_directory> --algo <algorithm_name>
```

#### Key Options

| Option | Type | Description |
|--------|------|-------------|
| `--algo` | str | Algorithm name (e.g., `sun_maps`) |
| `--data` | path | Root directory for scan discovery |
| `--outdir` | path | Output directory (default: `./offsets`) |
| `--date-start` | YYYY-MM-DD | Filter: inclusive start date |
| `--date-end` | YYYY-MM-DD | Filter: inclusive end date |
| `--signal` | int | Select signal column (0, 1, etc.) |
| `--config` | str | Load TOML profile from `config/<name>.toml` |
| `--examples` | flag | Show usage examples and exit |

#### Site Parameters

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `--site-lat` | float | -74.6950 | Latitude (degrees) |
| `--site-lon` | float | 164.1000 | Longitude (degrees) |
| `--site-height` | float | 30.0 | Altitude (meters) |
| `--site-location` | str | "Unknown" | Location name |
| `--site-code` | str | "Unknown" | Short site code |

#### Telescope Parameters

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `--frequency` | float | 100.0 | Frequency (GHz) |
| `--diameter` | float | 2.6 | Diameter (meters) |

#### Refraction Parameters

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `--enable-refraction` | flag | False | Enable atmospheric correction |
| `--pressure` | float | 1013.25 | Pressure (hPa) |
| `--temperature` | float | 15.0 | Temperature (°C) |
| `--humidity` | float | 0.5 | Relative humidity (0-1) |
| `--obswl` | float | 3.0 | Wavelength (mm) |

#### Offset Biases

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `--az-offset-bias` | float | 0.0 | AZ bias (degrees) |
| `--el-offset-bias` | float | 0.0 | EL bias (degrees) |

#### Examples

**Minimal run:**
```bash
python scripts/generate_offsets.py --data scans/ --algo sun_maps
```

**With date filters:**
```bash
python scripts/generate_offsets.py --data scans/ --algo sun_maps \
    --date-start 2025-01-01 --date-end 2025-01-31
```

**With refraction and site parameters:**
```bash
python scripts/generate_offsets.py --data scans/ --algo sun_maps \
    --site-location "Antarctica" --site-code MZS \
    --site-lat -74.6950 --site-lon 164.1000 --site-height 30 \
    --enable-refraction --pressure 690 --temperature -7 --humidity 0.5
```

**Using a configuration profile:**
```bash
python scripts/generate_offsets.py --algo sun_maps --config mzs --data scans/
```

**Multi-signal selection:**
```bash
python scripts/generate_offsets.py --data scans/ --algo sun_maps --signal 1
```

---

### 2. `generate_model.py`

**Purpose:** Fit, predict, and merge pointing-offset models.

#### Subcommands

```bash
python scripts/generate_model.py {fit|predict|merge} [options]
```

#### Subcommand: `fit`

Fit azimuth/elevation models from TSV offset data.

**Basic usage:**
```bash
python scripts/generate_model.py fit <tsv_file> [options]
```

**Key options:**

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `--degree` | int | 3 | Polynomial degree |
| `--zscore` | float | 2.5 | MAD outlier threshold |
| `--ridge-alpha` | float | 0.0 | L2 regularization |
| `--fourier-k` | int | 0 | Number of harmonics |
| `--periods-deg` | str | - | Custom periods (comma-sep) |
| `--az` | flag | - | Fit only AZ axis |
| `--el` | flag | - | Fit only EL axis |
| `--input-offset-unit` | str | deg | Input unit (deg/arcmin/arcsec) |
| `--plot-unit` | str | deg | Plot unit (deg/arcmin/arcsec) |

**Examples:**

```bash
# Minimal fit (both axes)
python scripts/generate_model.py fit offsets/sun_maps.tsv

# Custom polynomial + Fourier
python scripts/generate_model.py fit offsets.tsv --degree 3 --fourier-k 2

# Fit only azimuth
python scripts/generate_model.py fit offsets.tsv --az --degree 3

# Multiple input files
python scripts/generate_model.py fit a.tsv b.tsv --degree 2

# Arcsecond input data
python scripts/generate_model.py fit offsets.tsv --input-offset-unit arcsec
```

#### Subcommand: `predict`

Predict offsets at a given azimuth using saved models.

**Basic usage:**
```bash
python scripts/generate_model.py predict <model_stem> --azimuth <value> [options]
```

**Key options:**

| Option | Type | Description |
|--------|------|-------------|
| `--azimuth` | float | Azimuth for prediction (degrees) |
| `--unit` | str | Output unit (deg/arcmin/arcsec) |
| `--az` | flag | Predict only AZ |
| `--el` | flag | Predict only EL |
| `--allow-extrapolation` | flag | Allow prediction outside data range |

**Examples:**

```bash
# Predict both axes
python scripts/generate_model.py predict sun_maps --azimuth 45.0 --unit arcmin

# Predict only azimuth
python scripts/generate_model.py predict sun_maps --az --azimuth 45.0

# Allow extrapolation
python scripts/generate_model.py predict sun_maps --azimuth 355.0 --allow-extrapolation
```

#### Subcommand: `merge`

Merge per-axis models into a unified bundle.

**Basic usage:**
```bash
python scripts/generate_model.py merge <model_stem>
```

**Example:**
```bash
python scripts/generate_model.py merge sun_maps
# Reads: sun_maps_az.joblib, sun_maps_el.joblib
# Writes: sun_maps.joblib
```

---

## Using the Pointing Model in Telescope Code

This library covers the full lifecycle:

1. **Identify offsets** from Sun scan pairs (`.path` + `.sky`) using `generate_offsets.py`.
2. **Create a pointing model** from an offsets TSV using `generate_model.py fit`.
3. **Use the model at runtime** to turn apparent coordinates into commanded coordinates.

Below are two supported runtime integration options.

### Option 1: Copy/Paste Functions from Summary Files

When you run `generate_model.py fit`, the output directory contains per-axis summary files
(e.g. `*_summary_az.txt` and `*_summary_el.txt`). Each summary includes a small, standalone
Python function that computes the offset for that axis as a function of **apparent azimuth**.

Typical function names in the summary files are:

- `az_offset(az)`  → returns AZ offset in degrees
- `el_offset(az)`  → returns EL offset in degrees

**How to use (minimal dependency approach):**

1. Open the summary files produced by the fit.
2. Copy the entire `def az_offset(...):` and `def el_offset(...):` blocks into your pointing code.
3. Apply the returned offsets to the apparent coordinates.

Example integration:

```python
import math

# Paste the two functions here:
# - az_offset(az)
# - el_offset(az)

def apply_pointing_model(az_app_deg: float, el_app_deg: float) -> tuple[float, float]:
    """Return commanded (az, el) by applying fitted offsets to apparent coordinates."""
    off_az_deg = float(az_offset(az_app_deg))
    off_el_deg = float(el_offset(az_app_deg))  # 1D model: EL offset depends on azimuth only

    az_cmd_deg = (az_app_deg + off_az_deg) % 360.0
    el_cmd_deg = el_app_deg + off_el_deg
    return az_cmd_deg, el_cmd_deg
```

**Notes:**
- The functions usually assume input azimuth in degrees in `[0, 360]` and embed any
  linearization logic (e.g. a `CUT`) used during the fit.
- Treat the pasted functions as generated artifacts: keep them versioned and tied to the
  exact dataset/model you deployed.

### Option 2: Load the `.joblib` Model Bundle

Instead of copying code, you can load the serialized model bundle (`.joblib`) produced by the
fit/merge workflow and predict offsets at runtime.

The default backend is `model_1d` (a 1D azimuth-only predictor that outputs both AZ and EL
offsets). The bundle contains the fitted coefficients and metadata (range checks, cut, etc.).

Example integration:

```python
from solaris_pointing.models import model_1d

class PointingOffsetModel:
    def __init__(self, model_path: str, allow_extrapolation: bool = False) -> None:
        # Load once at startup.
        self.bundle = model_1d.load_model(model_path)
        self.allow_extrapolation = allow_extrapolation

    def apply(self, az_app_deg: float, el_app_deg: float) -> tuple[float, float]:
        # Predict offsets in degrees (1D model: depends on azimuth only).
        off_az_deg, off_el_deg = model_1d.predict_offsets_deg(
            self.bundle,
            az_deg=az_app_deg,
            allow_extrapolation=self.allow_extrapolation,
        )

        az_cmd_deg = (az_app_deg + off_az_deg) % 360.0
        el_cmd_deg = el_app_deg + off_el_deg
        return az_cmd_deg, el_cmd_deg
```

**Notes:**
- By default, predictions outside the observed azimuth range raise an error. If you
  understand the risk, you can enable extrapolation (CLI: `--allow-extrapolation`).
- The `.meta.json` sidecar written by the CLI is useful for deployment tracking.

---

## Configuration Profiles

Configuration profiles allow you to store commonly-used parameters in TOML files, avoiding repetitive command-line arguments.

### Creating a Profile

Create a file `config/<profile_name>.toml`:

```toml
# config/mzs.toml

# ---------------------------
# Site information
# ---------------------------
site-location = "Antarctica"
site-code = "MZS"
site-lat = -74.6950
site-lon = 164.1000
site-height = 64

# ---------------------------
# Telescope parameters
# ---------------------------
frequency = 100      # GHz
diameter = 2.0       # meters

# ---------------------------
# Pointing offset biases
# ---------------------------
az-offset-bias = 0.0
el-offset-bias = 0.0

# ---------------------------
# Atmospheric model
# ---------------------------
enable-refraction = true
pressure = 990
temperature = -5
humidity = 0.5
obswl = 3.0

# ---------------------------
# Selection thresholds
# ---------------------------
peak-frac = 0.75
central-power-frac = 0.60

```

### Using a Profile

```bash
python scripts/generate_offsets.py --algo sun_maps --config mzs --data scans/
```

**Note:** CLI arguments override profile values.

---

## Input Data Format

### Scan Files

Solaris Pointing expects paired files for each observation:

1. **`.path` file** - Telescope pointing data (azimuth/elevation vs. time)
2. **`.sky` file** - Power measurements from detector(s)

Pairing is driven by the `.sky` file:

- For each `<base>.sky` found under `--data` (searched recursively), the driver searches for
  `.path` files whose *filename stem* starts with `<base>` (any subdirectory).
- If exactly **1** candidate `.path` is found, it is used.
- If exactly **2** candidates are found, the driver uses **`<base>_offset.path`**.
- Any other number of candidates (**0** or **>2**) is treated as an error with a clear message.

Example (base = `251219T040038_ROSA`):

- `.../251219T040038_ROSA.sky`
- `.../251219T040038_ROSA.path`
- `.../251219T040038_ROSA_offset.path`

In this case, the chosen `.path` is `251219T040038_ROSA_offset.path`.

#### `.path` File Format (TSV)

```
Posix_time	ms	UTC	Azimuth	Elevation	Azimuth_raw	Elevation_raw	F0	F1
1765683247	751	2025-12-14T03:34:07.751	-44.665	34.448	-4163087	3210824	1	3230_66
1765683247	864	2025-12-14T03:34:07.864	-44.665	34.448	-4163091	3210823	1	3230_66
...
```

**Columns:**
- `Posix_time` - Unix timestamp (seconds)
- `ms` - Milliseconds
- `UTC` - ISO 8601 timestamp
- `Azimuth` - Azimuth angle (degrees)
- `Elevation` - Elevation angle (degrees)
- `Azimuth_raw` - Raw encoder counts
- `Elevation_raw` - Raw encoder counts
- Additional metadata columns (optional)

#### `.sky` File Format (TSV)

**Single signal:**
```
Posix_time	ms	UTC	Signal
1765683247	751	2025-12-14T03:34:07.751	1464866
1765683247	864	2025-12-14T03:34:07.864	1522818
...
```

**Multiple signals:**
```
Posix_time	ms	UTC	Signal0	Signal1
1765683247	751	2025-12-14T03:34:07.751	1464866	1282954
1765683247	864	2025-12-14T03:34:07.864	1522818	1346253
...
```

**Columns:**
- `Posix_time` - Unix timestamp (must match `.path` file)
- `ms` - Milliseconds
- `UTC` - ISO 8601 timestamp
- `Signal` / `Signal0`, `Signal1`, ... - Power measurements

**Note:** Use `--signal N` to select which column to use when multiple signals are present.

#### Naming Conventions

Scan files follow this naming pattern:

```
YYMMDDTHHMMSS_<SITE_CODE>.<ext>
```

Examples:
- `250114T033407_OASI.path`
- `250114T033407_OASI.sky`
- `250115T121500_MZS.path`
- `250115T121500_MZS.sky`

**Excluded patterns:** Files matching `T\d{6}b` (e.g., `250101T210109bOASI`) are automatically skipped.

---

## Output Files

### Offset TSV Files

Generated by `generate_offsets.py` and saved to `offsets/<algo>.tsv`.

**Format:**
```
azimuth	offset_az	offset_el	<algorithm-specific columns>
45.2	0.123	-0.045	...
46.1	0.119	-0.042	...
...
```

**Common columns:**
- `azimuth` - Azimuth angle (degrees)
- `offset_az` - Azimuth offset (degrees, unless specified otherwise)
- `offset_el` - Elevation offset (degrees, unless specified otherwise)

Additional columns depend on the algorithm (e.g., timestamp, site code, peak power).

### Model Files

Generated by `generate_model.py` and saved to `models/`.

#### Per-Axis Models

- **`<stem>_az.joblib`** - Serialized azimuth model
- **`<stem>_el.joblib`** - Serialized elevation model
- **`<stem>_az.meta.json`** - Metadata for AZ model (backend kind, timestamp)
- **`<stem>_el.meta.json`** - Metadata for EL model

#### Unified Bundle

- **`<stem>.joblib`** - Combined AZ+EL model
- **`<stem>.meta.json`** - Bundle metadata

#### Summary Files

- **`<stem>_summary_az.txt`** - AZ fit statistics and Python code
- **`<stem>_summary_el.txt`** - EL fit statistics and Python code

**Example summary content:**
```
=== AZIMUTH POINTING MODEL (SUN_MAPS) ===
Polynomial degree: 3
Fourier harmonics: 2
Number of data points: 245
Outliers rejected (MAD > 2.5σ): 12

Residual statistics:
  Mean: 0.001°
  Std: 0.023°
  MAD: 0.015°
  Min: -0.089°
  Max: 0.095°

Python function:
def offset_az(azimuth_deg):
    """Predict azimuth offset (degrees) at given azimuth."""
    import numpy as np
    az = np.deg2rad(azimuth_deg)
    return (0.123 + 0.045*az + 0.012*az**2 - 0.003*az**3 +
            0.008*np.sin(2*az) + 0.004*np.cos(2*az))
```

#### Plot Files

- **`<stem>_az.png`** - Azimuth residual plot
- **`<stem>_el.png`** - Elevation residual plot

Plots show:
- Raw offset data (scatter)
- Fitted model (line)
- Outliers (marked differently)
- Residuals (lower panel)
- Statistics box

---

## Python API Usage

While the CLI tools cover most use cases, the Python API allows programmatic access to all functionality.

### Example: Compute Offsets Programmatically

```python
from solaris_pointing.offsets.algos import sun_maps
from types import SimpleNamespace

# Configure parameters
params = SimpleNamespace(
    data="scans/",
    outdir="offsets/",
    site_lat=-74.6950,
    site_lon=164.1000,
    site_height=30.0,
    frequency=100.0,
    enable_refraction=True,
    pressure=690,
    temperature=-7,
    signal=0
)

# Run algorithm
sun_maps.run(params)
```

### Example: Fit Model Programmatically

```python
from solaris_pointing.models import model_1d
import pandas as pd

# Load offset data
df = pd.read_csv("offsets/sun_maps.tsv", sep="\t")

# Fit azimuth model
model_az = model_1d.fit_model(
    azimuths=df["azimuth"].values,
    offsets=df["offset_az"].values,
    degree=3,
    zscore_threshold=2.5,
    ridge_alpha=0.0,
    fourier_k=2
)

# Save model
model_1d.save_model("models/my_az.joblib", model_az)

# Load and predict
loaded_model = model_1d.load_model("models/my_az.joblib")
offset_pred = model_1d.predict_offset(loaded_model, azimuth=45.0)
print(f"Predicted offset at 45°: {offset_pred:.4f}°")
```

### Example: Load and Use a Model

```python
from solaris_pointing.models import model_1d

# Load unified bundle
bundle = model_1d.load_model("models/sun_maps.joblib")

# Predict at specific azimuth
az_offset, el_offset = model_1d.predict_offsets_deg(
    bundle, azimuth=45.0
)

print(f"AZ offset: {az_offset:.4f}°")
print(f"EL offset: {el_offset:.4f}°")
```

---

## Testing

### Run All Tests

```bash
pytest
```

### Run with Coverage

```bash
pytest --cov=solaris_pointing --cov-report=html
```

Coverage report will be in `htmlcov/index.html`.

### Run Specific Test File

```bash
pytest tests/test_models.py
```

### Run Tests Across Python Versions (tox)

```bash
# Run all environments (lint, format-check, py312, py313)
tox

# Run specific environment
tox -e py313

# Run only linting
tox -e lint

# Run formatting check (CI-safe)
tox -e format-check

# Apply formatting locally
tox -e format
```

### CI/CD

The repository includes GitHub Actions workflows for:
- **Continuous Integration** - Runs tests on Python 3.12 and 3.13
- **Coverage reporting** - Uploads to Coveralls
- **Linting** - Checks code style with Ruff

---

## Contributing

Contributions are welcome! Please follow these guidelines:

### Development Setup

```bash
# Fork and clone the repository
git clone https://github.com/YOUR_USERNAME/solaris-pointing.git
cd solaris-pointing

# Install with development dependencies
poetry install

# Activate virtual environment
poetry shell

# Run tests to verify setup
pytest
```

### Code Style

This project uses **Ruff** for linting and formatting:

```bash
# Check linting
tox -e lint

# Check formatting (no changes)
tox -e format-check

# Apply formatting
tox -e format
```

### Before Submitting a PR

1. **Run the full test suite:** `tox`
2. **Ensure coverage ≥95%:** `pytest --cov=solaris_pointing --cov-fail-under=95`
3. **Check code style:** `tox -e lint && tox -e format-check`
4. **Update documentation** if adding new features
5. **Add tests** for new functionality

### Reporting Issues

Please include:
- Python version
- Operating system
- Full error traceback
- Minimal reproducible example
- Expected vs. actual behavior

---

## Troubleshooting

### Common Issues

#### **Import errors after installation**

```bash
# Ensure package is installed in editable mode
pip install -e .

# Or reinstall with Poetry
poetry install
```

#### **Missing `.path` or `.sky` files**

Error: `No valid <map_id>.path / <map_id>.sky pairs found.`

**Solution:** Verify that:
- There is at least one `.sky` file under the `--data` directory (searched recursively).
- For each `<base>.sky`, there are `.path` candidates whose filename starts with `<base>`:
  - If there is exactly 1 candidate, it will be used.
  - If there are exactly 2 candidates, an exact `<base>_offset.path` must exist (it will be used).
  - If there are 0 or >2 candidates, the driver will stop with a clear error.
- Files don't match exclusion pattern `T\d{6}b`.

You may also see errors like:
- `No .path file found for .sky base.`
- `Ambiguous .path candidates (2 found), but the required offset filename is missing.`
- `Ambiguous .path candidates (>2 found) for a single .sky base.`

In these cases, fix the naming so the rule above produces exactly 1 candidate,
or exactly 2 candidates including `<base>_offset.path`.
#### **Date parsing failures**

Error: `Cannot parse date from stem`

**Solution:** Ensure file names follow format:
- `YYMMDDTHHMMSS_<SITE>.<ext>` (preferred)
- `YYMMDDTHHMMSS<SITE>.<ext>` (also supported)

#### **Model prediction outside data range**

Error: `Azimuth X.X outside observed range [Y.Y, Z.Z]`

**Solution:** Use `--allow-extrapolation` flag (use with caution):
```bash
python scripts/generate_model.py predict model --azimuth 355 --allow-extrapolation
```

#### **Refraction calculation errors**

Error: `pysolar refraction computation failed`

**Solution:** Check that:
- `--enable-refraction` is set
- Meteo parameters are reasonable:
  - Pressure: 500-1100 hPa
  - Temperature: -50 to +50°C
  - Humidity: 0.0-1.0

#### **High residuals after fitting**

**Diagnostic steps:**
1. Check summary files for outlier count
2. Adjust `--zscore` threshold (lower = more aggressive rejection)
3. Increase `--degree` for more complex fits
4. Add `--fourier-k` for periodic patterns
5. Inspect plots for systematic trends

#### **Memory issues with large datasets**

**Solution:**
- Process data in date-range chunks with `--date-start` / `--date-end`
- Use TSV format (not in-memory dataframes)
- Consider aggregating or downsampling data before fitting

---

## Algorithms

### Current Algorithms

#### `sun_maps`

The default algorithm for solar pointing offset computation.

**Method:**
1. Load `.path` and `.sky` paired files
2. Extract azimuth/elevation time series
3. Extract power measurements (single or multi-signal)
4. Identify Sun center via 2D Gaussian fitting or peak finding
5. Compute offset as difference between commanded and measured position
6. Apply optional refraction correction
7. Apply fixed biases if configured
8. Write results to TSV

**Output columns:**
- `azimuth` - Commanded azimuth (degrees)
- `offset_az` - Azimuth offset (degrees)
- `offset_el` - Elevation offset (degrees)
- `map_id` - Scan identifier
- `timestamp` - Observation time
- `site_code` - Observatory code
- Additional algorithm-specific metadata

### Adding New Algorithms

To implement a custom algorithm:

1. Create `src/solaris_pointing/offsets/algos/my_algo.py`
2. Implement required interface:
   ```python
   def run(params):
       """Main entry point called by generate_offsets.py"""
       pass
   
   def process_map(path_file, sky_file, params):
       """Process a single scan pair, return offset dict"""
       pass
   ```
3. Use the algorithm:
   ```bash
   python scripts/generate_offsets.py --algo my_algo --data scans/
   ```

---

## Model Backends

### Current Backends

#### `model_1d`

The default 1D polynomial + Fourier model for per-axis fitting.

**Features:**
- Polynomial base (degree 0-10)
- Additive Fourier harmonics
- MAD-based outlier rejection
- Ridge regularization
- Unwrapping for azimuth continuity

**Mathematical form:**
```
offset(az) = Σ(c_i * az^i) + Σ(a_k*sin(2π*az/P_k) + b_k*cos(2π*az/P_k))
```

### Adding Custom Model Backends

1. Create `src/solaris_pointing/models/model_<kind>.py`
2. Implement required API functions (see `model_1d.py` for reference)
3. Use with `--model <kind>` flag

---

## CI/CD Badges Explained

- **CI Badge** - Shows whether automated tests pass on the `main` branch
- **Coverage Badge** - Shows percentage of code covered by tests (target: ≥95%)
- **Python Version Badge** - Indicates minimum Python version (3.12+)
- **License Badge** - Displays project license (MIT)

---

## Roadmap

Planned features and improvements:

- [ ] Additional offset algorithms (cross-scan, drift scan)
- [ ] 2D (azimuth-elevation coupled) model backends
- [ ] Real-time offset monitoring dashboard
- [ ] Automated quality metrics and alerts
- [ ] Integration with telescope control systems
- [ ] Support for other astronomical targets (planets, point sources)
- [ ] Web-based visualization tools
- [ ] Docker container for reproducible deployments

---

## Citation

If you use Solaris Pointing in your research, please cite:

```bibtex
@software{solaris_pointing,
  author = {Buttu, Marco},
  title = {Solaris Pointing: Telescope Pointing Analysis for Millimetric Observatories},
  year = {2025},
  publisher = {GitHub},
  url = {https://github.com/solaris-observatory/solaris-pointing}
}
```

---

## Acknowledgments

Developed for the **Solaris Observatory** project.

Special thanks to contributors and the solar radio astronomy community.

---

## License

This project is licensed under the **MIT License**. See [LICENSE](LICENSE) for details.

---

## Contact

**Maintainer:** Marco Buttu (marco.buttu@inaf.it)

**Repository:** https://github.com/solaris-observatory/solaris-pointing

**Issues:** https://github.com/solaris-observatory/solaris-pointing/issues

---

**Happy observing! ☀️🔭**
