# Solaris Pointing
![CI](https://github.com/solaris-observatory/solaris-pointing/actions/workflows/ci.yml/badge.svg)![Coverage Status](https://coveralls.io/repos/github/solaris-observatory/solaris-pointing/badge.svg?branch=main)

# Solaris Pointing

Solaris Pointing provides command-line tools to **compute solar pointing offsets** from Sun scan maps and to **fit azimuth/elevation pointing models** (polynomial + Fourier).  
It is designed for telescope operations, diagnostics, and production of stable correction models used at millimetric observatories.

This repository exposes **two user-facing CLIs**:

- `generate_offsets.py` — discover Sun scan pairs, compute offset time series, and write a clean TSV.
- `generate_model.py` — fit pointing models (AZ/EL), generate summaries and plots, and produce ready-to-use `.joblib` bundles.

Both scripts include comprehensive built-in `--examples`.

---

## Quickstart

### 1) Compute offsets from Sun scans

```bash
python scripts/generate_offsets.py --data scans/ --algo sun_maps
```

This produces:

```
offsets/
└── sun_maps.tsv
```

Each line contains `azimuth`, `offset_az`, `offset_el`, metadata, and algorithm-specific fields.

---


### Using a configuration profile

```bash
python scripts/generate_offsets.py --algo sun_maps --config <profile> --data scans/
```

Profiles are loaded from `config/<profile>.toml` and override default CLI parameters.
### 2) Fit pointing models from one or more TSV files

```bash
python scripts/generate_model.py fit offsets/sun_maps.tsv
```

This writes per-axis models, summaries, and plots:

```
models/
├── sun_maps_az.joblib
├── sun_maps_el.joblib
├── sun_maps_summary_az.txt
├── sun_maps_summary_el.txt
├── sun_maps_az.png
├── sun_maps_el.png
└── sun_maps.joblib
```

---

## What this package provides

- **Offset computation** from raw Sun scans (`.path` + `.sky` pairs), including:
  - recursive scan discovery;
  - date filtering via stem timestamps: `YYMMDDTHHMMSS...`;
  - atmospheric refraction (optional);
  - telescope & site metadata;
  - power/peak selection thresholds.

- **Pointing-model fitting** with:
  - polynomial degree `--degree`;
  - MAD-based outlier rejection (`--zscore`);
  - ridge regularization (`--ridge-alpha`);
  - Fourier terms (`--fourier-k`);
  - custom periods (`--periods-deg`);
  - per-axis or unified fits;
  - auto-generated Python function for each axis (included in summaries).

- **Model prediction** at arbitrary azimuths.

- **Model merging** to create unified bundles.

---

# Examples

## 1) `generate_offsets.py`

### Minimal run

```bash
python scripts/generate_offsets.py --data scans/ --algo sun_maps
```

### Date filters

```bash
python scripts/generate_offsets.py --data scans/ --algo sun_maps --date-start 2025-01-01
python scripts/generate_offsets.py --data scans/ --algo sun_maps --date-end 2025-01-02
python scripts/generate_offsets.py --data scans/ --algo sun_maps --date-start 2025-01-01 --date-end 2025-01-03
```

### Observatory metadata

```bash
python scripts/generate_offsets.py --data scans/ --algo sun_maps     --site-location "Antarctica" --site-code MZS     --site-lat -74.6950 --site-lon 164.1000 --site-height 30
```

### Telescope parameters

```bash
python scripts/generate_offsets.py --data scans/ --algo sun_maps     --frequency 100 --diameter 2.0
```

### Atmospheric refraction

```bash
python scripts/generate_offsets.py --data scans/ --algo sun_maps     --enable-refraction --pressure 990 --temperature -5 --humidity 0.5 --obswl 3.0
```

### Biases

```bash
python scripts/generate_offsets.py --data scans/ --algo sun_maps     --az-offset-bias 0.10 --el-offset-bias -0.05
```

### Custom output directory

```bash
python scripts/generate_offsets.py --data scans/ --algo sun_maps --outdir offsets_run_42
```

---

## 2) `generate_model.py`

### Minimal fit

```bash
python scripts/generate_model.py fit offsets/sun_maps.tsv
```

### Polynomial + Fourier

```bash
python scripts/generate_model.py fit scans.tsv     --degree 3 --zscore 2.5 --fourier-k 2 --plot-unit arcmin
```

### Axis selection

```bash
python scripts/generate_model.py fit input.tsv --az --degree 3 --fourier-k 1
python scripts/generate_model.py fit input.tsv --el --degree 3 --fourier-k 1
```

### Multiple TSVs

```bash
python scripts/generate_model.py fit a.tsv b.tsv --degree 2 --zscore 2.0
```

### Input offset units

```bash
python scripts/generate_model.py fit offsets.tsv --input-offset-unit arcsec
```

### Predict offsets

```bash
python scripts/generate_model.py predict sun_maps --azimuth 12.0 --unit arcsec
python scripts/generate_model.py predict sun_maps --az --azimuth 45.0 --unit arcmin
python scripts/generate_model.py predict sun_maps --el --azimuth 45.0 --unit arcmin
python scripts/generate_model.py predict sun_maps --az --azimuth 355.0 --allow-extrapolation
```

### Merge models

```bash
python scripts/generate_model.py merge sun_maps
```

---

# Installation

## With Poetry

```bash
poetry install
poetry shell
```

## With pip

Runtime-only:

```bash
pip install -r requirements.txt
```

Development:

```bash
pip install -r requirements-dev.txt
```

---

# Repository structure

```
scripts/
src/solaris_pointing/
models/
offsets/
tests/
```

---

# License

MIT License.

---

# Configuration profiles (`--config`)

`generate_offsets.py` supports loading configuration profiles from `config/<name>.toml`.

A profile lets you store site parameters, telescope metadata, refraction settings,
biases, thresholds, and any other CLI option — without typing long commands every time.

## Usage

```bash
python scripts/generate_offsets.py --algo sun_maps --config mzs_default --data scans/
```

This loads:

```
config/mzs_default.toml
```

All keys in the TOML file override the default CLI parameters.
Unknown keys trigger a non-blocking warning.

## Example TOML profile

```toml
site-location = "Antarctica"
site-code = "MZS"
site-lat = -74.6950
site-lon = 164.1000
site-height = 30

enable-refraction = true
pressure = 690
temperature = -7
humidity = 0.5
obswl = 3.0

frequency = 100
diameter = 2.0

az-offset-bias = 0.10
el-offset-bias = -0.05
```

Using profiles is recommended for observatory operations, where repeatability and
clean reproducibility matter.
