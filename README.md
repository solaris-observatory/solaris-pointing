# Solaris Pointing
![CI](https://github.com/solaris-observatory/solaris-pointing/actions/workflows/ci.yml/badge.svg)![Coverage Status](https://coveralls.io/repos/github/solaris-observatory/solaris-pointing/badge.svg?branch=main)

**Solaris Pointing** is a Python package designed to support the analysis and
improvement of telescope pointing. It provides tools to record, store, and analyze
pointing offsets, to build and apply pointing models, and to handle related calibration
procedures.


---

## Requirements

- **Python** >= 3.11
- **Poetry** for dependency and environment management Install Poetry (if needed):
  ```bash
  curl -sSL https://install.python-poetry.org | python3 -
  ```
  (Or follow the official instructions at [python-poetry.org](https://python-poetry.org))

---

## Installation (with Poetry)

Clone the repository and install dependencies:

```bash
git clone https://github.com/solaris-observatory/solaris-pointing
cd solaris-pointing
poetry install
```

Activate the project shell (optional):

```bash
poetry shell
```

> Alternatively, without Poetry:
> ```bash
> pip install -e .
> ```

---

## Using `offset_io`

The **`offset_io`** library produces a **standard output file** containing telescope
pointing offsets (e.g., azimuth/altitude corrections, time, metadata, etc.), so that
different algorithms can **generate compatible results**.

The easiest way to start is by reading the included [example](https://github.com/solaris-observatory/solaris-pointing/blob/main/examples/offset_io_example.py), which contains
a detailed **docstring**. In the example you will see:

- How to **import** `offset_io`
- How to **prepare** pointing offset data
- How to **write** them into the **standard format**

To integrate `offset_io` into your own code, use the [example](https://github.com/solaris-observatory/solaris-pointing/blob/main/examples/offset_io_example.py) as a template and adapt
the part where you provide the computed offsets (e.g., after a scan procedure or
pointing model correction). The goal is always to produce a **consistent, reusable
file**.


---

## Azimuth-only pointing model (`src/solaris_pointing/fitting/az_model.py`)

This module implements an **azimuth-only** pointing model: **both** offsets
(`offset_az`, `offset_el`) are modeled as functions of **azimuth** only. All
internal computations use **degrees**.

### When to use it
When your observing program targets the Sun (or similarly constrained tracks),
the elevation at a given azimuth changes slowly across days, so a short-lived
model can approximate both offsets as polynomials of azimuth.

> Tip: re-fit the model every *N* days so the approximation remains valid for
> your current observing window.

### Input data
Use a tab-separated file (TSV produced by `offset_io`) with
optional comment lines starting with `#` and these required columns:

- `azimuth` (deg)
- `elevation` (deg, **not used in the fit**, kept for QA)
- `offset_az` (deg by default)
- `offset_el` (deg by default)

If your offsets are in **arcmin** or **arcsec**, see the CLI options below.

### Key API (Python)
```python
from solaris_pointing.fitting.az_model import (
    read_offsets_tsv,      # read a TSV; converts offsets to deg if needed
    fit_models,            # fit Poly models (degree N) with z-score outlier cut
    save_models,           # save models with joblib
    load_models,           # load models with joblib
    predict_offsets_deg,   # predict (off_az, off_el) at a given azimuth [deg]
    model_summary,         # human-readable summary (poly + R²)
)
```

### CLI helper (`scripts/az_model_cli.py`)

The repo includes a small CLI to fit models or predict offsets from the shell.
This CLI is not designed to show how to compute offsets in your real-time pointing routine.
Refer to *"[Minimal runtime example (Python)](#minimal-runtime-example-python)"* instead.

**Fit and save models (input in degrees, default):**
```bash
python scripts/az_model_cli.py     templates/output_offset_io_example.tsv     --degree 3     --summary models/fit_summary.txt
```

**Fit when input offsets are in arcminutes or arcseconds:**
```bash
# arcminutes
python scripts/az_model_cli.py     templates/output_offset_io_example.tsv     --input-offset-unit arcmin     --degree 3     --summary models/fit_summary.txt

# arcseconds
python scripts/az_model_cli.py     templates/output_offset_io_example.tsv     --input-offset-unit arcsec     --degree 3     --summary models/fit_summary.txt
```

**Choose custom output paths for the saved models:**
```bash
python scripts/az_model_cli.py     templates/output_offset_io_example.tsv     --degree 3     --save-az-model custom_models/az_model.joblib     --save-el-model custom_models/el_model.joblib     --summary custom_models/fit_summary.txt
```

**Plot the fit (and optionally save a PNG):**
```bash
python scripts/az_model_cli.py     templates/output_offset_io_example.tsv     --degree 3     --plot     --plot-unit arcmin     --plot-file models/fit_plot.png
```

**Predict offsets (using saved models):**
```bash
python scripts/az_model_cli.py --predict 125.0
# or with explicit model paths:
python scripts/az_model_cli.py     --predict 125.0     --az-model models/az_model.joblib     --el-model models/el_model.joblib
```

### Minimal runtime example (Python)
See this [example](https://github.com/solaris-observatory/solaris-pointing/blob/main/examples/az_model_example.py) for a tiny integration example that loads previously fitted models, predicts
the offsets for a given azimuth, and **applies** them to your ideal pointing:

```python
from solaris_pointing.fitting.az_model import load_models, predict_offsets_deg

ideal_az_deg, ideal_el_deg = 125.0, 40.0
az_model, el_model = load_models("models/az_model.joblib", "models/el_model.joblib")
off_az_deg, off_el_deg = predict_offsets_deg(az_model, el_model, ideal_az_deg)
corr_az_deg, corr_el_deg = ideal_az_deg + off_az_deg, ideal_el_deg + off_el_deg
print(corr_az_deg, corr_el_deg)
```

> The sign convention here is `corrected = ideal + offset`. Adjust if your
> control system uses a different convention.


---

## Developer Guide (with tox)

The project includes a **`tox`** configuration to simplify common development tasks
(tests, linting, etc.).

How it works:

- `tox` (without arguments) runs lint + format-check + tests (py313)
- `tox -e lint` runs only the linter.
- `tox -e format-check` just format check (CI-safe; no changes)
- `tox -e format` applies formatting locally (modifies files).
- `tox -e py310,py311` runs tests on other Python versions if available.


---

## License

Released under the **MIT** License.
