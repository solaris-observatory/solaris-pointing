# Solaris Pointing
![CI](https://github.com/solaris-observatory/solaris-pointing/actions/workflows/ci.yml/badge.svg)![Coverage Status](https://coveralls.io/repos/github/solaris-observatory/solaris-pointing/badge.svg?branch=main)

This library provides a framework for the following workflow:
**1)** creating a standard offset file from maps;
**2)** building a pointing model from the offset file;
**3)** applying the model during pointing.
Each of these steps is explained in detail after the *Installation* section.


---

## Installation

The recommended way to install solaris-pointing is by combining
[pyenv](https://github.com/pyenv/pyenv) and
[poetry](https://python-poetry.org)). With *pyenv* you can easily
manage multiple Python versions, while *Poetry* takes care of creating
virtual environments, handling dependencies, and packaging your project.


---

### Install and Configure pyenv

To install [pyenv](https://github.com/pyenv/pyenv)  on Linux/macOS:

```bash
curl https://pyenv.run | bash
```

Then add the following lines to your shell configuration file
(*~/.bashrc*, *~/.zshrc*, etc.):

```bash
export PATH="$HOME/.pyenv/bin:$PATH"
eval "$(pyenv init -)"
eval "$(pyenv virtualenv-init -)"
```

Reload your shell:

```
exec "$SHELL"
```

Install Python 3.11:

```
pyenv install 3.11.12
```

Activate the shell:

```
pyenv shell 3.11.12
```

Verify the installation:

```
python --version
```

You should see ``Python 3.11.12``

Now your shell is running Python 3.11 through *pyenv*.


### Install Poetry

On Linux/macOS:

```bash
curl -sSL https://install.python-poetry.org | python3 -
```

Add Poetry to your PATH (if not already there):

```bash
# Bash/Zsh
echo 'export PATH="$HOME/.local/bin:$PATH"' >> ~/.bashrc    # or ~/.zshrc
```

Reload your shell and install `poetry-plugin-shell`:

```bash
exec "$SHELL"
poetry self add poetry-plugin-shell
```

### Install solaris-pointing

```bash
git clone https://github.com/solaris-observatory/solaris-pointing.git
cd solaris-pointing
pyenv shell 3.11.12 # Activate the pyenv 3.11.12 shell
poetry install # Install project + dev dependencies
```

Whenever you want to use *solaris-pointing*, you just need to activate the
right shells:

```bash
pyenv shell 3.11.12
poetry shell   # Run this inside the solaris-pointing directory
```

The first command, ``pyenv shell 3.11.12``, tells your terminal to use
Python version 3.11.12 for this session. The second command,
``poetry shell``, activates the virtual environment that
Poetry created for solaris-pointing. This ensures you are running
Python with all the correct dependencies for the project.

To check the installation, run an example from the root of
``solaris-pointing``:

```bash
python examples/offset_io_example.py
```

---

## Workflow

Now let's look in detail at the following steps of the workflow:

* creating a standard offset file from maps;
* building a pointing model from the offset file;
* applying the model during pointing.


### Creating a standard offset file

Any script responsible for calculating offsets should use the
``offset_io`` module from solaris-pointing to generate a standard
output file. This file is required as input for step 2 of the workflow.

The easiest way to start is by reading the included [example](https://github.com/solaris-observatory/solaris-pointing/blob/main/examples/offset_io_example.py), which contains
a detailed **docstring**. In the example you will see:

- How to **import** `offset_io`
- How to **prepare** pointing offset data
- How to **write** them into the **standard format**

To integrate `offset_io` into your own code, use the [example](https://github.com/solaris-observatory/solaris-pointing/blob/main/examples/offset_io_example.py) as a template.


### Building a pointing model from the offset file

Once the offset file has been generated, you can use the ``az_model_cli.py``
script to create the pointing model. Suppose the offset file is located
in the *templates* directory at the root of ``solaris-pointing``. Here's
how to create a pointing model (in this case, of degree 5):

```bash
python scripts/az_model_cli.py templates/output_offset_io_example.tsv --degree 5 --plot --plot-file models/fit_plot.png
```

Note that the script ``az_model_cli.py`` creates an **azimuth-only** pointing
model: **both** offsets (`offset_az`, `offset_el`) are modeled as functions
of **azimuth** only. That's because, when the target is the Sun, the elevation
at a given azimuth changes slowly across days, so a short-lived model can
approximate both offsets as polynomials of azimuth.

**Tip**: re-fit the model every *N* days so the approximation remains valid for
your current observing window.

#### More about creating the model

This CLI ``az_model_cli.py`` is not designed to show how to compute
offsets in your real-time pointing routine. Refer to
*"[Applying the model during pointing](#applying-the-model-during-pointing)"*
instead.

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

### Applying the model during pointing.

Use the model to apply offsets when creating a map or, more generally,
when pointing to a source. See this [example](https://github.com/solaris-observatory/solaris-pointing/blob/main/examples/az_model_example.py) for a tiny integration
example that loads previously fitted models, predicts
the offsets for a given azimuth, and applies them to your ideal pointing.
Here is a summary:

```python
from solaris_pointing.fitting.az_model import load_models, predict_offsets_deg

ideal_az_deg, ideal_el_deg = 125.0, 40.0
az_model, el_model = load_models("models/az_model.joblib", "models/el_model.joblib")
off_az_deg, off_el_deg = predict_offsets_deg(az_model, el_model, ideal_az_deg)
corr_az_deg, corr_el_deg = ideal_az_deg + off_az_deg, ideal_el_deg + off_el_deg
print(corr_az_deg, corr_el_deg)
```

The sign convention here is `corrected = ideal + offset`.

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
