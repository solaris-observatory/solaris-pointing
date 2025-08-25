# Solaris Pointing
![CI](https://github.com/solaris-observatory/solaris-pointing/actions/workflows/ci.yml/badge.svg)

**Solaris Pointing** is a Python package designed to support the analysis and
improvement of telescope pointing. It provides tools to record, store, and analyze
pointing offsets, to build and apply pointing models, and to handle related calibration
procedures.


---

## Requirements

- **Python 3.x**
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
