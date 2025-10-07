
from __future__ import annotations
"""
algo_dummy.py
-------------
A minimal plugin conforming to the required API. It does **no** real computation:
for each map, it emits fixed offsets and copies the measured az/el passed in via
mock inputs (here we just reuse the Sun position as az/el for determinism).
This is intended **only** for integration tests of the runner and I/O.
"""

from typing import Iterable
from solaris_pointing.offset_core.model import Site, MapInput, Config, WriterFn

def compute_offsets(maps: Iterable[MapInput], site: Site, cfg: Config, write_row: WriterFn) -> None:
    for mp in maps:
        # Emit deterministic values (zeros) so tests can assert exact numbers.
        # az/el set to 0 for simplicity; offsets 0.
        write_row(mp.map_timestamp_iso, 0.0, 0.0, 0.0, 0.0)
