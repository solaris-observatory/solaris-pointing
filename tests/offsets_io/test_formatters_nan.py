from __future__ import annotations
import math

# We intentionally test internal helpers to cover the "NaN" branches.
from solaris_pointing.offsets import io as _io


def test_internal_formatters_return_NaN_for_none_and_nonfinite():
    # _fmt_4dec_or_nan: None -> "NaN"
    assert _io._fmt_4dec_or_nan(None) == "NaN"
    # _fmt_4dec_or_nan: non-finite -> "NaN"
    assert _io._fmt_4dec_or_nan(math.nan) == "NaN"

    # _fmt_default_or_nan: None -> "NaN"
    assert _io._fmt_default_or_nan(None) == "NaN"
    # _fmt_default_or_nan: float('nan') -> "NaN"
    assert _io._fmt_default_or_nan(math.nan) == "NaN"
    # _fmt_default_or_nan: float('inf') -> "NaN"
    assert _io._fmt_default_or_nan(math.inf) == "NaN"
