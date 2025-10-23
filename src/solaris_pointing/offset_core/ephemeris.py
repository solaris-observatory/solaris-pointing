from __future__ import annotations

from typing import Tuple, Optional, Dict, Any


def get_sun_altaz(
    timestamp_iso: str,
    site: Dict[str, Any],
    env: Dict[str, Any],
    backend: str = "astropy",
    refraction: str = "apparent",
) -> Tuple[float, float]:
    """Return (az_deg, el_deg) for the Sun at given site and time.
    This is a lightweight facade. It allows switching backends and
    refraction modes. Implementation details are abstracted away.
    """
    if backend not in ("astropy", "pysolar"):
        raise ValueError(f"Unsupported ephemeris backend: {backend}")
    # Placeholder implementations; integrate real libs in your runtime.
    # Here we just raise to avoid silent wrong outputs.
    raise NotImplementedError(
        "Ephemeris backend not wired. Plug astropy or pysolar here."
    )
