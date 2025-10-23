from __future__ import annotations

"""
time_inference.py
=================
Interface for time inference at the peak location of a 2D map.

This module provides a stable function signature for algorithms that want
to infer the timestamp corresponding to the fitted Gaussian peak.

No progress-reporting logic is included here.
"""

from typing import Dict, Any, Iterable, Tuple


def infer_peak_time(
    peak_az_el: Tuple[float, float],
    samples: Iterable[Tuple[float, float, float]],
    cfg: Dict[str, Any],
) -> Tuple[float, Dict[str, Any]]:
    """Infer the peak timestamp and return (t_peak, meta).

    Parameters
    ----------
    peak_az_el : (float, float)
        The fitted peak position in degrees ``(az, el)``.
    samples : iterable of (az, el, t)
        Calibration samples in degrees and time scale chosen by the caller.
    cfg : dict
        Method and strategy settings, if any.

    Returns
    -------
    (t_peak, meta) : (float, dict)
        ``t_peak`` is a float in the caller's time scale. ``meta`` may contain
        keys like ``method``, ``spread_s``, or ``residual_deg``.

    Notes
    -----
    This is a stub. Replace with your project-specific numeric code.
    """
    raise NotImplementedError("Time inference requires project data access.")
