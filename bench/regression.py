"""Decide whether a new result is a real regression or just noise.

A result is a REGRESSION only if BOTH are true:
  * z <= -z_limit        the drop is unusual compared with past run-to-run noise
  * delta <= -pct_limit  the drop is big enough to matter (protects against a tiny std)
"""
import math
import statistics

LOCAL = {"z_limit": 3.0, "pct_limit": 5.0}   # own machine: fairly stable
CI = {"z_limit": 3.0, "pct_limit": 10.0}     # shared cloud runners are noisier
MIN_HISTORY = 3


def check(new, history, z_limit=3.0, pct_limit=5.0):
    """Return (status, baseline, delta_pct, z). Higher values are better."""
    if len(history) < MIN_HISTORY:
        return "not enough history", None, None, None

    mean = statistics.mean(history)
    std = statistics.stdev(history)
    delta_pct = (new - mean) / mean * 100
    if std > 0:
        z = (new - mean) / std
    else:                       # perfectly flat history
        z = 0.0 if new == mean else math.copysign(math.inf, new - mean)

    regressed = z <= -z_limit and delta_pct <= -pct_limit
    return ("REGRESSION" if regressed else "ok"), mean, delta_pct, z
