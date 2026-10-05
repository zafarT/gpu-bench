"""Statistical regression check against a rolling baseline.

A new value is a REGRESSION only if BOTH hold (for higher-is-better metrics):
  * z = (new - mu) / sigma <= -z_threshold      (statistically unusual), and
  * delta% = (new - mu) / mu * 100 <= -pct      (practically significant;
                                                 protects against tiny sigma)
where mu / sigma are the mean / sample std-dev of the last `window` runs.
With fewer than `min_history` runs nothing is flagged.
"""
import math
import statistics
from dataclasses import dataclass

OK = "ok"
REGRESSION = "REGRESSION"
NO_HISTORY = "not enough history"


@dataclass(frozen=True)
class Tolerance:
    z: float = 3.0          # flag if z <= -z
    pct: float = 5.0        # ... and at least pct% worse than the baseline mean
    window: int = 10        # baseline = last N runs
    min_history: int = 3


LOCAL = Tolerance(z=3.0, pct=5.0)
CI = Tolerance(z=3.0, pct=10.0)   # shared cloud runners are noisy -> looser


@dataclass(frozen=True)
class Verdict:
    new: float
    baseline: float | None    # mu
    std: float | None         # sigma
    delta_pct: float | None   # negative = worse
    z: float | None           # negative = worse
    n: int                    # history size actually used
    status: str

    @property
    def regressed(self):
        return self.status == REGRESSION


def check(new, history, tol=LOCAL, higher_is_better=True):
    """Compare `new` against `history` (newest first; only the first
    tol.window values are used). z and delta_pct are oriented so that
    negative always means "worse"."""
    hist = [float(v) for v in list(history)[:tol.window]]
    if len(hist) < tol.min_history:
        return Verdict(new, None, None, None, None, len(hist), NO_HISTORY)

    mu = statistics.mean(hist)
    sigma = statistics.stdev(hist)
    diff = (new - mu) if higher_is_better else (mu - new)   # > 0 = better

    if sigma > 0:
        z = diff / sigma
    else:  # perfectly flat history: any change is "infinitely" many sigmas
        z = 0.0 if diff == 0 else math.copysign(math.inf, diff)

    if mu != 0:
        delta_pct = diff / abs(mu) * 100
    else:
        delta_pct = 0.0 if diff == 0 else math.copysign(math.inf, diff)

    regressed = z <= -tol.z and delta_pct <= -tol.pct
    return Verdict(new, mu, sigma, delta_pct, z, len(hist),
                   REGRESSION if regressed else OK)
