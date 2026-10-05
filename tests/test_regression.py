import math

import pytest

from bench.regression import CI, LOCAL, NO_HISTORY, OK, REGRESSION, Tolerance, check

# mean = 100, sample std = sqrt(10 / 4) = 1.5811...
HIST = [100, 102, 98, 101, 99]
SIGMA = math.sqrt(2.5)


def test_baseline_stats_are_correct():
    v = check(99, HIST)
    assert v.baseline == 100
    assert v.std == pytest.approx(SIGMA)
    assert v.delta_pct == pytest.approx(-1.0)
    assert v.z == pytest.approx(-1 / SIGMA)       # -0.632
    assert v.n == 5
    assert v.status == OK


def test_fake_regression_times_0_7_is_flagged():
    new = 100 * 0.7
    v = check(new, HIST)
    assert v.status == REGRESSION
    assert v.regressed
    assert v.delta_pct == pytest.approx(-30.0)
    assert v.z == pytest.approx(-30 / SIGMA)
    assert check(new, HIST, CI).regressed         # big drops fail CI too


def test_exactly_at_both_thresholds_is_regression():
    # 95: z = -5 / 1.58 = -3.16 (<= -3) and delta = -5% (<= -5%)
    assert check(95, HIST, LOCAL).status == REGRESSION


def test_ci_tolerance_is_looser_than_local():
    assert CI.pct > LOCAL.pct
    # -7%, z = -4.4: regression locally, but within CI noise allowance
    assert check(93, HIST, LOCAL).status == REGRESSION
    assert check(93, HIST, CI).status == OK


def test_tiny_sigma_needs_percent_threshold_too():
    hist = [100.0, 100.1, 99.9, 100.0, 100.0]     # sigma ~= 0.07
    v = check(98, hist)                           # z ~= -28, but only -2%
    assert v.z < -3
    assert v.status == OK


def test_noisy_history_needs_z_threshold_too():
    hist = [100, 130, 70, 120, 80]                # sigma = sqrt(650) = 25.5
    v = check(85, hist)                           # -15%, but z = -0.59
    assert v.delta_pct == pytest.approx(-15.0)
    assert v.z == pytest.approx(-15 / math.sqrt(650))
    assert v.status == OK


@pytest.mark.parametrize("hist", [[], [100], [100, 101]])
def test_fewer_than_3_runs_is_not_enough_history(hist):
    v = check(70, hist)
    assert v.status == NO_HISTORY
    assert not v.regressed
    assert v.baseline is None and v.z is None


def test_improvement_is_ok():
    v = check(130, HIST)
    assert v.z > 3 and v.status == OK


def test_only_last_window_runs_are_used():
    # newest first: 10 recent runs ~100, then old slow runs at 50
    hist = [100, 102, 98, 101, 99] * 2 + [50] * 10
    v = check(99, hist, Tolerance(window=10))
    assert v.n == 10
    assert v.baseline == 100


def test_flat_history_zero_sigma():
    assert check(100, [100] * 5).status == OK
    v = check(70, [100] * 5)
    assert v.z == -math.inf
    assert v.status == REGRESSION


def test_lower_is_better_metric():
    latency = [10, 10.2, 9.8, 10.1, 9.9]
    assert check(10 / 0.7, latency, higher_is_better=False).status == REGRESSION
    assert check(7, latency, higher_is_better=False).status == OK
