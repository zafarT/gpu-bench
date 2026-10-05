import math

from bench.regression import CI, LOCAL, check

HISTORY = [100, 102, 98, 101, 99]          # mean 100, std = sqrt(2.5) = 1.58


def test_normal_noise_is_ok():
    status, base, delta, z = check(99, HISTORY)
    assert status == "ok"
    assert base == 100
    assert math.isclose(delta, -1.0)


def test_30_percent_drop_is_a_regression():
    status, _, delta, z = check(70, HISTORY)
    assert status == "REGRESSION"
    assert math.isclose(delta, -30.0)
    assert z < -3


def test_needs_3_runs_of_history():
    assert check(70, [100, 100])[0] == "not enough history"


def test_tiny_std_alone_is_not_enough():
    # very stable history: -2% is "28 sigma", but too small to matter
    assert check(98, [100.0, 100.1, 99.9, 100.0, 100.0])[0] == "ok"


def test_big_drop_inside_huge_noise_is_not_enough():
    # -15%, but the history jumps around by +-30%, so z is only -0.6
    assert check(85, [100, 130, 70, 120, 80])[0] == "ok"


def test_ci_limits_are_looser():
    assert check(93, HISTORY, **LOCAL)[0] == "REGRESSION"   # -7%
    assert check(93, HISTORY, **CI)[0] == "ok"


def test_improvement_is_ok():
    assert check(130, HISTORY)[0] == "ok"
