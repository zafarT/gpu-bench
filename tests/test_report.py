from bench import db, regression
from bench.__main__ import compare

SYSTEM = {"os": "Linux", "gpu": "RTX 4050", "driver": "595.84", "git_sha": "abc123"}


def matmul(value):
    return {"benchmark": "matmul", "config": "n=2048 float32 cuda",
            "status": "ok", "value": value, "std": 5.0, "unit": "GFLOP/s"}


def seed(conn, values, system=SYSTEM):
    for v in values:
        db.save(conn, system, [matmul(v)])


def test_regression_is_detected_from_the_database():
    conn = db.connect(":memory:")
    seed(conn, [1000, 1010, 990, 1005, 995])
    run_id = db.save(conn, SYSTEM, [matmul(700)])
    rows, regressed = compare(conn, run_id, SYSTEM, [matmul(700)], regression.LOCAL)
    assert regressed
    assert rows[0][5] == "REGRESSION"


def test_normal_run_passes():
    conn = db.connect(":memory:")
    seed(conn, [1000, 1010, 990, 1005, 995])
    run_id = db.save(conn, SYSTEM, [matmul(1002)])
    assert compare(conn, run_id, SYSTEM, [matmul(1002)], regression.LOCAL)[1] is False


def test_only_same_config_is_compared():
    conn = db.connect(":memory:")
    seed(conn, [1000, 1010, 990, 1005, 995])
    cpu = {**matmul(300), "config": "n=2048 float32 cpu"}     # much slower, different config
    run_id = db.save(conn, SYSTEM, [cpu])
    rows, regressed = compare(conn, run_id, SYSTEM, [cpu], regression.LOCAL)
    assert not regressed
    assert rows[0][5] == "not enough history"


def test_driver_change_is_pointed_out():
    conn = db.connect(":memory:")
    seed(conn, [1000, 1010, 990, 1005, 995], {**SYSTEM, "driver": "580.10"})
    run_id = db.save(conn, SYSTEM, [matmul(700)])
    rows, regressed = compare(conn, run_id, SYSTEM, [matmul(700)], regression.LOCAL)
    assert regressed
    assert "driver changed 580.10 -> 595.84" in rows[0][5]


def test_skipped_benchmark_is_shown_but_does_not_fail():
    conn = db.connect(":memory:")
    skipped = {"benchmark": "glmark2", "status": "skipped", "note": "glmark2 not installed"}
    run_id = db.save(conn, SYSTEM, [skipped])
    rows, regressed = compare(conn, run_id, SYSTEM, [skipped], regression.LOCAL)
    assert not regressed
    assert rows[0][5] == "skipped: glmark2 not installed"
