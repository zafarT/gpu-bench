import pytest

from bench import cli, db

SYS = {"git_sha": "abc123", "os": "Linux", "device": "cuda",
       "gpu_name": "Fake GPU", "torch_version": "2.4"}
BASE = [1000, 1010, 990, 1005, 995, 1002, 998, 1008, 992, 1000]   # mean 1000


def _matmul(gflops):
    return {"benchmark": "matmul", "device": "cuda", "dtype": "float32", "n": 2048,
            "gflops_median": gflops, "gflops_std": 5.0, "cv": 0.005,
            "samples": [gflops] * 10}


def _seed(path, values):
    for i, v in enumerate(values):
        db.save_run({**SYS, "ts": f"2026-01-01T00:{i:02d}:00+00:00"},
                    [_matmul(v),
                     {"benchmark": "glmark2", "status": "skipped",
                      "reason": "glmark2 not installed"}],
                    path=path)


@pytest.fixture
def dbfile(tmp_path, monkeypatch):
    monkeypatch.delenv("CI", raising=False)
    return str(tmp_path / "bench.db")


def test_report_ok_exits_0(dbfile, capsys):
    _seed(dbfile, BASE + [1001])
    assert cli.main(["--db", dbfile, "report"]) == 0
    out = capsys.readouterr().out
    assert "matmul[dtype=float32,n=2048]" in out
    assert "REGRESSION" not in out
    assert "skipped" in out


def test_report_fake_regression_exits_1(dbfile, capsys):
    _seed(dbfile, BASE + [1000 * 0.7])
    assert cli.main(["--db", dbfile, "report"]) == 1
    out = capsys.readouterr().out
    assert "REGRESSION" in out
    assert "-30.00" in out                        # delta %


def test_report_not_enough_history(dbfile, capsys):
    _seed(dbfile, [1000, 1000 * 0.7])
    assert cli.main(["--db", dbfile, "report"]) == 0
    assert "not enough history" in capsys.readouterr().out


def test_report_ci_env_uses_looser_tolerance(dbfile, monkeypatch, capsys):
    _seed(dbfile, BASE + [930])                   # -7%: local fails, CI passes
    assert cli.main(["--db", dbfile, "report"]) == 1
    monkeypatch.setenv("CI", "true")
    assert cli.main(["--db", dbfile, "report"]) == 0
    assert "Tolerance [CI]" in capsys.readouterr().out


def test_baseline_excludes_other_configs(dbfile):
    _seed(dbfile, BASE)
    # a much slower float64 run must not be compared to the float32 baseline
    slow = {**_matmul(300), "dtype": "float64"}
    db.save_run(SYS, [slow], path=dbfile)
    assert cli.main(["--db", dbfile, "report"]) == 0


def test_report_empty_db(dbfile, capsys):
    assert cli.main(["--db", dbfile, "report"]) == 0
    assert "No runs" in capsys.readouterr().out


def test_db_option_after_subcommand(dbfile):
    # CI calls `bench.cli run --db results.db`, i.e. option after the subcommand
    _seed(dbfile, BASE + [700])
    assert cli.main(["report", "--db", dbfile]) == 1


def test_apply_slowdown_scales_values_not_cv():
    r = cli.apply_slowdown(_matmul(1000), 0.7)
    assert r["gflops_median"] == pytest.approx(700)
    assert r["samples"] == pytest.approx([700] * 10)
    assert r["cv"] == 0.005
    skipped = {"benchmark": "glmark2", "status": "skipped"}
    assert cli.apply_slowdown(skipped, 0.7) is skipped
