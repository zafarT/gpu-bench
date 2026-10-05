import os
import json
import sqlite3
from contextlib import closing
from datetime import datetime, timezone
from pathlib import Path

# Default DB lives at the repo root (gpu-bench/bench.db), override with BENCH_DB.
DB_PATH = os.environ.get("BENCH_DB", str(Path(__file__).resolve().parent.parent / "bench.db"))

SCHEMA = """
CREATE TABLE IF NOT EXISTS runs (
    id            INTEGER PRIMARY KEY,
    ts            TEXT NOT NULL,
    git_sha       TEXT,
    os            TEXT,
    device        TEXT,
    gpu_name      TEXT,
    torch_version TEXT
);

CREATE TABLE IF NOT EXISTS results (
    id        INTEGER PRIMARY KEY,
    run_id    INTEGER NOT NULL REFERENCES runs(id) ON DELETE CASCADE,
    benchmark TEXT NOT NULL,             -- 'matmul', 'glmark2', ...
    status    TEXT NOT NULL DEFAULT 'ok',-- 'ok' | 'skipped' | 'error'
    metric    TEXT,                      -- 'gflops_median', 'score_median', ... (NULL if not ok)
    value     REAL,
    unit      TEXT,
    params    TEXT,                      -- JSON: config that must match to compare (n, dtype, renderer...)
    samples   TEXT,                      -- JSON list of raw samples
    reason    TEXT                       -- skip/error reason, log tail
);

CREATE INDEX IF NOT EXISTS idx_results_lookup ON results(benchmark, metric);
CREATE INDEX IF NOT EXISTS idx_runs_os_device ON runs(os, device);
"""

# benchmark -> list of (result_key, metric_name, unit)
# The *_median metric is the run's value; cv (std/mean) is the noise indicator.
METRICS = {
    "matmul": [
        ("gflops_median", "gflops_median", "GFLOP/s"),
        ("gflops_std",    "gflops_std",    "GFLOP/s"),
        ("cv",            "cv",            "ratio"),
    ],
    "glmark2": [
        ("score_median", "score_median", "score"),
        ("score_std",    "score_std",    "score"),
        ("cv",           "cv",           "ratio"),
    ],
}

# benchmark -> keys that describe the configuration (stored as JSON in `params`)
PARAM_KEYS = {
    "matmul": ("device", "dtype", "n"),
    "glmark2": ("renderer",),
}


def _dumps(obj):
    return json.dumps(obj, sort_keys=True) if obj is not None else None


def _loads(s):
    return json.loads(s) if s else None


def connect(path=DB_PATH):
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    init_db(conn)
    return conn


def init_db(conn):
    conn.executescript(SCHEMA)
    conn.commit()


def _flatten(result):
    """Turn one runner dict (from run_matmul / run_glmark2) into result rows."""
    bench = result["benchmark"]
    status = result.get("status", "ok")
    params = {k: result[k] for k in PARAM_KEYS.get(bench, ()) if k in result} or None
    samples = result.get("samples")

    if status != "ok":
        reason = result.get("reason")
        if result.get("log_tail"):
            reason = f"{reason}\n{result['log_tail']}"
        return [(bench, status, None, None, None, _dumps(params), None, reason)]

    rows = []
    for key, metric, unit in METRICS.get(bench, []):
        if key in result and result[key] is not None:
            rows.append((bench, status, metric, float(result[key]), unit,
                         _dumps(params), _dumps(samples), None))
    return rows


def save_run(sysinfo, results, path=DB_PATH):
    """Store one run (system info + list of runner result dicts). Returns run_id."""
    ts = sysinfo.get("ts") or datetime.now(timezone.utc).isoformat(timespec="seconds")
    with closing(connect(path)) as conn, conn:   # inner `conn` = transaction
        cur = conn.execute(
            "INSERT INTO runs (ts, git_sha, os, device, gpu_name, torch_version) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (ts, sysinfo.get("git_sha"), sysinfo.get("os"), sysinfo.get("device"),
             sysinfo.get("gpu_name"), sysinfo.get("torch_version")),
        )
        run_id = cur.lastrowid
        rows = [(run_id, *row) for r in results for row in _flatten(r)]
        conn.executemany(
            "INSERT INTO results (run_id, benchmark, status, metric, value, unit, "
            "params, samples, reason) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            rows,
        )
        return run_id


def latest_run_id(path=DB_PATH):
    with closing(connect(path)) as conn:
        row = conn.execute("SELECT MAX(id) FROM runs").fetchone()
    return row[0]


def get_run(run_id, path=DB_PATH):
    """Return (run dict, list of result dicts) for one run, or (None, [])."""
    with closing(connect(path)) as conn:
        run = conn.execute("SELECT * FROM runs WHERE id = ?", (run_id,)).fetchone()
        if run is None:
            return None, []
        res = conn.execute(
            "SELECT * FROM results WHERE run_id = ? ORDER BY id", (run_id,)
        ).fetchall()
    results = []
    for r in res:
        d = dict(r)
        d["params"] = _loads(d["params"])
        d["samples"] = _loads(d["samples"])
        results.append(d)
    return dict(run), results


def history(benchmark, metric, os, device, limit=20, params=None,
            before_run=None, path=DB_PATH):
    """Most recent successful values for a benchmark/metric on a given os+device.

    `params` (e.g. {"device": "cuda", "dtype": "float32", "n": 2048}) restricts to
    runs with the exact same configuration so comparisons are like-for-like.
    `before_run` only returns runs older than that run id (used as the baseline).
    Returns a list of dicts, newest first.
    """
    sql = (
        "SELECT r.id AS run_id, r.ts, r.git_sha, r.os, r.device, r.gpu_name, "
        "r.torch_version, res.value, res.unit, res.params, res.samples "
        "FROM runs r JOIN results res ON r.id = res.run_id "
        "WHERE res.benchmark = ? AND res.metric = ? AND res.status = 'ok' "
        "AND r.os = ? AND r.device = ? "
    )
    args = [benchmark, metric, os, device]
    if params is not None:
        sql += "AND res.params = ? "
        args.append(_dumps(params))
    if before_run is not None:
        sql += "AND r.id < ? "
        args.append(before_run)
    sql += "ORDER BY r.ts DESC, r.id DESC LIMIT ?"
    args.append(limit)

    with closing(connect(path)) as conn:
        rows = conn.execute(sql, args).fetchall()

    out = []
    for row in rows:
        d = dict(row)
        d["params"] = _loads(d["params"])
        d["samples"] = _loads(d["samples"])
        out.append(d)
    return out


if __name__ == "__main__":
    with closing(connect()) as conn:
        pass
    print(f"Initialized {DB_PATH}")
