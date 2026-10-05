"""SQLite storage: one row per benchmark per run."""
import sqlite3
from datetime import datetime, timezone

SCHEMA = """
CREATE TABLE IF NOT EXISTS results (
    id        INTEGER PRIMARY KEY,
    run_id    INTEGER NOT NULL,     -- all benchmarks of one invocation share a run_id
    ts        TEXT NOT NULL,
    git_sha   TEXT,
    os        TEXT,
    gpu       TEXT,
    driver    TEXT,                 -- NVIDIA driver version
    benchmark TEXT NOT NULL,        -- 'matmul' or 'glmark2'
    config    TEXT,                 -- e.g. 'n=2048 float32 cuda', or the GL renderer
    status    TEXT NOT NULL,        -- 'ok', 'skipped' or 'error'
    value     REAL,                 -- median of the repeats
    std       REAL,                 -- spread of the repeats
    unit      TEXT,
    note      TEXT                  -- reason for skipped / error
)
"""


def connect(path):
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    conn.execute(SCHEMA)
    return conn


def save(conn, system, results):
    """Store one run. Returns its run_id."""
    run_id = conn.execute("SELECT COALESCE(MAX(run_id), 0) + 1 FROM results").fetchone()[0]
    ts = datetime.now(timezone.utc).isoformat(timespec="seconds")
    for r in results:
        conn.execute(
            "INSERT INTO results (run_id, ts, git_sha, os, gpu, driver, benchmark, "
            "config, status, value, std, unit, note) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (run_id, ts, system.get("git_sha"), system.get("os"), system.get("gpu"),
             system.get("driver"), r["benchmark"], r.get("config"), r["status"],
             r.get("value"), r.get("std"), r.get("unit"), r.get("note")))
    conn.commit()
    return run_id


def history(conn, os_name, benchmark, config, before_run, limit=10):
    """Last `limit` successful results with the same OS + benchmark + config,
    older than `before_run`, newest first."""
    rows = conn.execute(
        "SELECT value, driver FROM results "
        "WHERE os = ? AND benchmark = ? AND config = ? AND status = 'ok' AND run_id < ? "
        "ORDER BY run_id DESC LIMIT ?",
        (os_name, benchmark, config, before_run, limit))
    return [dict(row) for row in rows]
