"""Run the benchmarks, store the results, compare them with history.

    python -m bench                     run everything
    python -m bench --only matmul       run one benchmark
    python -m bench --slowdown 0.7      demo: fake a 30% slowdown

Exits with code 1 if a regression was found, so CI turns red.
"""
import argparse
import os
import sys

from bench import db, regression


def compare(conn, run_id, system, results, limits):
    """Build the report rows. Returns (rows, regressed?)."""
    rows, regressed = [], False
    for r in results:
        if r["status"] != "ok":
            first_line = (r.get("note") or "").splitlines()[:1]
            rows.append([r["benchmark"], "-", "-", "-", "-",
                         r["status"] + (": " + first_line[0] if first_line else "")])
            continue

        hist = db.history(conn, system["os"], r["benchmark"], r["config"], run_id)
        status, base, delta, z = regression.check(
            r["value"], [h["value"] for h in hist], **limits)
        regressed = regressed or status == "REGRESSION"

        # Driver updates are a common cause of GPU regressions: point them out.
        if hist and hist[0]["driver"] and system.get("driver") and hist[0]["driver"] != system["driver"]:
            status += f" (driver changed {hist[0]['driver']} -> {system['driver']})"

        fmt = lambda x, spec: "-" if x is None else format(x, spec)
        rows.append([f"{r['benchmark']} [{r['config']}]", fmt(r["value"], ".1f"),
                     fmt(base, ".1f"), fmt(delta, "+.1f"), fmt(z, "+.1f"), status])
    return rows, regressed


def print_table(rows):
    header = ["benchmark", "new", "baseline", "delta%", "z", "status"]
    widths = [max(len(str(row[i])) for row in [header] + rows) for i in range(len(header))]
    for row in [header] + rows:
        print("  ".join(str(c).ljust(w) for c, w in zip(row, widths)).rstrip())


def main(argv=None):
    p = argparse.ArgumentParser(prog="python -m bench", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--db", default="results.db", help="SQLite file (default: results.db)")
    p.add_argument("--only", choices=["matmul", "glmark2"])
    p.add_argument("--slowdown", type=float,
                   default=float(os.environ.get("BENCH_SLOWDOWN") or 1.0),
                   help="demo: multiply results by this factor (0.7 = 30%% slower)")
    args = p.parse_args(argv)

    from bench import system                 # imported here: they need torch
    from bench.glmark2 import run_glmark2
    from bench.matmul import run_matmul

    ci = os.environ.get("CI") == "true"      # GitHub Actions sets CI=true
    limits = regression.CI if ci else regression.LOCAL

    results = []
    if args.only in (None, "matmul"):
        print("Running matmul ...", flush=True)
        results.append(run_matmul())
    if args.only in (None, "glmark2"):
        print("Running glmark2 ...", flush=True)
        results.append(run_glmark2(ci=ci))

    if args.slowdown != 1.0:
        print(f"!! DEMO: results multiplied by {args.slowdown}")
        for r in results:
            if r["status"] == "ok":
                r["value"] *= args.slowdown

    info = system.info()
    conn = db.connect(args.db)
    run_id = db.save(conn, info, results)

    print(f"\nRun {run_id} | {info['os']} | GPU: {info['gpu'] or 'none (CPU)'} | "
          f"driver: {info['driver'] or '-'} | commit: {info['git_sha'] or '-'}")
    print(f"Regression if z <= -{limits['z_limit']:g} and delta <= -{limits['pct_limit']:g}% "
          f"({'CI' if ci else 'local'} limits)\n")
    rows, regressed = compare(conn, run_id, info, results, limits)
    print_table(rows)

    if regressed:
        print("\nREGRESSION DETECTED")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
