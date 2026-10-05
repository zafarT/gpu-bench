"""gpu-bench command line.

    python -m bench.cli run [--ci] [--report]   run benchmarks, store in DB
    python -m bench.cli report [--ci]           compare latest run to baseline

`report` exits with code 1 if any benchmark regressed.
"""
import argparse
import math
import os
import sys
from dataclasses import replace

from bench import db, regression, sysinfo

# benchmark -> (metric used for regression checks, higher_is_better)
PRIMARY = {
    "matmul": ("gflops_median", True),
    "glmark2": ("score_median", True),
}
# params that identify a config but are too long/noisy for the table label
_HIDDEN_PARAMS = ("device", "renderer")


def _label(bench, params):
    shown = {k: v for k, v in (params or {}).items() if k not in _HIDDEN_PARAMS}
    if not shown:
        return bench
    return f"{bench}[{','.join(f'{k}={v}' for k, v in sorted(shown.items()))}]"


def _tolerance(args):
    ci = args.ci or sysinfo.is_ci()
    tol = regression.CI if ci else regression.LOCAL
    overrides = {k: getattr(args, k) for k in ("z", "pct", "window")
                 if getattr(args, k) is not None}
    return replace(tol, **overrides), ci


def compare_run(run_id, tol, path=db.DB_PATH):
    """Return list of (label, Verdict | None, status, reason) for one run."""
    run, results = db.get_run(run_id, path)
    rows = []
    for res in results:
        bench = res["benchmark"]
        label = _label(bench, res["params"])
        if res["status"] != "ok":
            rows.append((label, None, res["status"], (res["reason"] or "").splitlines()[0:1]))
            continue
        if bench not in PRIMARY or res["metric"] != PRIMARY[bench][0]:
            continue  # std / cv rows are stored but not checked
        metric, higher = PRIMARY[bench]
        hist = db.history(bench, metric, run["os"], run["device"],
                          limit=tol.window, params=res["params"],
                          before_run=run_id, path=path)
        v = regression.check(res["value"], [h["value"] for h in hist], tol, higher)
        rows.append((label, v, v.status, None))
    return rows


def _fmt(x, spec):
    if x is None:
        return "-"
    if math.isinf(x):
        return "-inf" if x < 0 else "+inf"
    return format(x, spec)


def format_table(rows):
    header = ("benchmark", "new", "baseline", "Δ%", "z", "status")
    lines = []
    for label, v, status, reason in rows:
        if v is None:
            status = f"{status}: {reason[0]}" if reason else status
            lines.append((label, "-", "-", "-", "-", status))
        else:
            lines.append((label, _fmt(v.new, ".2f"), _fmt(v.baseline, ".2f"),
                          _fmt(v.delta_pct, "+.2f"), _fmt(v.z, "+.2f"), status))
    widths = [max(len(str(r[i])) for r in [header, *lines]) for i in range(len(header))]
    out = []
    for i, r in enumerate([header, *lines]):
        out.append("  ".join(
            str(c).ljust(w) if j in (0, 5) else str(c).rjust(w)
            for j, (c, w) in enumerate(zip(r, widths))).rstrip())
        if i == 0:
            out.append("  ".join("-" * w for w in widths))
    return "\n".join(out)


def cmd_report(args, run_id=None):
    run_id = run_id or args.run or db.latest_run_id(args.db)
    if run_id is None:
        print("No runs in database yet. Run `python -m bench.cli run` first.")
        return 0
    tol, ci = _tolerance(args)
    run, _ = db.get_run(run_id, args.db)
    if run is None:
        print(f"Run {run_id} not found.")
        return 2
    print(f"Run {run_id}  {run['ts']}  {run['git_sha'] or ''}  "
          f"{run['os']}/{run['device']}  {run['gpu_name'] or ''}")
    print(f"Tolerance [{'CI' if ci else 'local'}]: regression if z <= -{tol.z:g} "
          f"and Δ <= -{tol.pct:g}% vs last {tol.window} runs "
          f"(min {tol.min_history})\n")
    rows = compare_run(run_id, tol, args.db)
    print(format_table(rows))
    regressed = [label for label, v, *_ in rows if v is not None and v.regressed]
    if regressed:
        print(f"\nREGRESSION detected in: {', '.join(regressed)}")
        return 1
    return 0


def apply_slowdown(result, factor):
    """Demo only: scale a runner result as if the code got `factor`x as fast
    (0.7 = 30% slower). Medians/std/samples scale; cv is unchanged."""
    if factor == 1.0 or result.get("status", "ok") != "ok":
        return result
    r = dict(result)
    for key in ("gflops_median", "gflops_std", "score_median", "score_std"):
        if key in r and r[key] is not None:
            r[key] = r[key] * factor
    if r.get("samples"):
        r["samples"] = [x * factor for x in r["samples"]]
    return r


def cmd_run(args):
    from bench.runners.glmark2 import run_glmark2
    from bench.runners.matmul import run_matmul

    ci = args.ci or sysinfo.is_ci()
    slow = args.inject_slowdown
    if slow is None:
        slow = float(os.environ.get("BENCH_INJECT_SLOWDOWN") or 1.0)
    if slow != 1.0:
        print(f"!! Injecting artificial slowdown: results x{slow:g} (demo only)")

    info = sysinfo.collect()
    results = []
    if "matmul" in args.only:
        print(f"matmul n={args.n} x{args.repeats} ...", flush=True)
        results.append(run_matmul(n=args.n, repeats=args.repeats))
    if "glmark2" in args.only:
        print(f"glmark2 x{args.glmark2_repeats} ...", flush=True)
        results.append(run_glmark2(repeats=args.glmark2_repeats, ci=ci))
    results = [apply_slowdown(r, slow) for r in results]

    run_id = db.save_run(info, results, path=args.db)
    print(f"Saved run {run_id} to {args.db}\n")
    if args.report:
        return cmd_report(args, run_id)
    return 0


def _add_common(parser, suppress):
    """Global options; accepted before *or* after the subcommand
    (`bench --db x run` and `bench run --db x` both work)."""
    d = (lambda v: argparse.SUPPRESS) if suppress else (lambda v: v)
    parser.add_argument("--db", default=d(db.DB_PATH),
                        help=f"SQLite file (default: {db.DB_PATH})")
    parser.add_argument("--ci", action="store_true", default=d(False),
                        help="use CI tolerance / headless mode (auto if $CI is set)")
    parser.add_argument("--z", type=float, default=d(None), help="override z-score threshold")
    parser.add_argument("--pct", type=float, default=d(None), help="override %% threshold")
    parser.add_argument("--window", type=int, default=d(None), help="override baseline size N")


def build_parser():
    p = argparse.ArgumentParser(prog="bench", description="GPU benchmark + regression tracker")
    _add_common(p, suppress=False)
    sub = p.add_subparsers(dest="cmd", required=True)

    r = sub.add_parser("run", help="run benchmarks and store results")
    _add_common(r, suppress=True)
    r.add_argument("--only", nargs="+", choices=list(PRIMARY), default=list(PRIMARY))
    r.add_argument("--n", type=int, default=2048, help="matmul size")
    r.add_argument("--repeats", type=int, default=20, help="matmul repeats")
    r.add_argument("--glmark2-repeats", type=int, default=3)
    r.add_argument("--report", action="store_true", help="report right after running")
    r.add_argument("--inject-slowdown", type=float, metavar="FACTOR",
                   help="DEMO: multiply results by FACTOR (e.g. 0.7) to fake a "
                        "regression; also via $BENCH_INJECT_SLOWDOWN")
    r.set_defaults(func=cmd_run, run=None)

    rep = sub.add_parser("report", help="compare a run against the baseline")
    _add_common(rep, suppress=True)
    rep.add_argument("--run", type=int, help="run id (default: latest)")
    rep.set_defaults(func=cmd_report)
    return p


def main(argv=None):
    # Windows consoles/CI logs default to cp1252, which can't print "Δ".
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
