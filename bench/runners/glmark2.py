import os, re, shutil, platform, statistics, subprocess

SCORE_RE = re.compile(r"glmark2 Score:\s*(\d+)")
RENDERER_RE = re.compile(r"GL_RENDERER:\s*(.+)")


def _nvidia_present():
    return shutil.which("nvidia-smi") is not None


def parse_output(out):
    """Parse glmark2 stdout/stderr. Returns (score:int|None, renderer:str|None)."""
    m = SCORE_RE.search(out)
    rm = RENDERER_RE.search(out)
    score = int(m.group(1)) if m else None
    renderer = rm.group(1).strip() if rm else None
    return score, renderer


def run_glmark2(repeats=3, timeout=600, ci=False,
                scenes=("build:use-vbo=true", "texture", "shading")):
    if shutil.which("glmark2") is None:
        return {"benchmark": "glmark2", "status": "skipped",
                "reason": "glmark2 not installed"}

    env = os.environ.copy()
    cmd = ["glmark2", "--off-screen"]
    for s in scenes:
        cmd += ["-b", s]

    if platform.system() == "Linux":
        if ci:                                   # headless, no GPU
            env["LIBGL_ALWAYS_SOFTWARE"] = "1"
            cmd = ["xvfb-run", "-a"] + cmd
        elif _nvidia_present():                  # hybrid laptop: force NVIDIA
            env["__NV_PRIME_RENDER_OFFLOAD"] = "1"
            env["__GLX_VENDOR_LIBRARY_NAME"] = "nvidia"

    scores, renderer = [], None
    for i in range(repeats):
        try:
            r = subprocess.run(cmd, env=env, capture_output=True,
                               text=True, timeout=timeout)
        except subprocess.TimeoutExpired:
            return {"benchmark": "glmark2", "status": "error",
                    "reason": f"timeout after {timeout}s"}

        out = r.stdout + r.stderr
        score, rend = parse_output(out)
        if r.returncode != 0 or score is None:
            return {"benchmark": "glmark2", "status": "error",
                    "reason": f"exit code {r.returncode}",
                    "log_tail": out[-1000:]}      # last lines, for debugging
        scores.append(score)
        if rend:
            renderer = rend

    std = statistics.stdev(scores) if len(scores) > 1 else 0.0
    mean = statistics.mean(scores)
    return {
        "benchmark": "glmark2",
        "status": "ok",
        "renderer": renderer,                    # proves which GPU ran it
        "score_median": statistics.median(scores),
        "score_std": std,
        "cv": std / mean if mean else 0.0,       # noise level, e.g. 0.02 = 2%
        "samples": scores,
    }

if __name__ == "__main__":
    import sys
    print(run_glmark2(ci="--ci" in sys.argv))
