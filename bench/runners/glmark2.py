import os, re, shutil, platform, statistics, subprocess

def _nvidia_present():
    return shutil.which("nvidia-smi") is not None

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
        m = re.search(r"glmark2 Score:\s*(\d+)", out)
        if r.returncode != 0 or not m:
            return {"benchmark": "glmark2", "status": "error",
                    "reason": f"exit code {r.returncode}",
                    "log_tail": out[-1000:]}      # last lines, for debugging
        scores.append(int(m.group(1)))

        rm = re.search(r"GL_RENDERER:\s*(.+)", out)
        if rm:
            renderer = rm.group(1).strip()

    return {
        "benchmark": "glmark2",
        "status": "ok",
        "renderer": renderer,                    # proves which GPU ran it
        "score_median": statistics.median(scores),
        "score_std": statistics.stdev(scores) if len(scores) > 1 else 0.0,
        "samples": scores,
    }

if __name__ == "__main__":
    import sys
    print(run_glmark2(ci="--ci" in sys.argv))