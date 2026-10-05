"""glmark2 OpenGL benchmark (Linux only; there is no official Windows build)."""
import os
import re
import shutil
import statistics
import subprocess

from bench.system import nvidia_driver

SCENES = ["build", "texture", "shading"]
NVIDIA_EGL = "/usr/share/glvnd/egl_vendor.d/10_nvidia.json"


def parse(output):
    """Return (score, renderer) from glmark2's output, None if missing."""
    score = re.search(r"glmark2 Score:\s*(\d+)", output)
    renderer = re.search(r"GL_RENDERER:\s*(.+)", output)
    return (int(score.group(1)) if score else None,
            renderer.group(1).strip() if renderer else None)


def build_command(ci):
    """Return (command, extra environment variables)."""
    if ci:
        # GitHub runner: no screen and no GPU -> virtual X server + software rendering
        return ["xvfb-run", "-a", "glmark2"], {"LIBGL_ALWAYS_SOFTWARE": "1"}

    # On a Wayland desktop the X11 build can fail to create a GL context.
    use_wayland = os.environ.get("WAYLAND_DISPLAY") and shutil.which("glmark2-wayland")
    cmd = ["glmark2-wayland" if use_wayland else "glmark2"]

    env = {}
    if nvidia_driver():
        # Hybrid laptop: render on the NVIDIA GPU instead of the integrated one.
        env["__NV_PRIME_RENDER_OFFLOAD"] = "1"
        env["__GLX_VENDOR_LIBRARY_NAME"] = "nvidia"                 # X11 build (GLX)
        if os.path.exists(NVIDIA_EGL):
            env["__EGL_VENDOR_LIBRARY_FILENAMES"] = NVIDIA_EGL      # Wayland build (EGL)
    return cmd, env


def run_glmark2(ci=False, repeats=3, timeout=600):
    base, extra_env = build_command(ci)
    if shutil.which(base[-1]) is None:
        return {"benchmark": "glmark2", "status": "skipped",
                "note": f"{base[-1]} not installed"}

    cmd = base + ["--off-screen"]
    for scene in SCENES:
        cmd += ["-b", scene]
    env = {**os.environ, **extra_env}

    scores, renderer = [], None
    for _ in range(repeats):
        try:
            r = subprocess.run(cmd, env=env, capture_output=True, text=True, timeout=timeout)
        except subprocess.TimeoutExpired:
            return {"benchmark": "glmark2", "status": "error",
                    "note": f"timeout after {timeout}s"}
        output = r.stdout + r.stderr
        score, renderer = parse(output)
        if r.returncode != 0 or score is None:
            return {"benchmark": "glmark2", "status": "error",
                    "note": output.strip()[-500:]}     # keep the end of the log for debugging
        scores.append(score)

    return {
        "benchmark": "glmark2",
        "config": renderer,          # which GPU actually rendered (NVIDIA / Intel / llvmpipe)
        "status": "ok",
        "value": statistics.median(scores),
        "std": statistics.stdev(scores) if len(scores) > 1 else 0.0,
        "unit": "score",
    }


if __name__ == "__main__":
    print(run_glmark2())
