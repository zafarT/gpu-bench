import subprocess

from bench.runners import glmark2
from bench.runners.glmark2 import parse_output

FAKE_OUTPUT = """\
=======================================================
    glmark2 2023.01
=======================================================
    OpenGL Information
    GL_VENDOR:      NVIDIA Corporation
    GL_RENDERER:    NVIDIA GeForce RTX 3060 Laptop GPU/PCIe/SSE2
    GL_VERSION:     4.6.0 NVIDIA 560.35.03
=======================================================
[build] use-vbo=true: FPS: 9876 FrameTime: 0.101 ms
[texture] <default>: FPS: 8765 FrameTime: 0.114 ms
[shading] <default>: FPS: 7654 FrameTime: 0.131 ms
=======================================================
                                  glmark2 Score: 8765
=======================================================
"""


def test_parse_score_and_renderer():
    score, renderer = parse_output(FAKE_OUTPUT)
    assert score == 8765
    assert renderer == "NVIDIA GeForce RTX 3060 Laptop GPU/PCIe/SSE2"


def test_parse_missing_score():
    score, renderer = parse_output("Error: main: Could not initialize canvas\n")
    assert score is None
    assert renderer is None


def test_parse_software_renderer():
    out = "GL_RENDERER: llvmpipe (LLVM 17.0.6, 256 bits)\n glmark2 Score: 312 \n"
    assert parse_output(out) == (312, "llvmpipe (LLVM 17.0.6, 256 bits)")


def _fake_glmark2(monkeypatch, outputs, returncode=0):
    """Pretend glmark2 is installed and returns `outputs` one call at a time."""
    it = iter(outputs)
    monkeypatch.setattr(glmark2.shutil, "which",
                        lambda name: "/usr/bin/" + name if name == "glmark2" else None)
    monkeypatch.setattr(glmark2.subprocess, "run",
                        lambda cmd, **kw: subprocess.CompletedProcess(
                            cmd, returncode, stdout=next(it), stderr=""))


def test_run_glmark2_aggregates_samples(monkeypatch):
    outs = [FAKE_OUTPUT.replace("Score: 8765", f"Score: {s}") for s in (100, 110, 90)]
    _fake_glmark2(monkeypatch, outs)
    r = glmark2.run_glmark2(repeats=3)
    assert r["status"] == "ok"
    assert r["samples"] == [100, 110, 90]
    assert r["score_median"] == 100
    assert r["score_std"] == 10.0
    assert abs(r["cv"] - 0.1) < 1e-12          # std / mean = 10 / 100
    assert r["renderer"].startswith("NVIDIA GeForce")


def test_run_glmark2_error_on_unparseable_output(monkeypatch):
    _fake_glmark2(monkeypatch, ["garbage"], returncode=1)
    r = glmark2.run_glmark2(repeats=1)
    assert r["status"] == "error"
    assert r["log_tail"] == "garbage"


def test_run_glmark2_skipped_when_not_installed(monkeypatch):
    monkeypatch.setattr(glmark2.shutil, "which", lambda name: None)
    assert glmark2.run_glmark2()["status"] == "skipped"
