from bench import glmark2

OUTPUT = """
    GL_VENDOR:      NVIDIA Corporation
    GL_RENDERER:    NVIDIA GeForce RTX 4050 Laptop GPU/PCIe/SSE2
    GL_VERSION:     4.6.0 NVIDIA 595.84
[texture] <default>: FPS: 8765 FrameTime: 0.114 ms
                                  glmark2 Score: 8765
"""


def test_parse_score_and_renderer():
    assert glmark2.parse(OUTPUT) == (8765, "NVIDIA GeForce RTX 4050 Laptop GPU/PCIe/SSE2")


def test_parse_failed_run():
    assert glmark2.parse("X Error of failed request: BadValue") == (None, None)


def test_ci_uses_virtual_screen_and_software_rendering():
    cmd, env = glmark2.build_command(ci=True)
    assert cmd[:2] == ["xvfb-run", "-a"]
    assert env == {"LIBGL_ALWAYS_SOFTWARE": "1"}


def test_no_nvidia_offload_without_a_working_driver(monkeypatch):
    monkeypatch.setattr(glmark2, "nvidia_driver", lambda: None)
    assert glmark2.build_command(ci=False)[1] == {}


def test_skipped_when_not_installed(monkeypatch):
    monkeypatch.setattr(glmark2.shutil, "which", lambda name: None)
    assert glmark2.run_glmark2()["status"] == "skipped"
