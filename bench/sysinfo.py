import os
import platform
import subprocess
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent


def is_ci():
    """True on CI runners (GitHub Actions, GitLab, ... all set CI=true)."""
    return os.environ.get("CI", "").strip().lower() in ("1", "true", "yes")


def _git_sha():
    try:
        r = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=REPO_ROOT,
                           capture_output=True, text=True, timeout=5)
        return r.stdout.strip() or None
    except (OSError, subprocess.SubprocessError):
        return None


def collect():
    try:
        import torch
        torch_version = torch.__version__
        cuda = torch.cuda.is_available()
        gpu_name = torch.cuda.get_device_name(0) if cuda else None
    except ImportError:
        torch_version, cuda, gpu_name = None, False, None

    return {
        "ts": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "git_sha": _git_sha(),
        "os": platform.system(),
        "device": "cuda" if cuda else "cpu",
        "gpu_name": gpu_name or platform.processor() or None,
        "torch_version": torch_version,
    }


if __name__ == "__main__":
    print(collect())
