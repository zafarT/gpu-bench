"""Facts about the machine, stored with every result."""
import platform
import shutil
import subprocess


def _run(cmd):
    """Run a command and return its first output line, or None if it fails."""
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
    except (OSError, subprocess.SubprocessError):
        return None
    lines = r.stdout.strip().splitlines()
    return lines[0].strip() if r.returncode == 0 and lines else None


def nvidia_driver():
    """NVIDIA driver version, or None. nvidia-smi can be installed while the
    driver itself is not loaded; then it fails and we correctly return None."""
    if shutil.which("nvidia-smi") is None:
        return None
    return _run(["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"])


def info():
    import torch
    cuda = torch.cuda.is_available()
    return {
        "os": platform.system(),
        "gpu": torch.cuda.get_device_name(0) if cuda else None,
        "driver": nvidia_driver(),
        "git_sha": _run(["git", "rev-parse", "--short", "HEAD"]),
    }


if __name__ == "__main__":
    print(info())
