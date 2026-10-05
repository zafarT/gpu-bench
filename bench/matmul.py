"""PyTorch matrix-multiplication benchmark (GPU if available, else CPU)."""
import statistics
import time

import torch


def run_matmul(n=2048, repeats=20, warmup=3):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    a = torch.randn(n, n, device=device)
    b = torch.randn(n, n, device=device)

    # Warm-up: the first runs are slow (GPU clocks ramp up, kernels get loaded).
    for _ in range(warmup):
        a @ b
    if device == "cuda":
        torch.cuda.synchronize()

    gflops = []
    for _ in range(repeats):
        if device == "cuda":
            # GPU work is asynchronous, so time it with GPU events, not the CPU clock.
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            a @ b
            end.record()
            end.synchronize()
            seconds = start.elapsed_time(end) / 1000      # ms -> s
        else:
            t0 = time.perf_counter()
            a @ b
            seconds = time.perf_counter() - t0
        gflops.append(2 * n**3 / seconds / 1e9)           # n x n matmul = 2n^3 operations

    return {
        "benchmark": "matmul",
        "config": f"n={n} float32 {device}",   # only identical configs are compared
        "status": "ok",
        "value": statistics.median(gflops),    # median: robust against one slow outlier
        "std": statistics.stdev(gflops),
        "unit": "GFLOP/s",
    }


if __name__ == "__main__":
    print(run_matmul())
