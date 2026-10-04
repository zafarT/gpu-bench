import time, torch, statistics

def run_matmul(n=2048, repeats=20, warmup=3, dtype=torch.float32):
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    a = torch.randn(n, n, device=dev, dtype=dtype)
    b = torch.randn(n, n, device=dev, dtype=dtype)
    for _ in range(warmup):           # warm-up: GPU clocks, caches, kernel load
        a @ b
    times = []
    for _ in range(repeats):
        if dev == "cuda":
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            a @ b
            end.record()
            end.synchronize() # Wait for the GPU event to finish
            
            # elapsed_time returns milliseconds; convert to seconds
            times.append(start.elapsed_time(end) / 1000)
        else:
            # Fallback for CPU
            t0 = time.perf_counter()
            a @ b
            times.append(time.perf_counter() - t0)
            
    gflops = [2 * n**3 / t / 1e9 for t in times]
    median = statistics.median(gflops)
    std = statistics.stdev(gflops)
    return {
        "benchmark": "matmul",
        "device": dev,
        "dtype": str(dtype).replace("torch.", ""),
        "n": n,
        "gflops_median": median,
        "gflops_std": std,
        "cv": std / statistics.mean(gflops),   # noise level, e.g. 0.02 = 2%
        "samples": gflops,
    }

if __name__ == "__main__":
    run = print(run_matmul())