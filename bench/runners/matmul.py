import time, torch

def run_matmul(n=2048, repeats=10, warmup=3):
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    a, b = torch.randn(n, n, device=dev), torch.randn(n, n, device=dev)
    for _ in range(warmup):           # warm-up: GPU clocks, caches, kernel load
        a @ b
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    times = []
    for _ in range(repeats):
        if dev == "cuda":
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
    return {"device": dev, "n": n, "gflops": gflops}

if __name__ == "__main__":
    run = print(run_matmul())