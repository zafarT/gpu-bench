# gpu-bench

A small GPU benchmark harness that **catches performance regressions automatically**.

It runs two benchmarks, stores every result in SQLite, compares each new result
with the history of the same machine and configuration, and exits with code 1
(CI turns red) when performance dropped by more than normal run-to-run noise.

| Benchmark | Measures | Runs on |
|-----------|----------|---------|
| `matmul`  | PyTorch 2048×2048 float32 matrix multiply, GFLOP/s (median of 20 runs after 3 warm-ups) | Linux + Windows, GPU (CUDA) or CPU |
| `glmark2` | OpenGL rendering score (median of 3 runs) | Linux (no official Windows build, skipped there) |

## How it works

```
python -m bench
   │
   ├─ matmul.py / glmark2.py   run each benchmark several times → median + std
   ├─ system.py                OS, GPU name, NVIDIA driver version, git commit
   ├─ db.py                    save to SQLite, load the last 10 comparable results
   ├─ regression.py            z-score + percentage check
   └─ exit code 1 on regression → GitHub Actions job fails
```

**Regression rule.** From the last 10 results with the same OS, benchmark and
configuration, take the mean μ and standard deviation σ. The new value is a
regression only if **both** are true:

* `z = (new − μ) / σ ≤ −3`: the drop is unusual compared with normal noise, and
* `Δ = (new − μ) / μ ≤ −5 %` (−10 % in CI): the drop is big enough to matter.

Needing both avoids false alarms. On a very stable machine a 1 % blip can be
"20 σ", and on a noisy machine a 15 % dip can be normal. With fewer than 3
previous results nothing is flagged.

**Like-for-like only.** A CPU run is never compared with a GPU run, and an Intel
iGPU glmark2 score never with an NVIDIA one. The `config` column (matrix size,
dtype and device, or the OpenGL renderer) has to match.

**Driver tracking.** Every row stores the NVIDIA driver version. The driver is
not part of the comparison on purpose, because "is the new driver slower?" is
exactly what should be caught. When it changed, the report says so.

## Run it

```bash
python -m venv .venv && source .venv/bin/activate    # Windows: .venv\Scripts\activate
pip install -r requirements.txt
sudo apt install glmark2 glmark2-wayland              # optional, Linux

python -m bench                   # run all, save to results.db, print report
python -m bench --only matmul
python -m bench --slowdown 0.7    # demo: fake a 30% slowdown -> REGRESSION, exit 1
pytest
```

Example output (RTX 4050 laptop):

```
TODO: paste a real report from your own GPU here
```

## CI (GitHub Actions)

[`.github/workflows/bench.yml`](.github/workflows/bench.yml) runs on every push,
on Ubuntu and Windows: install → `pytest` → `python -m bench`.

* CI machines are deleted after each job, so `results.db` is carried between
  runs with `actions/cache`. A job that found a regression fails, so its
  database is not saved and the bad result never becomes part of the baseline.
* To demo a regression: **Actions → bench → Run workflow**, enter `0.7`.

**Limitation.** GitHub's free runners have **no GPU**. There, `matmul` runs on
the CPU and `glmark2` uses Mesa software rendering (llvmpipe) on a virtual
screen. CI therefore tests the *pipeline* (storage, statistics, exit codes),
not real GPU performance. Real GPU numbers come from running it on a GPU
machine. The next step would be a self-hosted runner on that machine.

## Files

```
bench/
  __main__.py    run benchmarks, print report, exit code
  matmul.py      PyTorch matmul benchmark
  glmark2.py     glmark2 wrapper and output parser
  system.py      OS, GPU, driver, git commit
  db.py          SQLite table, save, history
  regression.py  z-score + percentage rule
tests/           17 pytest tests (no GPU needed)
```
