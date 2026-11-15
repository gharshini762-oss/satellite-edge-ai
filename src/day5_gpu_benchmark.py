import time
import numpy as np
import torch

if __name__ == "__main__":
    N = 4096
    ITERS = 100
    WARMUP = 10

    a_cpu = torch.randn(N, N, dtype=torch.float32)
    b_cpu = torch.randn(N, N, dtype=torch.float32)

    t0 = time.perf_counter()
    _ = torch.matmul(a_cpu, b_cpu)
    cpu_ms = (time.perf_counter() - t0) * 1000.0
    print(f"CPU 4096x4096 matmul (1 run): {cpu_ms:.2f} ms")

    a_gpu = a_cpu.to("cuda")
    b_gpu = b_cpu.to("cuda")

    for _ in range(WARMUP):
        _ = torch.matmul(a_gpu, b_gpu)
    torch.cuda.synchronize()

    unsync_times = []
    for _ in range(ITERS):
        t0 = time.perf_counter()
        _ = torch.matmul(a_gpu, b_gpu)
        unsync_times.append((time.perf_counter() - t0) * 1000.0)
    torch.cuda.synchronize()

    sync_times = []
    for _ in range(ITERS):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        _ = torch.matmul(a_gpu, b_gpu)
        torch.cuda.synchronize()
        sync_times.append((time.perf_counter() - t0) * 1000.0)

    print(f"GPU Unsynchronized ({ITERS} runs) -> median: {np.median(unsync_times):.4f} ms | p95: {np.percentile(unsync_times, 95):.4f} ms")
    print(f"GPU Synchronized   ({ITERS} runs) -> median: {np.median(sync_times):.2f} ms | p95: {np.percentile(sync_times, 95):.2f} ms")