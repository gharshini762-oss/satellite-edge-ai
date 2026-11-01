import statistics
import time
import torch

print("=== DAY 5: PYTORCH TENSORS & GPU BENCHMARK ===")
print(
    "GPU Available:",
    torch.cuda.is_available(),
    "| Device:",
    torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU",
)

# 1. Creating tensors and moving to GPU with .to("cuda")
size = 4096
print(f"\n[1] Creating random {size}x{size} float32 tensors...")
a_cpu = torch.randn(size, size)
b_cpu = torch.randn(size, size)

print("Moving tensors to RTX 5060 GPU with .to('cuda')...")
a_gpu = a_cpu.to("cuda")
b_gpu = b_cpu.to("cuda")


print("\n[2] Running CPU Matmul...")
start_cpu = time.time()
_ = torch.matmul(a_cpu, b_cpu)
cpu_time = time.time() - start_cpu
print(f"CPU time: {cpu_time:.4f} seconds")

print("\n[3] Warming up GPU kernels...")
for _ in range(10):
  _ = torch.matmul(a_gpu, b_gpu)
torch.cuda.synchronize()

# 2 & 3. Time 100 runs UNSYNCHRONIZED vs SYNCHRONIZED
print(
    "\n[4] Running 100 trials: Unsynchronized vs Synchronized (Median of 100"
    " runs)..."
)

# --- UNSYNCHRONIZED ---
unsync_times = []
for _ in range(100):
  start = time.time()
  _ = torch.matmul(a_gpu, b_gpu)  
  end = time.time()
  unsync_times.append(end - start)
unsync_median = statistics.median(unsync_times)

# --- SYNCHRONIZED ---
sync_times = []
for _ in range(100):
  start = time.time()
  _ = torch.matmul(a_gpu, b_gpu)
  torch.cuda.synchronize()  
  end = time.time()
  sync_times.append(end - start)
sync_median = statistics.median(sync_times)

print(
    "   -> Fake Unsynchronized Median:"
    f" {unsync_median * 1000:.4f} ms (Measures CPU handoff only!)"
)
print(
    "   -> Real Synchronized Median  :"
    f" {sync_median * 1000:.4f} ms (Measures actual GPU compute time)"
)
print(
    f"\nSpeedup (CPU vs Synchronized GPU): {cpu_time / sync_median:.1f}x"
    " faster!"
)