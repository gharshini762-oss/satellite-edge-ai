import csv
import os
import time
import matplotlib.pyplot as plt
import pynvml
import torch


os.makedirs("results", exist_ok=True)

# Initialize NVML
pynvml.nvmlInit()
handle = pynvml.nvmlDeviceGetHandleByIndex(0)  # RTX 5060
gpu_name = pynvml.nvmlDeviceGetName(handle)
if isinstance(gpu_name, bytes):
  gpu_name = gpu_name.decode("utf-8")

print(f"=== DAY 6: GPU TELEMETRY MONITOR ===")
print(f"Monitoring Device: {gpu_name}\n")


csv_path = "results/telemetry_log.csv"
csv_file = open(csv_path, mode="w", newline="")
csv_writer = csv.writer(csv_file)
csv_writer.writerow(
    ["Elapsed_sec", "Temp_C", "Power_W", "Clock_MHz", "VRAM_Used_MB"]
)


print("[1] Initializing stress test workload on GPU...")
torch.cuda.reset_peak_memory_stats()
stress_tensors = [
    torch.randn(3000, 3000, device="cuda") for _ in range(15)
]  

duration_sec = 10
sample_rate_hz = 5  
interval = 1.0 / sample_rate_hz

print(
    f"[2] Running GPU stress test for {duration_sec} seconds at"
    f" {sample_rate_hz} Hz..."
)
start_time = time.time()
next_sample_time = start_time

try:
  while True:
    now = time.time()
    elapsed = now - start_time
    if elapsed > duration_sec:
      break

    
    for t in stress_tensors:
      _ = torch.matmul(t, t)
    torch.cuda.synchronize()

    # Query NVML hardware sensors
    temp = pynvml.nvmlDeviceGetTemperature(
        handle, pynvml.NVML_TEMPERATURE_GPU
    )  # Celsius
    power = (
        pynvml.nvmlDeviceGetPowerUsage(handle) / 1000.0
    )  # Milliwatts to Watts
    clock = pynvml.nvmlDeviceGetClockInfo(
        handle, pynvml.NVML_CLOCK_GRAPHICS
    )  # MHz
    mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
    vram_used_mb = mem_info.used / (1024 * 1024)

    
    csv_writer.writerow(
        [
            round(elapsed, 2),
            temp,
            round(power, 2),
            clock,
            round(vram_used_mb, 2),
        ]
    )

    
    next_sample_time += interval
    sleep_time = next_sample_time - time.time()
    if sleep_time > 0:
      time.sleep(sleep_time)

finally:
  csv_file.close()
  pynvml.nvmlShutdown()
  print(f"[3] Telemetry log saved successfully to {csv_path}")


peak_vram = torch.cuda.max_memory_allocated() / (1024 * 1024)
print(f"\n[4] Peak VRAM Allocated during workload: {peak_vram:.2f} MB")

# Plotting Temperature and Clock vs Time
print("[5] Generating telemetry performance plot...")
times, temps, powers, clocks, vrams = [], [], [], [], []
with open(csv_path, mode="r") as f:
  reader = csv.reader(f)
  next(reader)  # Skip header
  for row in reader:
    times.append(float(row[0]))
    temps.append(float(row[1]))
    powers.append(float(row[2]))
    clocks.append(float(row[3]))
    vrams.append(float(row[4]))

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6), sharex=True)

ax1.plot(times, temps, color="tab:red", marker="o", label="Temperature (°C)")
ax1.set_ylabel("Temp (°C)", color="tab:red")
ax1.grid(True)
ax1.set_title(f"RTX 5060 Telemetry Stress Test ({gpu_name})")

ax2.plot(
    times, clocks, color="tab:blue", marker="x", label="Core Clock (MHz)"
)
ax2.set_ylabel("Clock (MHz)", color="tab:blue")
ax2.set_xlabel("Elapsed Time (seconds)")
ax2.grid(True)

plt.tight_layout()
plot_path = "results/telemetry_plot.png"
plt.savefig(plot_path, dpi=150)
print(f"Saved telemetry plot to {plot_path}")
plt.show()