import os
import time
import subprocess
import threading
import pandas as pd
import numpy as np
import torch
import pynvml
import matplotlib.pyplot as plt

def run_gpu_load(stop_event):
    a = torch.randn(4096, 4096, device="cuda", dtype=torch.float32)
    b = torch.randn(4096, 4096, device="cuda", dtype=torch.float32)
    while not stop_event.is_set():
        _ = torch.matmul(a, b)
        torch.cuda.synchronize()

if __name__ == "__main__":
    DURATION_S = 20.0
    INTERVAL_S = 0.200  

    pynvml.nvmlInit()
    handle = pynvml.nvmlDeviceGetHandleByIndex(0)

    smi_csv = "results/day19_nvidiasmi.csv"
    pynvml_csv = "results/day19_pynvml.csv"

    smi_cmd = [
        "nvidia-smi",
        "--query-gpu=timestamp,temperature.gpu,power.draw,clocks.sm",
        "--format=csv,noheader,nounits",
        "-lms", "200"
    ]
    smi_file = open(smi_csv, "w")
    smi_file.write("timestamp,temp_c,power_w,clock_mhz\n")
    smi_file.flush()
    smi_proc = subprocess.Popen(smi_cmd, stdout=smi_file, stderr=subprocess.DEVNULL, text=True)

    stop_event = threading.Event()
    worker = threading.Thread(target=run_gpu_load, args=(stop_event,))
    worker.start()

    pynvml_rows = []
    t_start = time.perf_counter()
    next_tick = t_start
    while True:
        now = time.perf_counter()
        elapsed = now - t_start
        if elapsed > DURATION_S:
            break
        temp = pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)
        power = pynvml.nvmlDeviceGetPowerUsage(handle) / 1000.0
        clock = pynvml.nvmlDeviceGetClockInfo(handle, pynvml.NVML_CLOCK_SM)
        vram_used = pynvml.nvmlDeviceGetMemoryInfo(handle).used / (1024**2)
        pynvml_rows.append(dict(t_sec=round(elapsed, 4), temp_c=temp, power_w=round(power, 2), clock_mhz=clock, vram_mib=round(vram_used, 1)))

        next_tick += INTERVAL_S
        sleep_time = next_tick - time.perf_counter()
        if sleep_time > 0:
            time.sleep(sleep_time)

    stop_event.set()
    worker.join()
    smi_proc.terminate()
    smi_proc.wait()
    smi_file.close()
    pynvml.nvmlShutdown()

    df_py = pd.DataFrame(pynvml_rows)
    df_py.to_csv(pynvml_csv, index=False)

    df_smi = pd.read_csv(smi_csv, skipinitialspace=True)
    df_smi["t_ parsed"] = pd.to_datetime(df_smi["timestamp"], format="%Y/%m/%d %H:%M:%S.%f")
    df_smi["t_sec"] = (df_smi["t_ parsed"] - df_smi["t_ parsed"].iloc[0]).dt.total_seconds()
    df_smi = df_smi[df_smi["t_sec"] <= DURATION_S + 0.5].reset_index(drop=True)

    dt_py = np.diff(df_py["t_sec"].to_numpy())
    dt_smi = np.diff(df_smi["t_sec"].to_numpy())

    print("=== Day 19: 20-Second GPU Load Telemetry Comparison ===")
    print(f"pynvml  (perf_counter) -> samples: {len(df_py):3d} | dt median: {np.median(dt_py):.4f}s (min: {dt_py.min():.4f}s, max: {dt_py.max():.4f}s) | temp: {df_py['temp_c'].min()}C -> {df_py['temp_c'].max()}C | power max: {df_py['power_w'].max():.2f}W")
    print(f"nvidia-smi (-lms 200)  -> samples: {len(df_smi):3d} | dt median: {np.median(dt_smi):.4f}s (min: {dt_smi.min():.4f}s, max: {dt_smi.max():.4f}s) | temp: {df_smi['temp_c'].min()}C -> {df_smi['temp_c'].max()}C | power max: {df_smi['power_w'].max():.2f}W")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))
    ax1.plot(df_py["t_sec"], df_py["temp_c"], marker=".", label="pynvml (perf_counter)")
    ax1.plot(df_smi["t_sec"], df_smi["temp_c"], marker="x", alpha=0.8, label="nvidia-smi -lms 200")
    ax1.set_xlabel("Time (s)")
    ax1.set_ylabel("GPU Temp (C)")
    ax1.set_title("20s Load: Temperature Curve")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2.plot(df_py["t_sec"], df_py["power_w"], marker=".", label="pynvml (perf_counter)")
    ax2.plot(df_smi["t_sec"], df_smi["power_w"], marker="x", alpha=0.8, label="nvidia-smi -lms 200")
    ax2.set_xlabel("Time (s)")
    ax2.set_ylabel("Power Draw (W)")
    ax2.set_title("20s Load: Power Draw")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("results/day19_telemetry_compare.png", dpi=150)
    plt.close()
    print("Saved comparison plot: results/day19_telemetry_compare.png")