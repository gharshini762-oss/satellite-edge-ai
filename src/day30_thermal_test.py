import os
import sys
import csv
import time
import datetime
import argparse
import numpy as np
import torch
import pynvml
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.models import build_resnet18
from src.verify_weights import verify_phase4_weights


def read_nvml(handle):
    temp = int(pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU))
    power_w = float(pynvml.nvmlDeviceGetPowerUsage(handle) / 1000.0)
    sm_mhz = int(pynvml.nvmlDeviceGetClockInfo(handle, pynvml.NVML_CLOCK_SM))
    try:
        throttle_mask = int(pynvml.nvmlDeviceGetCurrentClocksThrottleReasons(handle))
    except Exception:
        throttle_mask = -1
    return temp, power_w, sm_mhz, throttle_mask


def run_sustained(ckpt_path, prec, batch, duration_s, window_s, handle, power_mode):
    dtype = torch.float16 if prec == "fp16" else torch.float32
    torch.cuda.empty_cache()

    model = build_resnet18()
    model.load_state_dict(torch.load(ckpt_path, map_location="cpu"))
    if prec == "fp16":
        model = model.half()
    model = model.cuda().eval()
    x = torch.randn(batch, 3, 224, 224, device="cuda", dtype=dtype)

    # Short 20-iter warm-up before clock starts
    with torch.no_grad():
        for _ in range(20):
            _ = model(x)
        torch.cuda.synchronize()

    s_ev = torch.cuda.Event(enable_timing=True)
    e_ev = torch.cuda.Event(enable_timing=True)

    rows = []
    t_run_start = time.perf_counter()
    win_idx = 0

    print(f"\nStarting {prec.upper()} sustained load (batch={batch}, {duration_s}s)...")
    with torch.no_grad():
        while True:
            elapsed_total = time.perf_counter() - t_run_start
            if elapsed_total >= duration_s:
                break

            t_win_start = time.perf_counter()
            lat_ms = []
            while (time.perf_counter() - t_win_start) < window_s:
                s_ev.record()
                _ = model(x)
                e_ev.record()
                torch.cuda.synchronize()
                lat_ms.append(s_ev.elapsed_time(e_ev))

            win_elapsed = time.perf_counter() - t_win_start
            t_mid = (t_win_start - t_run_start) + win_elapsed / 2.0
            temp, power_w, sm_mhz, throttle_mask = read_nvml(handle)
            arr = np.array(lat_ms, dtype=np.float64)
            med_ms = float(np.median(arr))
            p95_ms = float(np.percentile(arr, 95))
            pps = (len(lat_ms) * batch) / win_elapsed

            row = {
                "date": datetime.date.today().isoformat(),
                "precision": prec,
                "batch": batch,
                "window_idx": win_idx,
                "elapsed_s": round(t_mid, 2),
                "n_iters": len(lat_ms),
                "median_ms": round(med_ms, 4),
                "p95_ms": round(p95_ms, 4),
                "throughput_pps": round(pps, 2),
                "temp_c": temp,
                "power_w": round(power_w, 2),
                "sm_clock_mhz": sm_mhz,
                "throttle_mask": throttle_mask,
                "tf32": False,
                "plugged_in": True,
                "power_mode": power_mode,
            }
            rows.append(row)
            if win_idx % 12 == 0:
                print(
                    f"  t={t_mid:5.1f}s | med={med_ms:6.3f} ms | {pps:7.1f} p/s | "
                    f"T={temp:2d}C | {sm_mhz:4d} MHz | {power_w:5.1f} W"
                )
            win_idx += 1

    del model, x
    torch.cuda.empty_cache()
    return rows


def plot_thermal(rows, out_png):
    fig, axes = plt.subplots(3, 1, figsize=(9.5, 8.2), sharex=True)
    colors = {"fp32": "#1f77b4", "fp16": "#ff7f0e"}

    for prec in ["fp32", "fp16"]:
        sub = [r for r in rows if r["precision"] == prec]
        t_min = np.array([r["elapsed_s"] / 60.0 for r in sub])
        temp = np.array([r["temp_c"] for r in sub])
        clk = np.array([r["sm_clock_mhz"] for r in sub])
        med = np.array([r["median_ms"] for r in sub])
        lbl = prec.upper()

        axes[0].plot(t_min, temp, color=colors[prec], lw=1.8, label=lbl)
        axes[1].plot(t_min, clk, color=colors[prec], lw=1.8, label=lbl)
        axes[2].plot(t_min, med, color=colors[prec], lw=1.8, label=f"{lbl} median")

    axes[0].set_ylabel("GPU Temp (°C)")
    axes[0].set_title("10-Minute Sustained GPU Inference (RTX 5060, Batch=32, TF32=False, Balanced)")
    axes[0].grid(alpha=0.3)
    axes[0].legend(loc="lower right")

    axes[1].set_ylabel("SM Clock (MHz)")
    axes[1].grid(alpha=0.3)
    axes[1].legend(loc="lower right")

    axes[2].set_ylabel("Batch-32 Latency (ms)")
    axes[2].set_xlabel("Elapsed Time (minutes)")
    axes[2].grid(alpha=0.3)
    axes[2].legend(loc="upper right")

    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    plt.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--duration", type=int, default=600, help="Seconds per precision (default 600 = 10 min)")
    ap.add_argument("--window", type=float, default=5.0, help="Logging window in seconds")
    ap.add_argument("--cooldown", type=int, default=180, help="Cooldown seconds between FP32 and FP16")
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--power-mode", type=str, default="Balanced")
    args = ap.parse_args()

    verify_phase4_weights(["results/ckpt_seed0.pt"])
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    assert torch.cuda.is_available()

    pynvml.nvmlInit()
    handle = pynvml.nvmlDeviceGetHandleByIndex(0)
    t0_temp, t0_pow, t0_clk, _ = read_nvml(handle)
    print(f"Pre-test idle state: T={t0_temp}C, P={t0_pow:.2f}W, SM={t0_clk} MHz")

    all_rows = []
    all_rows.extend(run_sustained("results/ckpt_seed0.pt", "fp32", args.batch, args.duration, args.window, handle, args.power_mode))
    print(f"\nCooling down for {args.cooldown}s before FP16 run...")
    time.sleep(args.cooldown)
    all_rows.extend(run_sustained("results/ckpt_seed0.pt", "fp16", args.batch, args.duration, args.window, handle, args.power_mode))
    pynvml.nvmlShutdown()

    out_csv = "results/day30_thermal_10min.csv"
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(all_rows[0].keys()))
        w.writeheader()
        w.writerows(all_rows)
    print(f"\nSaved {out_csv}")

    out_png = "results/day30_thermal_drift.png"
    plot_thermal(all_rows, out_png)
    print(f"Saved {out_png}")

    print("\n=== 5B 10-Minute Thermal Drift Summary (First 60s vs Last 60s) ===")
    for prec in ["fp32", "fp16"]:
        sub = [r for r in all_rows if r["precision"] == prec]
        early = [r for r in sub if r["elapsed_s"] <= 60.0]
        late = [r for r in sub if r["elapsed_s"] >= (args.duration - 60.0)]
        e_lat = float(np.mean([r["median_ms"] for r in early]))
        l_lat = float(np.mean([r["median_ms"] for r in late]))
        drift_pct = (l_lat - e_lat) / e_lat * 100.0
        temps = [r["temp_c"] for r in sub]
        clks = [r["sm_clock_mhz"] for r in sub]
        pows = [r["power_w"] for r in sub]
        print(
            f"{prec.upper():4s}: lat {e_lat:.4f} -> {l_lat:.4f} ms ({drift_pct:+.2f}%) | "
            f"T {temps[0]}->{temps[-1]}C (max {max(temps)}C) | "
            f"SM {clks[0]}->{clks[-1]} MHz (min {min(clks)}) | mean P={np.mean(pows):.2f} W"
        )


if __name__ == "__main__":
    main()