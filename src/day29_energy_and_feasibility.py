import os
import sys
import csv
import time
import datetime
import threading
import argparse
import numpy as np
import torch
import pynvml

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.models import build_resnet18
from src.verify_weights import verify_phase4_weights


class PowerPoller:
    def __init__(self, handle, interval_s=0.1):
        self.handle = handle
        self.interval_s = interval_s
        self.samples_w = []
        self.temps_c = []
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def _run(self):
        while not self._stop.is_set():
            t0 = time.perf_counter()
            self.samples_w.append(pynvml.nvmlDeviceGetPowerUsage(self.handle) / 1000.0)
            self.temps_c.append(
                pynvml.nvmlDeviceGetTemperature(self.handle, pynvml.NVML_TEMPERATURE_GPU)
            )
            sleep_for = self.interval_s - (time.perf_counter() - t0)
            if sleep_for > 0:
                time.sleep(sleep_for)

    def start(self):
        self._thread.start()

    def stop(self):
        self._stop.set()
        self._thread.join()
        return np.array(self.samples_w, dtype=np.float64), np.array(self.temps_c, dtype=np.float64)


def measure_idle(handle, seconds=15.0):
    print(f"Measuring GPU idle power for {seconds:.0f}s (keep laptop untouched)...")
    poller = PowerPoller(handle, interval_s=0.1)
    poller.start()
    time.sleep(seconds)
    w, tc = poller.stop()
    return float(w.mean()), float(w.std(ddof=1)), float(tc.mean())


def run_energy_window(ckpt_path, prec, batch, handle, window_s=12.0, warmup_s=2.0):
    dtype = torch.float16 if prec == "fp16" else torch.float32
    torch.cuda.empty_cache()

    model = build_resnet18()
    model.load_state_dict(torch.load(ckpt_path, map_location="cpu"))
    if prec == "fp16":
        model = model.half()
    model = model.cuda().eval()
    x = torch.randn(batch, 3, 224, 224, device="cuda", dtype=dtype)

    # Warm-up
    t_warm = time.perf_counter()
    with torch.no_grad():
        while time.perf_counter() - t_warm < warmup_s:
            _ = model(x)
        torch.cuda.synchronize()

    # Timed energy & throughput window
    poller = PowerPoller(handle, interval_s=0.1)
    poller.start()
    n_batches = 0
    t0 = time.perf_counter()
    with torch.no_grad():
        while time.perf_counter() - t0 < window_s:
            _ = model(x)
            n_batches += 1
        torch.cuda.synchronize()
    elapsed_s = time.perf_counter() - t0
    w, tc = poller.stop()

    del model, x
    torch.cuda.empty_cache()

    patches = n_batches * batch
    throughput = patches / elapsed_s
    return {
        "elapsed_s": elapsed_s,
        "patches": patches,
        "throughput_pps": throughput,
        "mean_power_w": float(w.mean()),
        "std_power_w": float(w.std(ddof=1)),
        "n_power_samples": int(len(w)),
        "start_temp_c": int(tc[0]),
        "end_temp_c": int(tc[-1]),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--cooldown", type=int, default=30)
    ap.add_argument("--window", type=float, default=12.0)
    ap.add_argument("--power-mode", type=str, default="Balanced")
    args = ap.parse_args()

    verify_phase4_weights(["results/ckpt_seed0.pt"])

    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    assert torch.cuda.is_available()

    pynvml.nvmlInit()
    handle = pynvml.nvmlDeviceGetHandleByIndex(0)

    idle_w, idle_std, idle_temp = measure_idle(handle, seconds=15.0)
    print(f"Idle baseline: {idle_w:.2f} ± {idle_std:.2f} W (temp {idle_temp:.1f} C)")

    ckpt = "results/ckpt_seed0.pt"
    rows = []
    for batch in [1, 8, 32]:
        for rep in range(1, args.repeats + 1):
            for prec in ["fp32", "fp16"]:
                print(f"Cooldown {args.cooldown}s before {prec} batch={batch} rep={rep}...")
                time.sleep(args.cooldown)
                res = run_energy_window(ckpt, prec, batch, handle, window_s=args.window)
                total_mj = (res["mean_power_w"] * res["elapsed_s"] / res["patches"]) * 1000.0
                net_mj = (max(res["mean_power_w"] - idle_w, 0.0) * res["elapsed_s"] / res["patches"]) * 1000.0
                row = {
                    "date": datetime.date.today().isoformat(),
                    "precision": prec,
                    "batch": batch,
                    "repeat": rep,
                    "throughput_pps": round(res["throughput_pps"], 2),
                    "mean_power_w": round(res["mean_power_w"], 2),
                    "idle_power_w": round(idle_w, 2),
                    "total_energy_mj_per_patch": round(total_mj, 3),
                    "net_energy_mj_per_patch": round(net_mj, 3),
                    "patches": res["patches"],
                    "elapsed_s": round(res["elapsed_s"], 3),
                    "n_power_samples": res["n_power_samples"],
                    "start_temp_c": res["start_temp_c"],
                    "end_temp_c": res["end_temp_c"],
                    "tf32": False,
                    "plugged_in": True,
                    "power_mode": args.power_mode,
                }
                rows.append(row)
                print(
                    f"  {prec:4s} b={batch:2d} r={rep}: {row['throughput_pps']:7.1f} p/s | "
                    f"P={row['mean_power_w']:5.2f}W | total={row['total_energy_mj_per_patch']:6.2f} mJ/p | "
                    f"net={row['net_energy_mj_per_patch']:6.2f} mJ/p | T={row['start_temp_c']}->{row['end_temp_c']}C"
                )

    pynvml.nvmlShutdown()

    out_path = "results/day29_energy_and_throughput.csv"
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"\nSaved {out_path}")

    # 5A Summary (ddof=1) & 5C Real-Time Feasibility
    swath_km, speed_kms = 290.0, 6.7
    area_rate_km2s = swath_km * speed_kms
    patch_224_km2 = (224 * 0.01) ** 2
    patch_509_km2 = (509 * 0.01) ** 2
    req_224_pps = area_rate_km2s / patch_224_km2
    req_509_pps = area_rate_km2s / patch_509_km2

    print("\n=== 5A Energy & Throughput Summary (3 Repeats, ddof=1) ===")
    for prec in ["fp32", "fp16"]:
        for batch in [1, 8, 32]:
            sub = [r for r in rows if r["precision"] == prec and r["batch"] == batch]
            pps = [r["throughput_pps"] for r in sub]
            tot = [r["total_energy_mj_per_patch"] for r in sub]
            net = [r["net_energy_mj_per_patch"] for r in sub]
            print(
                f"{prec.upper():4s} b={batch:2d}: {np.mean(pps):7.1f} ± {np.std(pps, ddof=1):5.1f} p/s | "
                f"total {np.mean(tot):6.2f} ± {np.std(tot, ddof=1):4.2f} mJ/p | "
                f"net {np.mean(net):6.2f} ± {np.std(net, ddof=1):4.2f} mJ/p"
            )

    print("\n=== 5C Real-Time Feasibility (Sentinel-2 Swath Math) ===")
    print(f"Swath coverage rate: {swath_km:.0f} km x {speed_kms:.1f} km/s = {area_rate_km2s:.1f} km^2/s")
    print(f"Native 224x224 @ 10m ({patch_224_km2:.2f} km^2/patch): needs ~{req_224_pps:.0f} patches/s")
    print(f"CloudSEN12+ 509x509 @ 10m resized to 224 ({patch_509_km2:.2f} km^2/patch): needs ~{req_509_pps:.0f} patches/s")
    print(f"Reference Phase 4 CPU INT8 (i7-13620H, b=1, 4T, 3.4573 ms): {1000.0/3.4573:.1f} patches/s")


if __name__ == "__main__":
    main()