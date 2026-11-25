import os
import sys
import csv
import time
import datetime
import argparse
import torch
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.models import build_resnet18

try:
    import pynvml
    pynvml.nvmlInit()
    NVML_HANDLE = pynvml.nvmlDeviceGetHandleByIndex(0)
except Exception:
    NVML_HANDLE = None


def gpu_temp():
    if NVML_HANDLE is None:
        return ""
    return int(pynvml.nvmlDeviceGetTemperature(NVML_HANDLE, pynvml.NVML_TEMPERATURE_GPU))


def load_model_on_gpu(ckpt_path, prec):
    m = build_resnet18()
    m.load_state_dict(torch.load(ckpt_path, map_location="cpu"))
    if prec == "fp16":
        m = m.half()
    return m.cuda().eval()


def bench_one(ckpt_path, prec, batch, warmup=50, iters=500):
    dtype = torch.float16 if prec == "fp16" else torch.float32
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    model = load_model_on_gpu(ckpt_path, prec)
    x = torch.randn(batch, 3, 224, 224, device="cuda", dtype=dtype)

    with torch.no_grad():
        for _ in range(warmup):
            _ = model(x)
        torch.cuda.synchronize()

        times = []
        for _ in range(iters):
            s = torch.cuda.Event(enable_timing=True)
            e = torch.cuda.Event(enable_timing=True)
            s.record()
            _ = model(x)
            e.record()
            torch.cuda.synchronize()
            times.append(s.elapsed_time(e))

    peak_mb = torch.cuda.max_memory_allocated() / (1024 ** 2)
    del model, x
    torch.cuda.empty_cache()

    arr = np.array(times, dtype=np.float64)
    return float(np.median(arr)), float(np.percentile(arr, 95)), float(peak_mb)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--cooldown", type=int, default=60)
    ap.add_argument("--power-mode", type=str, default="Balanced")
    args = ap.parse_args()

    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    assert torch.cuda.is_available()

    ckpt = "results/ckpt_seed0.pt"
    fp16_ckpt = "results/ckpt_seed0_fp16.pt"
    if not os.path.exists(fp16_ckpt):
        m16 = build_resnet18()
        m16.load_state_dict(torch.load(ckpt, map_location="cpu"))
        torch.save(m16.half().state_dict(), fp16_ckpt)
        del m16

    size_map = {
        "fp32": round(os.path.getsize(ckpt) / (1024 ** 2), 2),
        "fp16": round(os.path.getsize(fp16_ckpt) / (1024 ** 2), 2),
    }

    rows = []
    for batch in [1, 8, 32]:
        for rep in range(1, args.repeats + 1):
            for prec in ["fp32", "fp16"]:
                print(f"Cooldown {args.cooldown}s before {prec} batch={batch} rep={rep}...")
                time.sleep(args.cooldown)
                t_start = gpu_temp()
                med, p95, vram = bench_one(ckpt, prec, batch)
                t_end = gpu_temp()
                row = {
                    "date": datetime.date.today().isoformat(),
                    "precision": prec,
                    "batch": batch,
                    "repeat": rep,
                    "median_ms": round(med, 4),
                    "p95_ms": round(p95, 4),
                    "peak_vram_mb": round(vram, 2),
                    "size_mb": size_map[prec],
                    "start_temp_c": t_start,
                    "end_temp_c": t_end,
                    "tf32": False,
                    "plugged_in": True,
                    "power_mode": args.power_mode,
                }
                rows.append(row)
                print(f"  {prec} b={batch} r={rep}: med={med:.4f} ms, p95={p95:.4f} ms, vram={vram:.2f} MiB, T={t_start}->{t_end}C")

    out_path = "results/day22_gpu_paired_bench.csv"
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"Saved {out_path}")


if __name__ == "__main__":
    main()