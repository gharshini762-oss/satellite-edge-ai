import os
import sys
import csv
import time
import argparse
from datetime import datetime

import numpy as np
import psutil
import torch
from torch.utils.data import DataLoader
import pynvml

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.dataset import CloudDS
from src.models import build_resnet18
from src.metrics import auroc, ece, basic


def get_norm_stats():
    mean_path = "results/norm_mean.npy"
    std_path = "results/norm_std.npy"
    if os.path.exists(mean_path) and os.path.exists(std_path):
        return np.load(mean_path), np.load(std_path)
    x_train = np.load("data/processed/train_X.npy", mmap_mode="r")
    sample = np.clip(np.asarray(x_train[::10], dtype=np.float32) / 10000.0, 0.0, 1.0)
    return sample.mean(axis=(0, 2, 3)), sample.std(axis=(0, 2, 3))


def softmax_pos(logits):
    e = np.exp(logits - logits.max(axis=1, keepdims=True))
    return (e / e.sum(axis=1, keepdims=True))[:, 1]


def get_plugged_in():
    batt = psutil.sensors_battery()
    return bool(batt.power_plugged) if batt is not None else None


@torch.no_grad()
def infer_fp16_logits(model_half, loader, device):
    model_half.eval()
    zs, ys = [], []
    for x, y in loader:
        out = model_half(x.to(device=device, dtype=torch.float16)).float().cpu()
        zs.append(out)
        ys.append(y)
    return torch.cat(zs).numpy(), torch.cat(ys).numpy()


@torch.no_grad()
def bench_fp16_cuda(model_half, batch_size, handle, warmup=50, iters=500):
    model_half.eval()
    inp = torch.randn(batch_size, 3, 224, 224, device="cuda", dtype=torch.float16)

    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    temp_start = float(pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU))

    for _ in range(warmup):
        _ = model_half(inp)
    torch.cuda.synchronize()

    starts = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    ends = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]

    for i in range(iters):
        starts[i].record()
        _ = model_half(inp)
        ends[i].record()
    torch.cuda.synchronize()

    ms = [s.elapsed_time(e) for s, e in zip(starts, ends)]
    peak_alloc_mb = round(torch.cuda.max_memory_allocated() / (1024 ** 2), 2)
    nvml_used_mb = round(pynvml.nvmlDeviceGetMemoryInfo(handle).used / (1024 ** 2), 2)

    return {
        "median_ms": round(float(np.median(ms)), 4),
        "p95_ms": round(float(np.percentile(ms, 95)), 4),
        "peak_vram_mb": peak_alloc_mb,
        "nvml_used_vram_mb": nvml_used_mb,
        "temp_start_c": temp_start,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--power-mode", type=str, default="Balanced")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--cooldown", type=int, default=60)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    args = parser.parse_args()

    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False

    plugged = get_plugged_in()
    print(f"Plugged in: {plugged} | Power mode: {args.power_mode}")
    if plugged is False:
        print("WARNING: Running on battery. Plug in charger before benchmarking.")

    device = torch.device("cuda")
    mean, std = get_norm_stats()
    val_loader = DataLoader(CloudDS("validation", mean, std, train=False), batch_size=64, shuffle=False, num_workers=0)
    test_loader = DataLoader(CloudDS("test", mean, std, train=False), batch_size=64, shuffle=False, num_workers=0)

    y_val_ref = np.load("data/processed/validation_y.npy")
    y_test_ref = np.load("data/processed/test_y.npy")

    summary_rows = []
    test_s0 = {"acc": 0.0, "f1": 0.0, "auroc": 0.0}

    for seed in args.seeds:
        ckpt_path = f"results/ckpt_seed{seed}.pt"
        m16 = build_resnet18()
        m16.load_state_dict(torch.load(ckpt_path, map_location="cpu"))
        m16 = m16.half().to(device).eval()

        z16_val, yv = infer_fp16_logits(m16, val_loader, device)
        z16_test, yt = infer_fp16_logits(m16, test_loader, device)
        assert np.array_equal(yv, y_val_ref), f"Val label mismatch for seed {seed}"
        assert np.array_equal(yt, y_test_ref), f"Test label mismatch for seed {seed}"

        np.save(f"results/val_logits_fp16_seed{seed}.npy", z16_val)
        np.save(f"results/test_logits_fp16_seed{seed}.npy", z16_test)

        if seed == 0:
            z32_val = np.load("results/val_logits_fp32_seed0.npy")
            z32_test = np.load("results/test_logits_fp32_seed0.npy")
        else:
            z32_val = np.load(f"results/val_logits_fp32_seed{seed}.npy")
            z32_test = np.load(f"results/test_logits_fp32_seed{seed}.npy")

        for split_name, z16, z32, y_true in [
            ("validation", z16_val, z32_val, y_val_ref),
            ("test", z16_test, z32_test, y_test_ref),
        ]:
            p16 = softmax_pos(z16)
            p32 = softmax_pos(z32)
            b16 = basic(y_true, p16, thr=0.5)
            b32 = basic(y_true, p32, thr=0.5)
            auc16 = auroc(y_true, p16)
            auc32 = auroc(y_true, p32)
            ece16 = ece(y_true, p16, bins=10)
            ece32 = ece(y_true, p32, bins=10)

            agree = float((z16.argmax(axis=1) == z32.argmax(axis=1)).mean())
            max_diff = float(np.abs(z16 - z32).max())

            summary_rows.append({
                "seed": seed,
                "split": split_name,
                "acc_fp16": round(b16["acc"], 6),
                "acc_delta_vs_fp32": round(b16["acc"] - b32["acc"], 6),
                "f1_fp16": round(b16["f1"], 6),
                "f1_delta_vs_fp32": round(b16["f1"] - b32["f1"], 6),
                "auroc_fp16": round(auc16, 6),
                "auroc_delta_vs_fp32": round(auc16 - auc32, 6),
                "ece_fp16": round(ece16, 6),
                "ece_delta_vs_fp32": round(ece16 - ece32, 6),
                "agree_with_fp32": round(agree, 6),
                "max_logit_diff": round(max_diff, 6),
            })

            if split_name == "test" and seed == 0:
                test_s0 = {"acc": b16["acc"], "f1": b16["f1"], "auroc": auc16}

            print(
                f"Seed {seed} | {split_name:10s} | acc={b16['acc']:.4f} ({b16['acc']-b32['acc']:+.4f}) "
                f"| auroc={auc16:.4f} | agree={agree:.4f} | max_logit_diff={max_diff:.5f}"
            )

    with open("results/day22_fp16_metrics.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
        writer.writeheader()
        writer.writerows(summary_rows)

    pynvml.nvmlInit()
    handle = pynvml.nvmlDeviceGetHandleByIndex(0)

    m_bench = build_resnet18()
    m_bench.load_state_dict(torch.load("results/ckpt_seed0.pt", map_location="cpu"))
    m_bench = m_bench.half().to(device).eval()
    n_params = sum(p.numel() for p in m_bench.parameters())

    tmp_path = "results/_tmp_fp16_ckpt.pt"
    torch.save(m_bench.state_dict(), tmp_path)
    size_mb = round(os.path.getsize(tmp_path) / (1024 ** 2), 2)
    os.remove(tmp_path)

    bench_rows = []
    date_str = datetime.now().strftime("%Y-%m-%d")

    for batch in [1, 8, 32]:
        for rep in range(1, args.repeats + 1):
            print(f"Resting {args.cooldown}s before batch={batch}, repeat={rep}...")
            time.sleep(args.cooldown)
            stats = bench_fp16_cuda(m_bench, batch, handle, warmup=50, iters=500)
            row = {
                "date": date_str,
                "model": "resnet18",
                "precision": "fp16",
                "seed": 0,
                "repeat": rep,
                "batch": batch,
                "median_ms": stats["median_ms"],
                "p95_ms": stats["p95_ms"],
                "peak_vram_mb": stats["peak_vram_mb"],
                "nvml_used_vram_mb": stats["nvml_used_vram_mb"],
                "params": n_params,
                "size_mb": size_mb,
                "temp_start_c": stats["temp_start_c"],
                "plugged_in": plugged,
                "power_mode": args.power_mode,
                "test_acc": round(test_s0["acc"], 6),
                "test_f1": round(test_s0["f1"], 6),
                "test_auroc": round(test_s0["auroc"], 6),
            }
            bench_rows.append(row)
            print(
                f"batch={batch:2d} rep={rep} | start_temp={stats['temp_start_c']:.0f}C | "
                f"median={stats['median_ms']:.4f} ms | p95={stats['p95_ms']:.4f} ms | peak_vram={stats['peak_vram_mb']} MiB"
            )

    pynvml.nvmlShutdown()

    with open("results/day22_fp16_bench.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(bench_rows[0].keys()))
        writer.writeheader()
        writer.writerows(bench_rows)


if __name__ == "__main__":
    main()