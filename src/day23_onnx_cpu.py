import os
import sys
import csv
import time

import numpy as np
import psutil
import torch
import onnxruntime as ort

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.dataset import CloudDS
from src.models import build_resnet18


def get_norm_stats():
    mean_path = "results/norm_mean.npy"
    std_path = "results/norm_std.npy"
    if os.path.exists(mean_path) and os.path.exists(std_path):
        return np.load(mean_path), np.load(std_path)
    x_train = np.load("data/processed/train_X.npy", mmap_mode="r")
    sample = np.clip(np.asarray(x_train[::10], dtype=np.float32) / 10000.0, 0.0, 1.0)
    return sample.mean(axis=(0, 2, 3)), sample.std(axis=(0, 2, 3))


def time_ort_cpu(onnx_path, threads, iters=300, warmup=50):
    opts = ort.SessionOptions()
    opts.intra_op_num_threads = threads
    opts.inter_op_num_threads = 1
    sess = ort.InferenceSession(onnx_path, sess_options=opts, providers=["CPUExecutionProvider"])
    x = np.random.randn(1, 3, 224, 224).astype(np.float32)

    for _ in range(warmup):
        sess.run(None, {"input": x})

    latencies_ms = []
    for _ in range(iters):
        t0 = time.perf_counter()
        sess.run(None, {"input": x})
        latencies_ms.append((time.perf_counter() - t0) * 1000.0)

    return round(float(np.median(latencies_ms)), 4), round(float(np.percentile(latencies_ms, 95)), 4)


def main():
    model = build_resnet18()
    model.load_state_dict(torch.load("results/ckpt_seed0.pt", map_location="cpu"))
    model.eval()

    onnx_path = "results/resnet18_fp32.onnx"
    dummy = torch.randn(1, 3, 224, 224, dtype=torch.float32)

    try:
        torch.onnx.export(
            model,
            dummy,
            onnx_path,
            input_names=["input"],
            output_names=["logits"],
            dynamic_axes={"input": {0: "batch"}, "logits": {0: "batch"}},
            opset_version=17,
            dynamo=False,
        )
    except TypeError:
        torch.onnx.export(
            model,
            dummy,
            onnx_path,
            input_names=["input"],
            output_names=["logits"],
            dynamic_axes={"input": {0: "batch"}, "logits": {0: "batch"}},
            opset_version=17,
        )

    size_mb = round(os.path.getsize(onnx_path) / (1024 ** 2), 2)
    print(f"Saved {onnx_path} ({size_mb} MiB)")

    mean, std = get_norm_stats()
    val_ds = CloudDS("validation", mean, std, train=False)
    x_256 = np.stack([val_ds[i][0].numpy() for i in range(256)]).astype(np.float32)
    z32_256 = np.load("results/val_logits_fp32_seed0.npy")[:256]

    sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    z_ort = sess.run(None, {"input": x_256})[0]
    max_abs_diff = float(np.abs(z_ort - z32_256).max())
    agree = float((z_ort.argmax(axis=1) == z32_256.argmax(axis=1)).mean())
    print(f"256-val parity: max_abs_diff={max_abs_diff:.6e}, argmax_agree={agree:.4f}")

    batt = psutil.sensors_battery()
    plugged = bool(batt.power_plugged) if batt is not None else None

    rows = []
    for threads in [1, 2, 4]:
        for rep in [1, 2, 3]:
            med, p95 = time_ort_cpu(onnx_path, threads, iters=300, warmup=50)
            rows.append({
                "model": "resnet18",
                "engine": "onnxruntime_cpu",
                "precision": "fp32",
                "seed": 0,
                "threads": threads,
                "repeat": rep,
                "batch": 1,
                "median_ms": med,
                "p95_ms": p95,
                "size_mb": size_mb,
                "parity_max_diff": round(max_abs_diff, 7),
                "plugged_in": plugged,
            })
            print(f"threads={threads} rep={rep} | median={med:.4f} ms | p95={p95:.4f} ms")

    with open("results/day23_cpu_fp32_bench.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()