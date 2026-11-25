
import os
import sys
import csv
import time

import numpy as np
import psutil
import onnxruntime as ort
from onnxruntime.quantization import (
    quantize_static,
    CalibrationDataReader,
    QuantFormat,
    QuantType,
    CalibrationMethod,
)

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.dataset import CloudDS
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


class ArrayCalibReader(CalibrationDataReader):
    def __init__(self, arrays):
        self.iterator = iter([{"input": arr[None].astype(np.float32)} for arr in arrays])

    def get_next(self):
        return next(self.iterator, None)


def run_ort_batches(onnx_path, x_arr, batch_size=64, threads=4):
    opts = ort.SessionOptions()
    opts.intra_op_num_threads = threads
    sess = ort.InferenceSession(onnx_path, sess_options=opts, providers=["CPUExecutionProvider"])
    outs = []
    for start in range(0, len(x_arr), batch_size):
        outs.append(sess.run(None, {"input": x_arr[start : start + batch_size]})[0])
    return np.concatenate(outs, axis=0)


def time_ort_cpu(onnx_path, threads, iters=300, warmup=50):
    opts = ort.SessionOptions()
    opts.intra_op_num_threads = threads
    opts.inter_op_num_threads = 1
    sess = ort.InferenceSession(onnx_path, sess_options=opts, providers=["CPUExecutionProvider"])
    x = np.random.randn(1, 3, 224, 224).astype(np.float32)

    for _ in range(warmup):
        sess.run(None, {"input": x})

    ms = []
    for _ in range(iters):
        t0 = time.perf_counter()
        sess.run(None, {"input": x})
        ms.append((time.perf_counter() - t0) * 1000.0)

    return round(float(np.median(ms)), 4), round(float(np.percentile(ms, 95)), 4)


def main():
    mean, std = get_norm_stats()
    train_eval_ds = CloudDS("train", mean, std, train=False)
    val_ds = CloudDS("validation", mean, std, train=False)
    test_ds = CloudDS("test", mean, std, train=False)

    rng = np.random.default_rng(0)
    calib_idx = rng.choice(len(train_eval_ds), size=256, replace=False)
    calib_samples = [train_eval_ds[int(i)][0].numpy() for i in calib_idx]

    x_val = np.stack([val_ds[i][0].numpy() for i in range(len(val_ds))]).astype(np.float32)
    y_val = np.load("data/processed/validation_y.npy")
    
    z32_val = np.load("results/val_logits_fp32_seed0.npy")
    fp32_val_acc = basic(y_val, softmax_pos(z32_val), thr=0.5)["acc"]
    methods = [
        ("minmax", CalibrationMethod.MinMax),
        ("percentile", CalibrationMethod.Percentile),
        ("entropy", CalibrationMethod.Entropy),
    ]

    val_rows = []
    bench_rows = []
    best_name, best_score, best_path = None, (-1.0, -1.0), None

    batt = psutil.sensors_battery()
    plugged = bool(batt.power_plugged) if batt is not None else None

    for name, cal_enum in methods:
        out_onnx = f"results/resnet18_int8_{name}.onnx"
        quantize_static(
            model_input="results/resnet18_fp32.onnx",
            model_output=out_onnx,
            calibration_data_reader=ArrayCalibReader(calib_samples),
            quant_format=QuantFormat.QDQ,
            activation_type=QuantType.QUInt8,
            weight_type=QuantType.QInt8,
            per_channel=True,
            calibrate_method=cal_enum,
        )

        size_mb = round(os.path.getsize(out_onnx) / (1024 ** 2), 2)
        z_val = run_ort_batches(out_onnx, x_val, batch_size=64, threads=4)
        p_val = softmax_pos(z_val)
        m_val = basic(y_val, p_val, thr=0.5)
        auc_val = auroc(y_val, p_val)
        ece_val = ece(y_val, p_val, bins=10)
        agree_val = float((z_val.argmax(axis=1) == z32_val.argmax(axis=1)).mean())

        val_rows.append({
            "method": name,
            "split": "validation",
            "seed": 0,
            "acc": round(m_val["acc"], 6),
            "acc_drop": round(m_val["acc"] - fp32_val_acc, 6),
            "f1": round(m_val["f1"], 6),
            "auroc": round(auc_val, 6),
            "ece": round(ece_val, 6),
            "agree_with_fp32": round(agree_val, 6),
            "size_mb": size_mb,
        })
        print(
            f"INT8 {name:10s} | val_acc={m_val['acc']:.4f} (diff={m_val['acc']-fp32_val_acc:+.4f}) "
            f"| auroc={auc_val:.4f} | ece={ece_val:.4f} | agree={agree_val:.4f} | size={size_mb} MiB"
        )

        score = (round(m_val["acc"], 6), round(auc_val, 6))
        if score > best_score:
            best_name, best_score, best_path = name, score, out_onnx

        for threads in [1, 2, 4]:
            for rep in [1, 2, 3]:
                med, p95 = time_ort_cpu(out_onnx, threads, iters=300, warmup=50)
                bench_rows.append({
                    "model": "resnet18",
                    "engine": "onnxruntime_cpu",
                    "precision": f"int8_{name}",
                    "seed": 0,
                    "threads": threads,
                    "repeat": rep,
                    "batch": 1,
                    "median_ms": med,
                    "p95_ms": p95,
                    "size_mb": size_mb,
                    "plugged_in": plugged,
                })

    with open("results/day24_int8_val_comparison.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(val_rows[0].keys()))
        writer.writeheader()
        writer.writerows(val_rows)

    with open("results/day24_int8_cpu_bench.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(bench_rows[0].keys()))
        writer.writeheader()
        writer.writerows(bench_rows)

    print(f"\nSelected INT8 recipe on validation: {best_name} ({best_path})")
    z_best_val = run_ort_batches(best_path, x_val, batch_size=64, threads=4)
    x_test = np.stack([test_ds[i][0].numpy() for i in range(len(test_ds))]).astype(np.float32)
    z_best_test = run_ort_batches(best_path, x_test, batch_size=64, threads=4)

    np.save("results/val_logits_int8_seed0.npy", z_best_val)
    np.save("results/test_logits_int8_seed0.npy", z_best_test)

    y_test = np.load("data/processed/test_y.npy")
    z32_test = np.load("results/test_logits_fp32_seed0.npy")
    p_test = softmax_pos(z_best_test)
    m_test = basic(y_test, p_test, thr=0.5)
    print(
        f"INT8 ({best_name}) Test Once | acc={m_test['acc']:.4f} | f1={m_test['f1']:.4f} "
        f"| auroc={auroc(y_test, p_test):.4f} | agree={(z_best_test.argmax(1)==z32_test.argmax(1)).mean():.4f}"
    )


if __name__ == "__main__":
    main()