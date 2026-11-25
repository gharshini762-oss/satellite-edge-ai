import os
import sys
import csv
import numpy as np
import torch
import onnxruntime as ort
from onnxruntime.quantization import (
    quantize_static,
    CalibrationDataReader,
    QuantFormat,
    QuantType,
    CalibrationMethod,
)

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.models import build_resnet18
from src.dataset import CloudDS, load_norm_stats
from src.metrics import auroc, ece, basic
from src.policy import softmax_p


class Reader(CalibrationDataReader):
    def __init__(self, arrays):
        self.it = iter([{"input": a[None].astype(np.float32)} for a in arrays])

    def get_next(self):
        return next(self.it, None)


def ort_predict(onnx_path, loader):
    sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    outs = []
    for x, _ in loader:
        outs.append(sess.run(None, {"input": x.numpy().astype(np.float32)})[0])
    return np.concatenate(outs, axis=0)


def main():
    mean, std = load_norm_stats()
    train_ds = CloudDS("train", mean, std, train=False)
    val_loader = torch.utils.data.DataLoader(
        CloudDS("validation", mean, std, train=False), batch_size=64, shuffle=False
    )
    test_loader = torch.utils.data.DataLoader(
        CloudDS("test", mean, std, train=False), batch_size=64, shuffle=False
    )
    y_val = np.load("data/processed/validation_y.npy")
    y_test = np.load("data/processed/test_y.npy")

    # Same 256 train images (seed=0) used for Seed 0 calibration
    rng = np.random.default_rng(0)
    idx = rng.choice(len(train_ds), 256, replace=False)
    calib = [train_ds[int(i)][0].numpy() for i in idx]

    for seed in [1, 2]:
        m = build_resnet18()
        m.load_state_dict(torch.load(f"results/ckpt_seed{seed}.pt", map_location="cpu"))
        m.eval()

        fp32_onnx = f"results/resnet18_fp32_seed{seed}.onnx"
        int8_onnx = f"results/resnet18_int8_minmax_seed{seed}.onnx"

        torch.onnx.export(
            m,
            torch.randn(1, 3, 224, 224),
            fp32_onnx,
            input_names=["input"],
            output_names=["logits"],
            dynamic_axes={"input": {0: "batch"}, "logits": {0: "batch"}},
            opset_version=17,
            dynamo=False,
        )

        # Quick ONNX FP32 parity check against saved FP32 validation logits
        z32_val = np.load(f"results/val_logits_fp32_seed{seed}.npy")
        z_onnx_val = ort_predict(fp32_onnx, val_loader)
        parity = float(np.abs(z_onnx_val - z32_val).max())
        print(f"Seed {seed} ONNX FP32 val parity max_abs_diff={parity:.2e}")

        quantize_static(
            fp32_onnx,
            int8_onnx,
            Reader(calib),
            quant_format=QuantFormat.QDQ,
            activation_type=QuantType.QUInt8,
            weight_type=QuantType.QInt8,
            per_channel=True,
            calibrate_method=CalibrationMethod.MinMax,
        )

        z_val_q = ort_predict(int8_onnx, val_loader)
        z_test_q = ort_predict(int8_onnx, test_loader)
        np.save(f"results/val_logits_int8_seed{seed}.npy", z_val_q)
        np.save(f"results/test_logits_int8_seed{seed}.npy", z_test_q)
        print(f"Saved Seed {seed} INT8 val/test logits ({os.path.getsize(int8_onnx)/1024**2:.2f} MiB)")

    # Print and save 3-seed INT8 summary
    rows = []
    for seed in [0, 1, 2]:
        z32_v = np.load("results/val_logits_fp32_seed0.npy" if seed == 0 else f"results/val_logits_fp32_seed{seed}.npy")
        z32_t = np.load("results/test_logits_fp32_seed0.npy" if seed == 0 else f"results/test_logits_fp32_seed{seed}.npy")
        zq_v = np.load(f"results/val_logits_int8_seed{seed}.npy")
        zq_t = np.load(f"results/test_logits_int8_seed{seed}.npy")

        for split, y, z32, zq in [("validation", y_val, z32_v, zq_v), ("test", y_test, z32_t, zq_t)]:
            p32, pq = softmax_p(z32), softmax_p(zq)
            b32, bq = basic(y, p32, 0.5), basic(y, pq, 0.5)
            row = {
                "seed": seed,
                "split": split,
                "acc_int8": round(bq["acc"], 6),
                "acc_delta_vs_fp32": round(bq["acc"] - b32["acc"], 6),
                "f1_int8": round(bq["f1"], 6),
                "auroc_int8": round(auroc(y, pq), 6),
                "ece_int8": round(ece(y, pq, 10), 6),
                "agree_with_fp32": round(float((zq.argmax(1) == z32.argmax(1)).mean()), 6),
            }
            rows.append(row)
            print(f"Seed {seed} | {split:10s} | acc={row['acc_int8']:.4f} ({row['acc_delta_vs_fp32']:+.4f}) "
                  f"| auroc={row['auroc_int8']:.4f} | ece={row['ece_int8']:.4f} | agree={row['agree_with_fp32']:.4f}")

    with open("results/day24_int8_multiseed_metrics.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


if __name__ == "__main__":
    main()