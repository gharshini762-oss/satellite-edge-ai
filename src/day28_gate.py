import os
import sys
import csv
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.metrics import auroc, ece, basic
from src.policy import softmax_p


def read_csv(path):
    with open(path, "r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write_csv(path, rows):
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def sample_std(vals):
    if len(vals) <= 1:
        return ""
    return round(float(np.std(vals, ddof=1)), 4)


def main():
    y_test = np.load("data/processed/test_y.npy")

    # Table II: Accuracy, F1, AUROC, ECE per precision (3-seed, sample std ddof=1)
    table2 = []
    for prec, seeds in [("fp32", [0, 1, 2]), ("fp16", [0, 1, 2]), ("int8", [0, 1, 2])]:
        accs, f1s, aucs, eces = [], [], [], []
        for s in seeds:
            path = f"results/test_logits_{prec}_seed{s}.npy"
            p = softmax_p(np.load(path))
            b = basic(y_test, p, thr=0.5)
            accs.append(b["acc"])
            f1s.append(b["f1"])
            aucs.append(auroc(y_test, p))
            eces.append(ece(y_test, p, bins=10))

        table2.append({
            "precision": prec,
            "seeds": ",".join(map(str, seeds)),
            "test_acc_mean": round(float(np.mean(accs)), 4),
            "test_acc_std": sample_std(accs),
            "test_f1_mean": round(float(np.mean(f1s)), 4),
            "test_f1_std": sample_std(f1s),
            "test_auroc_mean": round(float(np.mean(aucs)), 4),
            "test_auroc_std": sample_std(aucs),
            "test_ece_mean": round(float(np.mean(eces)), 4),
            "test_ece_std": sample_std(eces),
        })
    write_csv("results/day28_table2_accuracy.csv", table2)

    # Table III: Systems (Same-session paired GPU bench + CPU ONNX Runtime, ddof=1, max VRAM)
    table3 = []
    gpu_rows = read_csv("results/day22_gpu_paired_bench.csv")
    gpu_means = {}
    for prec in ["fp32", "fp16"]:
        for batch in ["1", "8", "32"]:
            sub = [
                r for r in gpu_rows
                if r["precision"] == prec and int(float(r["batch"])) == int(batch)
            ]
            if not sub:
                continue
            meds = [float(r["median_ms"]) for r in sub]
            p95s = [float(r["p95_ms"]) for r in sub]
            vrams = [float(r["peak_vram_mb"]) for r in sub]
            m_med = float(np.mean(meds))
            gpu_means[(prec, batch)] = (m_med, max(vrams))
            table3.append({
                "hardware": "GPU (RTX 5060, paired)",
                "precision": prec,
                "setting": f"batch={batch}",
                "median_ms_mean": round(m_med, 4),
                "median_ms_std": sample_std(meds),
                "p95_ms_mean": round(float(np.mean(p95s)), 4),
                "size_mb": sub[0].get("size_mb", ""),
                "peak_vram_mb": round(max(vrams), 2),
            })

    for path in ["results/day23_cpu_fp32_bench.csv", "results/day24_int8_cpu_bench.csv"]:
        rows = read_csv(path)
        for prec in sorted(set(r["precision"] for r in rows)):
            for th in ["1", "2", "4"]:
                sub = [r for r in rows if r["precision"] == prec and str(r["threads"]) == th]
                if not sub:
                    continue
                meds = [float(r["median_ms"]) for r in sub]
                p95s = [float(r["p95_ms"]) for r in sub]
                table3.append({
                    "hardware": "CPU (ONNX Runtime)",
                    "precision": prec,
                    "setting": f"batch=1, threads={th}",
                    "median_ms_mean": round(float(np.mean(meds)), 4),
                    "median_ms_std": sample_std(meds),
                    "p95_ms_mean": round(float(np.mean(p95s)), 4),
                    "size_mb": sub[0].get("size_mb", ""),
                    "peak_vram_mb": "N/A",
                })
    write_csv("results/day28_table3_systems.csv", table3)

    # Table IV: Threshold sweep table
    table4 = read_csv("results/day25_test_policy_ci.csv")
    write_csv("results/day28_table4_policy.csv", table4)

    print("\n=== Paired GPU FP32 -> FP16 Speedups & VRAM Drops ===")
    for b in ["1", "8", "32"]:
        m32, v32 = gpu_means[("fp32", b)]
        m16, v16 = gpu_means[("fp16", b)]
        print(f"batch={b:2s}: {m32:.4f} -> {m16:.4f} ms ({m32/m16:.2f}x faster) | VRAM {v32:.2f} -> {v16:.2f} MiB ({(1 - v16/v32)*100:.1f}% drop)")

    required = [
        "results/day28_table2_accuracy.csv",
        "results/day28_table3_systems.csv",
        "results/day28_table4_policy.csv",
        "results/day25_policy_tradeoff.png",
        "results/day26_calibration.png",
        "results/day27_compression_bars.png",
    ]
    print("\n=== Phase 4 Gate Artifact Check ===")
    for p in required:
        print(f"[{'OK' if os.path.exists(p) else 'MISSING'}] {p}")


if __name__ == "__main__":
    main()