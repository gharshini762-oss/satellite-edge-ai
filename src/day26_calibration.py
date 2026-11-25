import os
import sys
import csv
import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.metrics import ece
from src.policy import softmax_p, policy, load_split_manifest


def fit_temperature(logits, y):
    best_nll, best_t = 1e9, 1.0
    for temp in np.linspace(0.5, 5.0, 91):
        p = softmax_p(logits / temp)
        nll = float(-(y * np.log(p + 1e-7) + (1.0 - y) * np.log(1.0 - p + 1e-7)).mean())
        if nll < best_nll:
            best_nll, best_t = nll, float(round(temp, 3))
    return best_t


def reliability_curve(y, p, bins=10):
    edges = np.linspace(0.0, 1.0, bins + 1)
    confs, emp_fracs = [], []
    for lo, hi in zip(edges[:-1], edges[1:]):
        mask = (p >= lo) & ((p < hi) if hi < 1.0 else (p <= hi))
        if mask.any():
            confs.append(float(p[mask].mean()))
            emp_fracs.append(float(y[mask].mean()))
    return confs, emp_fracs


def main():
    y_val = np.load("data/processed/validation_y.npy")
    y_test = np.load("data/processed/test_y.npy")
    _, test_cloud, test_clear = load_split_manifest("test")

    specs = [
        ("fp32", "results/val_logits_fp32_seed0.npy", "results/test_logits_fp32_seed0.npy"),
        ("fp16", "results/val_logits_fp16_seed0.npy", "results/test_logits_fp16_seed0.npy"),
        ("int8", "results/val_logits_int8_seed0.npy", "results/test_logits_int8_seed0.npy"),
    ]

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8))
    for ax, heading in zip(axes, ["Test Reliability (Raw)", "Test Reliability (Temp-Scaled)"]):
        ax.plot([0, 1], [0, 1], "k--", alpha=0.5, label="Ideal")
        ax.set_title(heading)
        ax.set_xlabel("Mean Predicted Confidence")
        ax.set_ylabel("Empirical Cloudy Fraction")
        ax.grid(True, alpha=0.3)

    rows = []
    for prec, v_path, t_path in specs:
        zv = np.load(v_path)
        zt = np.load(t_path)

        pv_raw = softmax_p(zv)
        pt_raw = softmax_p(zt)

        t_opt = fit_temperature(zv, y_val)
        pv_cal = softmax_p(zv / t_opt)
        pt_cal = softmax_p(zt / t_opt)

        disc95 = pt_raw >= 0.95
        true_cloudy_frac = float((test_cloud[disc95] >= 70.0).mean()) if disc95.any() else 0.0
        mean_clear_in_disc95 = float(test_clear[disc95].mean()) if disc95.any() else 0.0

        pol95_raw = policy(pt_raw, test_cloud, test_clear, 0.95)
        pol95_cal = policy(pt_cal, test_cloud, test_clear, 0.95)

        rows.append({
            "precision": prec,
            "T_val": t_opt,
            "val_ece_raw": round(ece(y_val, pv_raw, bins=10), 6),
            "val_ece_scaled": round(ece(y_val, pv_cal, bins=10), 6),
            "test_ece_raw": round(ece(y_test, pt_raw, bins=10), 6),
            "test_ece_scaled": round(ece(y_test, pt_cal, bins=10), 6),
            "discarded_at_0_95": int(disc95.sum()),
            "true_cloudy_frac_at_0_95": round(true_cloudy_frac, 6),
            "mean_clear_pct_at_0_95": round(mean_clear_in_disc95, 4),
            "fd_at_0_95_raw": round(pol95_raw["false_deletion"], 6),
            "fd_at_0_95_scaled": round(pol95_cal["false_deletion"], 6),
        })

        c_raw, f_raw = reliability_curve(y_test, pt_raw, bins=10)
        c_cal, f_cal = reliability_curve(y_test, pt_cal, bins=10)
        axes[0].plot(c_raw, f_raw, marker="o", label=f"{prec.upper()} (ECE={ece(y_test, pt_raw, 10):.4f})")
        axes[1].plot(c_cal, f_cal, marker="o", label=f"{prec.upper()} T={t_opt} (ECE={ece(y_test, pt_cal, 10):.4f})")

    axes[0].legend()
    axes[1].legend()
    plt.tight_layout()
    plt.savefig("results/day26_calibration.png", dpi=200)

    with open("results/day26_calibration_results.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()