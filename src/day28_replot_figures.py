import os
import sys
import csv
import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.policy import softmax_p
from src.metrics import ece


def pick_col(cols, candidates):
    for c in candidates:
        if c in cols:
            return c
    for c in candidates:
        for col in cols:
            if c in col.lower():
                return col
    return None


def replot_tradeoff():
    with open("results/day25_test_policy_ci.csv", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    cols = list(rows[0].keys())
    print("day25_test_policy_ci.csv columns:", cols)

    c_prec = pick_col(cols, ["precision", "prec", "model"])
    c_tau = pick_col(cols, ["tau", "threshold", "thr"])
    c_bw = pick_col(cols, ["bandwidth_saved", "bw_saved", "bw_mean", "bw"])
    c_fd = pick_col(cols, ["false_deletion", "false_del", "fd_rate", "fdr", "fd"])
    c_bw_lo = pick_col(cols, ["bw_ci95_lo", "bw_lo", "bw_ci_lo", "bandwidth_saved_lo"])
    c_bw_hi = pick_col(cols, ["bw_ci95_hi", "bw_hi", "bw_ci_hi", "bandwidth_saved_hi"])
    c_fd_lo = pick_col(cols, ["fd_ci95_lo", "fd_lo", "fd_ci_lo", "false_deletion_lo", "fdr_lo"])
    c_fd_hi = pick_col(cols, ["fd_ci95_hi", "fd_hi", "fd_ci_hi", "false_deletion_hi", "fdr_hi"])

    # Detect whether values are stored as fractions [0,1] or percentages [0,100]
    scale = 100.0 if max(float(r[c_bw]) for r in rows) <= 1.0 else 1.0

    plt.figure(figsize=(8.6, 5.6))
    styles = {
        "fp32": dict(color="#1f77b4", marker="o", ls="-", lw=2.8, ms=7.2, label="FP32 (coincident with FP16)", zorder=3),
        "fp16": dict(color="#ff7f0e", marker="s", ls="--", lw=1.8, ms=5.2, mfc="none", mew=1.8, label="FP16 (coincident with FP32)", zorder=4),
        "int8": dict(color="#2ca02c", marker="^", ls="-", lw=2.0, ms=6.5, label="INT8 (MinMax)", zorder=5),
    }

    for prec in ["fp32", "fp16", "int8"]:
        sub = [r for r in rows if r[c_prec].strip().lower() == prec]
        sub.sort(key=lambda r: float(r[c_bw]))
        bw = np.array([float(r[c_bw]) * scale for r in sub])
        fd = np.array([float(r[c_fd]) * scale for r in sub])
        plt.plot(bw, fd, **styles[prec])

        if prec in ("fp32", "int8") and all([c_bw_lo, c_bw_hi, c_fd_lo, c_fd_hi]):
            bw_lo = np.maximum(0.0, bw - np.array([float(r[c_bw_lo]) * scale for r in sub]))
            bw_hi = np.maximum(0.0, np.array([float(r[c_bw_hi]) * scale for r in sub]) - bw)
            fd_lo = np.maximum(0.0, fd - np.array([float(r[c_fd_lo]) * scale for r in sub]))
            fd_hi = np.maximum(0.0, np.array([float(r[c_fd_hi]) * scale for r in sub]) - fd)
            plt.errorbar(
                bw, fd, xerr=[bw_lo, bw_hi], yerr=[fd_lo, fd_hi],
                fmt="none", ecolor=styles[prec]["color"], alpha=0.35, capsize=3, lw=1.1, zorder=2
            )

        if prec == "fp32":
            offsets = [(4, 8), (4, -14), (4, 8), (6, -14), (8, -10), (-42, 6)]
            for idx, (r, x, y) in enumerate(zip(sub, bw, fd)):
                tau_val = float(r[c_tau])
                ox, oy = offsets[idx % len(offsets)]
                plt.annotate(f"τ={r[c_tau]}", (x, y), textcoords="offset points", xytext=(ox, oy),
                             fontsize=8.5, color="#1f77b4", fontweight="bold" if tau_val in (0.5, 0.8) else "normal")
                if tau_val in (0.5, 0.8):
                    plt.scatter([x], [y], s=170, facecolors="none", edgecolors="black", linewidths=1.6, zorder=6)

    plt.scatter([], [], s=120, facecolors="none", edgecolors="black", linewidths=1.6,
                label="Val-selected τ (0.8 for ≤1%, 0.5 for ≤5%)")
    plt.axhline(1.0, color="#d62728", ls="--", lw=1.2, label="1% False-Deletion Cap")
    plt.axhline(5.0, color="#ff9800", ls=":", lw=1.4, label="5% False-Deletion Cap")
    plt.xlabel("Bandwidth Saved (% of Test Tiles Discarded)")
    plt.ylabel("False-Deletion Rate (% of Keep-Worthy Tiles Discarded)")
    plt.title("Test Policy Trade-Off with 95% Cluster Bootstrap CIs (Seed 0; FP32 ≡ FP16)")
    plt.grid(alpha=0.3)
    plt.legend(loc="upper left", fontsize=8.3)
    plt.tight_layout()
    plt.savefig("results/day25_policy_tradeoff.png", dpi=200)
    plt.close()


def reliability_points(y, p, bins=10):
    edges = np.linspace(0.0, 1.0, bins + 1)
    xs, ys, ns = [], [], []
    for i in range(bins):
        lo, hi = edges[i], edges[i + 1]
        mask = (p >= lo) & (p <= hi if i == bins - 1 else p < hi)
        if mask.sum() > 0:
            xs.append(float(p[mask].mean()))
            ys.append(float(y[mask].mean()))
            ns.append(int(mask.sum()))
    return xs, ys, ns


def replot_calibration():
    y = np.load("data/processed/test_y.npy")
    with open("results/day26_calibration_results.csv", encoding="utf-8") as f:
        cal_rows = list(csv.DictReader(f))
    cols = list(cal_rows[0].keys())
    print("day26_calibration_results.csv columns:", cols)

    c_prec = pick_col(cols, ["precision", "prec", "model"])
    c_T = pick_col(cols, ["T_val", "T_opt", "T", "temp", "temperature"])
    cal = {r[c_prec].strip().lower(): float(r[c_T]) for r in cal_rows}

    logits = {
        "fp32": np.load("results/test_logits_fp32_seed0.npy"),
        "fp16": np.load("results/test_logits_fp16_seed0.npy"),
        "int8": np.load("results/test_logits_int8_seed0.npy"),
    }

    fig, axes = plt.subplots(1, 2, figsize=(11.8, 5.0), sharey=True)
    styles = {
        "fp32": dict(color="#1f77b4", marker="o", ls="-", lw=2.6, ms=6.8, zorder=3),
        "fp16": dict(color="#ff7f0e", marker="s", ls="--", lw=1.7, ms=5.0, mfc="none", mew=1.6, zorder=4),
        "int8": dict(color="#2ca02c", marker="^", ls="-", lw=1.9, ms=6.0, zorder=5),
    }

    for ax, scaled, title in [(axes[0], False, "Test Reliability (Raw)"), (axes[1], True, "Test Reliability (Temp-Scaled)")]:
        ax.plot([0, 1], [0, 1], "--", color="gray", lw=1.1, label="Ideal")
        for prec in ["fp32", "fp16", "int8"]:
            T = cal[prec] if scaled else 1.0
            p = softmax_p(logits[prec] / T)
            val_ece = ece(y, p, 10)
            xs, ys, ns = reliability_points(y, p, 10)
            lbl = f"{prec.upper()} (ECE={val_ece:.4f})" if not scaled else f"{prec.upper()} T={T:.2f} (ECE={val_ece:.4f})"
            ax.plot(xs, ys, label=lbl, **styles[prec])
            if prec == "fp32":
                for idx, (bx, by, bn) in enumerate(zip(xs, ys, ns)):
                    dy = 8 if idx % 2 == 0 else -13
                    ax.annotate(f"n={bn}", (bx, by), textcoords="offset points", xytext=(0, dy),
                                ha="center", fontsize=7.2, color="#1f77b4")
        ax.set_title(title)
        ax.set_xlabel("Mean Predicted Confidence")
        ax.set_xlim(-0.03, 1.03)
        ax.set_ylim(-0.05, 1.06)
        ax.grid(alpha=0.3)
        ax.legend(loc="upper left", fontsize=8.2)

    axes[0].set_ylabel("Empirical Cloudy Fraction")
    fig.suptitle("Test Reliability Diagrams (Seed 0, 10 Bins; FP32 ≡ FP16; Blue Labels = FP32 Bin Counts)", y=1.01)
    plt.tight_layout()
    plt.savefig("results/day26_calibration.png", dpi=200, bbox_inches="tight")
    plt.close()


if __name__ == "__main__":
    replot_tradeoff()
    replot_calibration()
    print("Regenerated results/day25_policy_tradeoff.png and results/day26_calibration.png")