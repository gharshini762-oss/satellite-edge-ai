import os
import sys
import csv
import time
import datetime
import argparse
import numpy as np
import torch
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.models import build_resnet18
from src.verify_weights import verify_phase4_weights
from src.fault_injection import (
    load_test_bundle,
    get_target_weight_names,
    precompute_int8_cache,
    inject_fp32_flips,
    inject_int8_flips,
    evaluate_state,
)

T_CRIT_MAP = {
    5: 2.77645,
    10: 2.26216,
    15: 2.14479,
    20: 2.09302,
    25: 2.06390,
    30: 2.04523,
    50: 2.00958,
    100: 1.98422,
}


def ci95_halfwidth(vals):
    n = len(vals)
    if n <= 1:
        return 0.0
    t_crit = T_CRIT_MAP.get(n, 1.96)
    return float(t_crit * np.std(vals, ddof=1) / np.sqrt(n))


def summarize_block(rows, experiment, precision, bit_group, bit_label, k_flips):
    accs = np.array([r["acc"] for r in rows], dtype=np.float64)
    fd08 = np.array([r["fd_08"] for r in rows], dtype=np.float64)
    bw08 = np.array([r["bw_08"] for r in rows], dtype=np.float64)
    fd05 = np.array([r["fd_05"] for r in rows], dtype=np.float64)
    nf = np.array([r["non_finite_frac"] for r in rows], dtype=np.float64)

    q25_a, q50_a, q75_a = np.percentile(accs, [25, 50, 75])
    q25_f, q50_f, q75_f = np.percentile(fd08, [25, 50, 75])

    return {
        "experiment": experiment,
        "precision": precision,
        "bit_group": bit_group,
        "bit_label": bit_label,
        "k_flips": k_flips,
        "n_trials": len(rows),
        "acc_mean": round(float(accs.mean()), 4),
        "acc_std": round(float(np.std(accs, ddof=1)), 4) if len(accs) > 1 else 0.0,
        "acc_ci95": round(ci95_halfwidth(accs), 4),
        "acc_median": round(float(q50_a), 4),
        "acc_iqr": round(float(q75_a - q25_a), 4),
        "acc_min": round(float(accs.min()), 4),
        "fd08_mean": round(float(fd08.mean()), 4),
        "fd08_std": round(float(np.std(fd08, ddof=1)), 4) if len(fd08) > 1 else 0.0,
        "fd08_ci95": round(ci95_halfwidth(fd08), 4),
        "fd08_median": round(float(q50_f), 4),
        "fd08_iqr": round(float(q75_f - q25_f), 4),
        "fd08_max": round(float(fd08.max()), 4),
        "fd05_mean": round(float(fd05.mean()), 4),
        "bw08_mean": round(float(bw08.mean()), 4),
        "non_finite_trials": int((nf > 0).sum()),
        "collapse_acc_lt_85": int((accs < 0.85).sum()),
        "catastrophic_fd08_gt_10pct": int((fd08 > 0.10).sum()),
    }


def plot_bit_position(summary_rows, out_png):
    fig, axes = plt.subplots(1, 2, figsize=(13.2, 5.4), gridspec_kw={"width_ratios": [1.35, 1.0]})

    # Panel A: Per-bit accuracy at K=10 (FP32 selected bits vs INT8 all 8 bits)
    ax = axes[0]
    per_bit = [r for r in summary_rows if r["experiment"] == "per_bit_k10"]
    labels = [r["bit_label"] for r in per_bit]
    means = np.array([r["acc_mean"] * 100.0 for r in per_bit])
    cis = np.array([r["acc_ci95"] * 100.0 for r in per_bit])
    err_lo = np.minimum(cis, means)
    err_hi = np.minimum(cis, 100.0 - means)

    bar_colors = []
    for r in per_bit:
        if r["precision"] == "sim_int8":
            bar_colors.append("#2ca02c")
        elif r["bit_group"] == "b31":
            bar_colors.append("#9467bd")  # FP32 Sign
        elif r["bit_group"] == "b30":
            bar_colors.append("#d62728")  # FP32 Exponent MSB
        elif r["bit_group"] in {"b29", "b28", "b27", "b26", "b25", "b24", "b23"}:
            bar_colors.append("#ff7f0e")  # FP32 Other Exponent
        else:
            bar_colors.append("#1f77b4")  # FP32 Mantissa

    xs = np.arange(len(labels))
    ax.bar(xs, means, yerr=[err_lo, err_hi], color=bar_colors, capsize=3, width=0.72, alpha=0.9, edgecolor="black", linewidth=0.5)
    ax.axvline(11.5, color="black", ls="--", lw=1.1, alpha=0.6)
    ax.axhline(91.79, color="#1f77b4", ls=":", lw=1.1, alpha=0.7, label="Clean FP32 (91.8%)")
    ax.axhline(79.59, color="gray", ls="--", lw=1.0, alpha=0.7, label="All-Keep floor (79.6%)")

    ax.set_xticks(xs)
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8.2)
    ax.set_ylim(0, 104)
    ax.set_ylabel("Test Accuracy (%)")
    ax.set_title("Per-Bit Sensitivity at K = 10 Flips (30 Trials/Bit)")
    ax.grid(axis="y", alpha=0.3)

    # Color legend handles
    from matplotlib.patches import Patch
    handles = [
        Patch(facecolor="#1f77b4", edgecolor="black", label="FP32 Mantissa (b0, b11, b22)"),
        Patch(facecolor="#ff7f0e", edgecolor="black", label="FP32 Exponent (b23–b29)"),
        Patch(facecolor="#d62728", edgecolor="black", label="FP32 Exponent MSB (b30)"),
        Patch(facecolor="#9467bd", edgecolor="black", label="FP32 Sign (b31)"),
        Patch(facecolor="#2ca02c", edgecolor="black", label="Sim-INT8 (b0–b7)"),
    ]
    ax.legend(handles=handles, loc="lower left", fontsize=7.6, framealpha=0.92)

    # Panel B: Single-flip (K=1) and Heavy-flip (K=100) field comparison
    ax2 = axes[1]
    grp_rows = [r for r in summary_rows if r["experiment"] in ("single_flip_k1", "field_groups_k100")]
    g_labels = [r["bit_label"] for r in grp_rows]
    g_means = np.array([r["acc_mean"] * 100.0 for r in grp_rows])
    g_cis = np.array([r["acc_ci95"] * 100.0 for r in grp_rows])
    g_lo = np.minimum(g_cis, g_means)
    g_hi = np.minimum(g_cis, 100.0 - g_means)

    g_colors = []
    for r in grp_rows:
        if r["precision"] == "sim_int8":
            g_colors.append("#2ca02c")
        elif "b30" in r["bit_group"]:
            g_colors.append("#d62728")
        elif "exp" in r["bit_group"]:
            g_colors.append("#ff7f0e")
        elif "sign" in r["bit_group"]:
            g_colors.append("#9467bd")
        else:
            g_colors.append("#1f77b4")

    gx = np.arange(len(g_labels))
    bars = ax2.bar(gx, g_means, yerr=[g_lo, g_hi], color=g_colors, capsize=3, width=0.68, alpha=0.9, edgecolor="black", linewidth=0.5)
    ax2.axvline(2.5, color="black", ls="--", lw=1.1, alpha=0.6)
    ax2.axhline(91.79, color="#1f77b4", ls=":", lw=1.1, alpha=0.7)
    ax2.axhline(79.59, color="gray", ls="--", lw=1.0, alpha=0.7)

    for b, m, r in zip(bars, g_means, grp_rows):
        y_txt = min(m + r["acc_ci95"] * 100.0 + 1.8, 99.0)
        ax2.text(b.get_x() + b.get_width() / 2.0, y_txt, f"{m:.1f}%", ha="center", va="bottom", fontsize=7.6, fontweight="bold")

    ax2.text(1.0, 101.5, "K = 1 Flip", ha="center", fontsize=8.5, fontweight="bold")
    ax2.text(5.0, 101.5, "K = 100 Flips", ha="center", fontsize=8.5, fontweight="bold")
    ax2.set_xticks(gx)
    ax2.set_xticklabels(g_labels, rotation=38, ha="right", fontsize=8.2)
    ax2.set_ylim(0, 106)
    ax2.set_ylabel("Test Accuracy (%)")
    ax2.set_title("Field Breakdown: Single Flip (K=1) & Heavy (K=100)")
    ax2.grid(axis="y", alpha=0.3)

    fig.suptitle(
        "ResNet-18 (Seed 0) Bit-Position Fault Analysis: Mean ± 95% t-CI (30 Trials/Point)",
        fontsize=12.2,
        y=1.01,
    )
    plt.tight_layout()
    plt.savefig(out_png, dpi=200, bbox_inches="tight")
    plt.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=30)
    ap.add_argument("--batch-size", type=int, default=64)
    args = ap.parse_args()

    verify_phase4_weights(["results/ckpt_seed0.pt"])
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    assert torch.cuda.is_available()

    X_gpu, y_np, cloud_pct, clear_pct, _ = load_test_bundle("cuda")
    model = build_resnet18().cuda().eval()
    clean_state = torch.load("results/ckpt_seed0.pt", map_location="cpu")
    target_names = get_target_weight_names(model)
    int8_cache = precompute_int8_cache(clean_state, target_names)

    # Plan: (experiment, precision, bit_group, bit_label, bit_positions, k_flips)
    plan = []
    # Part A: Per-bit sensitivity at K=10
    for b in [0, 11, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31]:
        plan.append(("per_bit_k10", "fp32", f"b{b}", f"FP32 b{b}", [b], 10))
    for b in range(8):
        plan.append(("per_bit_k10", "sim_int8", f"b{b}", f"INT8 b{b}", [b], 10))

    # Part B: Single-flip (K=1) catastrophe proof
    plan.append(("single_flip_k1", "fp32", "fp32_b30_k1", "FP32 b30\n(K=1)", [30], 1))
    plan.append(("single_flip_k1", "fp32", "fp32_exp_k1", "FP32 Exp\n(K=1)", list(range(23, 31)), 1))
    plan.append(("single_flip_k1", "sim_int8", "int8_b7_k1", "INT8 b7\n(K=1)", [7], 1))

    # Part C: Heavy-flip (K=100) field comparison
    plan.append(("field_groups_k100", "fp32", "fp32_mant_k100", "FP32 Mant\n(b0–22)", list(range(0, 23)), 100))
    plan.append(("field_groups_k100", "fp32", "fp32_sign_k100", "FP32 Sign\n(b31)", [31], 100))
    plan.append(("field_groups_k100", "fp32", "fp32_exp_k100", "FP32 Exp\n(b23–30)", list(range(23, 31)), 100))
    plan.append(("field_groups_k100", "sim_int8", "int8_low_k100", "INT8 Low\n(b0–6)", list(range(0, 7)), 100))
    plan.append(("field_groups_k100", "sim_int8", "int8_msb_k100", "INT8 MSB\n(b7)", [7], 100))

    today = datetime.date.today().isoformat()
    raw_path = "results/day32_bit_position_raw.csv"
    raw_fields = [
        "date", "experiment", "precision", "bit_group", "k_flips", "trial_seed",
        "acc", "non_finite_frac", "fd_08", "bw_08", "fd_05", "bw_05",
    ]

    summary_rows = []
    total_blocks = len(plan)
    t_all = time.perf_counter()

    with open(raw_path, "w", newline="", encoding="utf-8") as f_raw:
        writer = csv.DictWriter(f_raw, fieldnames=raw_fields)
        writer.writeheader()

        for idx, (exp_name, prec, bgrp, blbl, bpos, k_flips) in enumerate(plan):
            t_blk = time.perf_counter()
            block_rows = []
            for trial in range(args.seeds):
                ss = np.random.SeedSequence([20261126, idx, k_flips, trial])
                rng = np.random.default_rng(ss)
                if prec == "fp32":
                    st, nf = inject_fp32_flips(clean_state, target_names, rng, n_flips=k_flips, bit_positions=bpos)
                else:
                    st, nf = inject_int8_flips(clean_state, int8_cache, target_names, rng, n_flips=k_flips, bit_positions=bpos)
                assert nf == k_flips
                res = evaluate_state(model, st, X_gpu, y_np, cloud_pct, clear_pct, args.batch_size)
                row = {
                    "date": today,
                    "experiment": exp_name,
                    "precision": prec,
                    "bit_group": bgrp,
                    "k_flips": k_flips,
                    "trial_seed": trial,
                    "acc": round(res["acc"], 6),
                    "non_finite_frac": round(res["non_finite_frac"], 6),
                    "fd_08": round(res["fd_08"], 6),
                    "bw_08": round(res["bw_08"], 6),
                    "fd_05": round(res["fd_05"], 6),
                    "bw_05": round(res["bw_05"], 6),
                }
                writer.writerow(row)
                block_rows.append(row)

            f_raw.flush()
            srow = summarize_block(block_rows, exp_name, prec, bgrp, blbl, k_flips)
            summary_rows.append(srow)
            elapsed = time.perf_counter() - t_all
            eta = (elapsed / (idx + 1)) * (total_blocks - (idx + 1))
            print(
                f"[{idx+1:2d}/{total_blocks}] {exp_name:17s} {prec:8s} {bgrp:14s} K={k_flips:3d} | "
                f"acc={srow['acc_mean']:.4f}±{srow['acc_ci95']:.4f} (med {srow['acc_median']:.4f}, min {srow['acc_min']:.4f}) | "
                f"fd@0.8={srow['fd08_mean']:.4f} (max {srow['fd08_max']:.4f}) | "
                f"nf={srow['non_finite_trials']:2d} col={srow['collapse_acc_lt_85']:2d} | {time.perf_counter()-t_blk:.1f}s (ETA {eta/60:.1f}m)"
            )

    sum_path = "results/day32_bit_position_summary.csv"
    with open(sum_path, "w", newline="", encoding="utf-8") as f_sum:
        w = csv.DictWriter(f_sum, fieldnames=list(summary_rows[0].keys()))
        w.writeheader()
        w.writerows(summary_rows)

    png_path = "results/day32_bit_position.png"
    plot_bit_position(summary_rows, png_path)

    print(f"\nSaved {raw_path}, {sum_path}, and {png_path}")
    print(f"Total elapsed: {(time.perf_counter() - t_all)/60:.2f} minutes")


if __name__ == "__main__":
    main()