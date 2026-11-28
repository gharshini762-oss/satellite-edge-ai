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

BER_LEVELS = [1e-8, 1e-7, 1e-6, 1e-5, 1e-4]
FLIP_LEVELS = [1, 10, 30, 100, 1000, 10000]

# Exact two-sided Student-t 95% critical values (df = n - 1)
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


def summarize_group(rows, mode, precision, level):
    accs = np.array([r["acc"] for r in rows], dtype=np.float64)
    fd08 = np.array([r["fd_08"] for r in rows], dtype=np.float64)
    bw08 = np.array([r["bw_08"] for r in rows], dtype=np.float64)
    fd05 = np.array([r["fd_05"] for r in rows], dtype=np.float64)
    flips = np.array([r["n_flips"] for r in rows], dtype=np.float64)
    nf = np.array([r["non_finite_frac"] for r in rows], dtype=np.float64)

    q25_a, q50_a, q75_a = np.percentile(accs, [25, 50, 75])
    q25_f, q50_f, q75_f = np.percentile(fd08, [25, 50, 75])

    return {
        "mode": mode,
        "precision": precision,
        "level": level,
        "n_trials": len(rows),
        "mean_flips": round(float(flips.mean()), 2),
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
        "fd05_ci95": round(ci95_halfwidth(fd05), 4),
        "bw08_mean": round(float(bw08.mean()), 4),
        "non_finite_trials": int((nf > 0).sum()),
        "collapse_acc_lt_85": int((accs < 0.85).sum()),
        "catastrophic_fd08_gt_10pct": int((fd08 > 0.10).sum()),
    }


def plot_curves(summary_rows, clean_refs, out_png):
    fig, axes = plt.subplots(2, 2, figsize=(11.6, 8.6))
    styles = {
        "fp32": dict(color="#1f77b4", marker="o", lw=2.2, ms=6.0, label="FP32 (native)"),
        "sim_int8": dict(color="#2ca02c", marker="^", lw=2.2, ms=6.2, label="INT8 (sim. per-channel)"),
    }

    configs = [
        ("equal_ber", "Per-Bit Error Rate (BER)", 0),
        ("equal_flips", "Total Bit Flips (K)", 1),
    ]

    for mode, xlabel, col in configs:
        ax_acc = axes[0, col]
        ax_fd = axes[1, col]

        for prec in ["fp32", "sim_int8"]:
            sub = [r for r in summary_rows if r["mode"] == mode and r["precision"] == prec]
            sub.sort(key=lambda r: float(r["level"]))
            xs = np.array([float(r["level"]) for r in sub])

            # Accuracy (in %)
            ym = np.array([r["acc_mean"] * 100.0 for r in sub])
            yc = np.array([r["acc_ci95"] * 100.0 for r in sub])
            ylo = np.clip(ym - yc, 0.0, 100.0)
            yhi = np.clip(ym + yc, 0.0, 100.0)
            ax_acc.plot(xs, ym, **styles[prec])
            ax_acc.fill_between(xs, ylo, yhi, color=styles[prec]["color"], alpha=0.18)

            # False-deletion rate at tau=0.8 (in %)
            fm = np.array([r["fd08_mean"] * 100.0 for r in sub])
            fc = np.array([r["fd08_ci95"] * 100.0 for r in sub])
            flo = np.clip(fm - fc, 0.0, 100.0)
            fhi = np.clip(fm + fc, 0.0, 100.0)
            ax_fd.plot(xs, fm, **styles[prec])
            ax_fd.fill_between(xs, flo, fhi, color=styles[prec]["color"], alpha=0.18)

        # Reference lines on accuracy panel
        ax_acc.axhline(clean_refs["fp32"]["acc"] * 100.0, color="#1f77b4", ls=":", lw=1.1, alpha=0.7, label="Clean FP32 (91.8%)")
        ax_acc.axhline(79.59, color="gray", ls="--", lw=1.1, label="All-Keep floor (79.6%)")
        ax_acc.set_xscale("log")
        ax_acc.set_ylim(-2.0, 100.0)
        ax_acc.set_ylabel("Test Accuracy (%)" if col == 0 else "")
        ax_acc.set_title(
            "Equal Per-Bit BER (FP32 = 4× Flips)" if col == 0 else "Equal Total Bit Flips (Representation Only)"
        )
        ax_acc.grid(alpha=0.3, which="both")
        ax_acc.legend(loc="lower left", fontsize=8.0)

        # Reference lines on false-deletion panel
        ax_fd.axhline(1.0, color="#d62728", ls="--", lw=1.1, label="1% False-Deletion Cap")
        ax_fd.axhline(5.0, color="#ff9800", ls=":", lw=1.3, label="5% False-Deletion Cap")
        ax_fd.set_xscale("log")
        ax_fd.set_xlabel(xlabel)
        ax_fd.set_ylabel("False-Deletion Rate at τ=0.8 (%)" if col == 0 else "")
        ax_fd.grid(alpha=0.3, which="both")
        ax_fd.legend(loc="upper left", fontsize=8.0)

    fig.suptitle(
        "ResNet-18 (Seed 0) Weight Bit-Flip Resilience: Mean ± 95% t-CI Across 30 Trials/Point",
        fontsize=12.5,
        y=0.99,
    )
    plt.tight_layout()
    plt.savefig(out_png, dpi=200, bbox_inches="tight")
    plt.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=30, help="Monte Carlo trials per point (default 30)")
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

    # Clean BER=0 baselines
    clean_fp32 = evaluate_state(model, clean_state, X_gpu, y_np, cloud_pct, clear_pct, args.batch_size)
    int8_clean_state, _ = inject_int8_flips(clean_state, int8_cache, target_names, np.random.default_rng(0), n_flips=0)
    clean_int8 = evaluate_state(model, int8_clean_state, X_gpu, y_np, cloud_pct, clear_pct, args.batch_size)
    clean_refs = {"fp32": clean_fp32, "sim_int8": clean_int8}
    print(f"Clean FP32    : acc={clean_fp32['acc']:.4f}, fd@0.8={clean_fp32['fd_08']:.4f}, fd@0.5={clean_fp32['fd_05']:.4f}")
    print(f"Clean Sim-INT8: acc={clean_int8['acc']:.4f}, fd@0.8={clean_int8['fd_08']:.4f}, fd@0.5={clean_int8['fd_05']:.4f}")

    today = datetime.date.today().isoformat()
    raw_path = "results/day31_ber_sweep_raw.csv"
    raw_fields = [
        "date", "mode", "precision", "level", "trial_seed", "n_flips",
        "acc", "non_finite_frac", "fd_08", "bw_08", "fd_05", "bw_05",
    ]

    summary_rows = []
    sweep_plan = [
        (0, "equal_ber", BER_LEVELS),
        (1, "equal_flips", FLIP_LEVELS),
    ]
    total_blocks = sum(len(lvls) * 2 for _, _, lvls in sweep_plan)
    done_blocks = 0
    t_all = time.perf_counter()

    with open(raw_path, "w", newline="", encoding="utf-8") as f_raw:
        writer = csv.DictWriter(f_raw, fieldnames=raw_fields)
        writer.writeheader()

        for mode_id, mode, levels in sweep_plan:
            for lvl_idx, lvl in enumerate(levels):
                for prec_id, prec in enumerate(["fp32", "sim_int8"]):
                    block_rows = []
                    t_blk = time.perf_counter()

                    for trial in range(args.seeds):
                        ss = np.random.SeedSequence([20261125, mode_id, prec_id, lvl_idx, trial])
                        rng = np.random.default_rng(ss)

                        if mode == "equal_ber":
                            if prec == "fp32":
                                st, nf = inject_fp32_flips(clean_state, target_names, rng, ber=lvl)
                            else:
                                st, nf = inject_int8_flips(clean_state, int8_cache, target_names, rng, ber=lvl)
                        else:
                            if prec == "fp32":
                                st, nf = inject_fp32_flips(clean_state, target_names, rng, n_flips=int(lvl))
                            else:
                                st, nf = inject_int8_flips(clean_state, int8_cache, target_names, rng, n_flips=int(lvl))

                        res = evaluate_state(model, st, X_gpu, y_np, cloud_pct, clear_pct, args.batch_size)
                        row = {
                            "date": today,
                            "mode": mode,
                            "precision": prec,
                            "level": lvl,
                            "trial_seed": trial,
                            "n_flips": nf,
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
                    srow = summarize_group(block_rows, mode, prec, lvl)
                    summary_rows.append(srow)
                    done_blocks += 1
                    elapsed = time.perf_counter() - t_all
                    eta = (elapsed / done_blocks) * (total_blocks - done_blocks)

                    print(
                        f"[{done_blocks:2d}/{total_blocks}] {mode:11s} {prec:8s} lvl={str(lvl):>6s} | "
                        f"flips={srow['mean_flips']:7.1f} | "
                        f"acc={srow['acc_mean']:.4f}±{srow['acc_ci95']:.4f} (med {srow['acc_median']:.4f}, min {srow['acc_min']:.4f}) | "
                        f"fd@0.8={srow['fd08_mean']:.4f} (max {srow['fd08_max']:.4f}) | "
                        f"nf={srow['non_finite_trials']:2d} | {time.perf_counter()-t_blk:.1f}s (ETA {eta/60:.1f}m)"
                    )

    sum_path = "results/day31_ber_sweep_summary.csv"
    with open(sum_path, "w", newline="", encoding="utf-8") as f_sum:
        w = csv.DictWriter(f_sum, fieldnames=list(summary_rows[0].keys()))
        w.writeheader()
        w.writerows(summary_rows)

    png_path = "results/day31_ber_and_flips.png"
    plot_curves(summary_rows, clean_refs, png_path)

    print(f"\nSaved {raw_path}, {sum_path}, and {png_path}")
    print(f"Total elapsed: {(time.perf_counter() - t_all)/60:.2f} minutes")


if __name__ == "__main__":
    main()