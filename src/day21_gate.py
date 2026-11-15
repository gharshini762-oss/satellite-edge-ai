# src/day21_gate.py
import os
import sys
import time
import hashlib
import pandas as pd
import numpy as np
import torch
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.metrics import auroc, ece, basic

def softmax_pos(logits):
    e = np.exp(logits - logits.max(axis=1, keepdims=True))
    return (e / e.sum(axis=1, keepdims=True))[:, 1]

def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest().upper()

def plot_seed0_test_diagnostics(y_test, p_seed0, out_path="results/day21_test_cm_and_reliability.png", bins=10):
    b = basic(y_test, p_seed0, thr=0.5)
    cm = np.array([[b["tn"], b["fp"]], [b["fn"], b["tp"]]])
    test_ece = ece(y_test, p_seed0, bins=bins)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.5, 4.5))
    im = ax1.imshow(cm, cmap="Blues")
    ax1.set_xticks([0, 1]); ax1.set_yticks([0, 1])
    ax1.set_xticklabels(["Keep (pred=0)", "Cloudy (pred=1)"])
    ax1.set_yticklabels(["Keep (y=0)", "Cloudy (y=1)"])
    ax1.set_xlabel("Predicted Label (thr = 0.5)")
    ax1.set_ylabel("True Label (cloud_pct >= 70%)")
    ax1.set_title(f"Seed 0 Test Confusion Matrix (N={len(y_test)})\nAcc={b['acc']:.4f}, F1={b['f1']:.4f}")

    for r in range(2):
        for c in range(2):
            val = cm[r, c]
            pct = val / len(y_test) * 100.0
            color = "white" if val > cm.max() / 2 else "black"
            ax1.text(c, r, f"{val}\n({pct:.1f}%)", ha="center", va="center", color=color, fontsize=10, fontweight="bold")

    edges = np.linspace(0, 1, bins + 1)
    bin_confs, bin_accs = [], []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (p_seed0 >= lo) & ((p_seed0 < hi) if hi < 1 else (p_seed0 <= hi))
        if m.any():
            bin_confs.append(float(p_seed0[m].mean()))
            bin_accs.append(float(y_test[m].mean()))

    ax2.plot([0, 1], [0, 1], "k--", alpha=0.6, label="Perfect calibration")
    ax2.plot(bin_confs, bin_accs, marker="o", color="tab:blue", linewidth=2, label="Seed 0 (10 bins)")
    ax2.set_xlim(0, 1); ax2.set_ylim(0, 1)
    ax2.set_xlabel("Mean Predicted Probability")
    ax2.set_ylabel("Empirical Fraction Cloudy (y=1)")
    ax2.set_title(f"Seed 0 Test Reliability Diagram\n10-Bin ECE = {test_ece:.4f}")
    ax2.grid(True, alpha=0.3); ax2.legend(loc="upper left")

    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()

if __name__ == "__main__":
    man = pd.read_csv("results/manifest_export.csv")
    val_man = man[man["split"] == "validation"].reset_index(drop=True)
    y_val = np.load("data/processed/validation_y.npy")
    y_test = np.load("data/processed/test_y.npy")

    # 1. Recompute Validation Metrics from Frozen TF32=False Logits
    val_rows, val_probs = [], {}
    for s in [0, 1, 2]:
        zv = np.load(f"results/val_logits_fp32_seed{s}.npy")
        pv = softmax_pos(zv)
        val_probs[s] = pv
        b = basic(y_val, pv, thr=0.5)
        val_rows.append(dict(
            seed=s, acc=b["acc"], precision=b["precision"], recall=b["recall"],
            f1=b["f1"], auroc=auroc(y_val, pv), ece=ece(y_val, pv, bins=10),
            tp=b["tp"], fp=b["fp"], fn=b["fn"], tn=b["tn"]
        ))
    val_df = pd.DataFrame(val_rows)
    val_df.to_csv("results/day18_val_metrics.csv", index=False)
    print("=== 1. Validation Metrics (Recomputed from TF32=False Saved Logits) ===")
    print(val_df.to_string(index=False))

    # 2. Update day18_val_error_bins.csv without shadow >= 50% (n=2)
    bins_def = [
        ("0 to 10%",   (val_man["cloud_pct"] >= 0)  & (val_man["cloud_pct"] < 10)),
        ("10 to 30%",  (val_man["cloud_pct"] >= 10) & (val_man["cloud_pct"] < 30)),
        ("30 to 70%",  (val_man["cloud_pct"] >= 30) & (val_man["cloud_pct"] < 70)),
        ("70 to 90%",  (val_man["cloud_pct"] >= 70) & (val_man["cloud_pct"] < 90)),
        ("90 to 100%", (val_man["cloud_pct"] >= 90) & (val_man["cloud_pct"] <= 100)),
        ("shadow >= 20%", val_man["shadow_pct"] >= 20.0),
    ]
    bin_rows = []
    for name, mask in bins_def:
        m_np = mask.to_numpy()
        errs = [float((((val_probs[s][m_np] >= 0.5).astype(int)) != y_val[m_np]).mean()) for s in [0, 1, 2]]
        bin_rows.append(dict(
            group=name, count=int(m_np.sum()),
            err_seed0=round(errs[0], 4), err_seed1=round(errs[1], 4), err_seed2=round(errs[2], 4),
            err_mean=round(float(np.mean(errs)), 4)
        ))
    pd.DataFrame(bin_rows).to_csv("results/day18_val_error_bins.csv", index=False)

    # 3. Exact Count of 12 Worst Validation Errors (Seed 0)
    p0_val = val_probs[0]
    pred0_val = (p0_val >= 0.5).astype(int)
    wrong_idx = np.where(pred0_val != y_val)[0]
    conf_err = np.abs(y_val[wrong_idx] - p0_val[wrong_idx])
    worst12 = wrong_idx[np.argsort(-conf_err)][:12]
    w_df = val_man.iloc[worst12].copy()
    w_df["row"] = worst12
    w_df["p"] = np.round(p0_val[worst12], 4)
    w_df["err_type"] = np.where(w_df["label"] == 0, "FP (y=0, pred=1)", "FN (y=1, pred=0)")
    print("\n=== 2. Seed 0 Validation 12 Worst Errors Breakdown ===")
    print(w_df[["row", "label", "p", "err_type", "cloud_pct", "clear_pct", "shadow_pct"]].to_string(index=False))
    n_fp12 = int((w_df["label"] == 0).sum())
    n_fn12 = int((w_df["label"] == 1).sum())
    print(f"Counts: {n_fp12} False Positives (y=0), {n_fn12} False Negatives (y=1)")

    # 4. Combined TF32=False Validation & Test Summary Table
    test_df = pd.read_csv("results/day20_test_metrics.csv")
    metrics_map = [
        ("acc", "acc", "test_acc"),
        ("precision", "precision", "test_precision"),
        ("recall", "recall", "test_recall"),
        ("f1", "f1", "test_f1"),
        ("auroc", "auroc", "test_auroc"),
        ("ece", "ece", "test_ece"),
    ]
    sum_rows = []
    for name, vc, tc in metrics_map:
        sum_rows.append(dict(
            metric=name,
            val_mean=round(float(val_df[vc].mean()), 4),
            val_std=round(float(val_df[vc].std(ddof=1)), 4),
            test_mean=round(float(test_df[tc].mean()), 4),
            test_std=round(float(test_df[tc].std(ddof=1)), 4),
        ))
    sum_df = pd.DataFrame(sum_rows)
    sum_df.to_csv("results/day21_baseline_summary.csv", index=False)
    print("\n=== 3. Day 21 Summary (Both Val and Test from TF32=False Logits) ===")
    print(sum_df.to_string(index=False))

    # 5. Exact 10-Bin Reliability Table for Seed 0 Test Logits
    p0_test = softmax_pos(np.load("results/test_logits_fp32_seed0.npy"))
    plot_seed0_test_diagnostics(y_test, p0_test)
    edges = np.linspace(0, 1, 11)
    print("\n=== 4. Seed 0 Test 10-Bin Reliability Numbers (Computed from Logits) ===")
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (p0_test >= lo) & ((p0_test < hi) if hi < 1 else (p0_test <= hi))
        if m.any():
            print(f"bin [{lo:.1f}, {hi:.1f}] | n={int(m.sum()):3d} | mean_p={p0_test[m].mean():.4f} | emp_cloudy={y_test[m].mean():.4f}")

    # 6. Fix baseline_results.csv (replace batch=0 on seed 1 & 2 rows with NaN)
    base_df = pd.read_csv("results/baseline_results.csv")
    mask_s12 = base_df["seed"].isin([1, 2])
    base_df.loc[mask_s12, ["repeat", "batch"]] = np.nan
    base_df.to_csv("results/baseline_results.csv", index=False)
    print("\n=== 5. Cleaned results/baseline_results.csv (Seed 1 & 2 batch/repeat set to NaN) ===")

    # 7. Telemetry Repeat Rate Analysis (Day 19)
    for fn, cols in [("results/day19_pynvml.csv", ["temp_c", "power_w", "clock_mhz"]),
                     ("results/day19_nvidiasmi.csv", [" temp_c", " power_w", " clock_mhz"])]:
        if os.path.exists(fn):
            tdf = pd.read_csv(fn)
            c_use = [c for c in tdf.columns if "temp" in c or "power" in c or "clock" in c]
            dups = (tdf[c_use].iloc[1:].to_numpy() == tdf[c_use].iloc[:-1].to_numpy()).all(axis=1)
            n_rep = int(dups.sum())
            n_tot = len(dups)
            rep_frac = n_rep / n_tot
            est_hz = 5.0 * (1.0 - rep_frac)
            print(f"Telemetry {fn}: {n_rep}/{n_tot} repeats ({rep_frac*100:.1f}%), ~1 every {n_tot/max(n_rep,1):.1f} rows -> sensor update ~{est_hz:.2f} Hz (~{1000/est_hz:.0f} ms)")

    # 8. Unsynchronized vs Synchronized perf_counter() Matmul Timing (Day 19 fix)
    if torch.cuda.is_available():
        a = torch.randn(4096, 4096, device="cuda")
        b = torch.randn(4096, 4096, device="cuda")
        for _ in range(10):
            _ = a @ b
        torch.cuda.synchronize()
        unsync_t = []
        for _ in range(100):
            t0 = time.perf_counter()
            _ = a @ b
            unsync_t.append((time.perf_counter() - t0) * 1000.0)
        torch.cuda.synchronize()
        print(f"\n=== 6. Day 19 Unsynchronized Dispatch Time (perf_counter, 4096x4096): median={np.median(unsync_t):.4f} ms (p95={np.percentile(unsync_t, 95):.4f} ms) ===")

    # 9. Phase 4 Carry-Forward Data Diagnostics (Label Flips & Clipped Pixels)
    flips = ((man["mask_cloud_pct"] >= 70.0) != (man["cloud_pct"] >= 70.0))
    print(f"\n=== 7. Phase 4 Data Diagnostics ===")
    print(f"Label flips at T=70% (mask_cloud_pct vs metadata cloud_pct): {int(flips.sum())} / {len(man)} ({flips.mean()*100:.2f}%)")
    for sp, sdf in man.groupby("split"):
        sp_flips = ((sdf["mask_cloud_pct"] >= 70.0) != (sdf["cloud_pct"] >= 70.0))
        print(f"  {sp:10s}: {int(sp_flips.sum())} / {len(sdf)} ({sp_flips.mean()*100:.2f}%)")

    X_tr = np.load("data/processed/train_X.npy", mmap_mode="r")
    y_tr = np.load("data/processed/train_y.npy")
    cloudy_X = X_tr[y_tr == 1]
    clip_frac = float((cloudy_X > 10000).mean())
    all_clip_frac = float((X_tr > 10000).mean())
    print(f"Pixels clipped at reflectance 1.0 (DN > 10,000): {clip_frac*100:.2f}% in cloudy train tiles (y=1), {all_clip_frac*100:.2f}% across all train tiles")

    # 10. Checkpoint SHA-256 Hashes
    print("\n=== 8. Checkpoint SHA-256 Hashes ===")
    for s in [0, 1, 2]:
        p = f"results/ckpt_seed{s}.pt"
        print(f"ckpt_seed{s}.pt: {sha256_file(p)}")