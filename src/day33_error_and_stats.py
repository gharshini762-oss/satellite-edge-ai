import os
import sys
import csv
import math
import numpy as np
import torch
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.metrics import auroc, ece, basic
from src.policy import softmax_p
from src.models import build_resnet18
from src.verify_weights import verify_phase4_weights
from src.fault_injection import (
    load_test_bundle,
    get_target_weight_names,
    precompute_int8_cache,
    inject_int8_flips,
    evaluate_state,
)


def mcnemar_exact(a_flag, b_flag):
    """Exact two-sided McNemar binomial test on paired boolean outcomes."""
    b = int((a_flag & (~b_flag)).sum())   # A True, B False
    c = int(((~a_flag) & b_flag).sum())   # A False, B True
    n = b + c
    if n == 0:
        return b, c, 1.0, 0.0
    k = min(b, c)
    p_two = min(1.0, 2.0 * sum(math.comb(n, i) for i in range(k + 1)) / (2 ** n))
    chi2_cc = ((abs(b - c) - 1.0) ** 2 / n) if abs(b - c) > 0 else 0.0
    return b, c, float(p_two), float(chi2_cc)


def cluster_bootstrap_metrics(y, p, roi_ids, B=2000, seed=0):
    """ROI-cluster bootstrap 95% CIs for Accuracy, F1, AUROC, and 10-bin ECE."""
    rng = np.random.default_rng(seed)
    groups = {}
    for idx, rid in enumerate(roi_ids):
        groups.setdefault(rid, []).append(idx)
    keys = np.array(list(groups.keys()))
    group_arrays = [np.array(groups[k], dtype=np.int64) for k in keys]
    n_groups = len(group_arrays)

    accs, f1s, aucs, eces = [], [], [], []
    for _ in range(B):
        picks = rng.integers(0, n_groups, size=n_groups)
        idx = np.concatenate([group_arrays[k] for k in picks])
        ys, ps = y[idx], p[idx]
        if ys.sum() == 0 or ys.sum() == len(ys):
            continue
        b = basic(ys, ps, thr=0.5)
        accs.append(b["acc"])
        f1s.append(b["f1"])
        aucs.append(auroc(ys, ps))
        eces.append(ece(ys, ps, bins=10))

    def ci(arr):
        lo, hi = np.percentile(arr, [2.5, 97.5])
        return round(float(lo), 4), round(float(hi), 4)

    pt_b = basic(y, p, thr=0.5)
    return {
        "acc": round(float(pt_b["acc"]), 4),
        "acc_ci95_lo": ci(accs)[0],
        "acc_ci95_hi": ci(accs)[1],
        "f1": round(float(pt_b["f1"]), 4),
        "f1_ci95_lo": ci(f1s)[0],
        "f1_ci95_hi": ci(f1s)[1],
        "auroc": round(float(auroc(y, p)), 4),
        "auroc_ci95_lo": ci(aucs)[0],
        "auroc_ci95_hi": ci(aucs)[1],
        "ece": round(float(ece(y, p, bins=10)), 4),
        "ece_ci95_lo": ci(eces)[0],
        "ece_ci95_hi": ci(eces)[1],
    }


def to_display_rgb(tile_chw):
    """Convert (3, 224, 224) int16 reflectance/10000 to gamma-adjusted RGB [0,1]."""
    rgb = np.clip(tile_chw.astype(np.float32) / 10000.0, 0.0, 1.0).transpose(1, 2, 0)
    return np.power(rgb, 1.0 / 2.2)


def plot_failure_gallery(X_raw, worst_fd_idx, worst_leak_idx, meta, p32, p_int8, out_png):
    fig, axes = plt.subplots(2, 6, figsize=(15.0, 6.8))

    for col, idx in enumerate(worst_fd_idx[:6]):
        ax = axes[0, col]
        ax.imshow(to_display_rgb(X_raw[idx]))
        ax.set_xticks([])
        ax.set_yticks([])
        m = meta[idx]
        ax.set_title(
            f"FD #{col+1} (idx {idx})\n"
            f"Cld {m['cloud_pct']:.1f}% | Shd {m['shadow_pct']:.1f}%\n"
            f"FP32 p={p32[idx]:.3f} | I8={p_int8[idx]:.3f}",
            fontsize=8.0,
        )

    for col, idx in enumerate(worst_leak_idx[:6]):
        ax = axes[1, col]
        ax.imshow(to_display_rgb(X_raw[idx]))
        ax.set_xticks([])
        ax.set_yticks([])
        m = meta[idx]
        ax.set_title(
            f"Leak #{col+1} (idx {idx})\n"
            f"Cld {m['cloud_pct']:.1f}% | Shd {m['shadow_pct']:.1f}%\n"
            f"FP32 p={p32[idx]:.3f} | I8={p_int8[idx]:.3f}",
            fontsize=8.0,
        )

    axes[0, 0].set_ylabel("Worst False Deletions\n(True Keep, y=0)", fontsize=9.5, fontweight="bold")
    axes[1, 0].set_ylabel("Worst Cloud Leaks\n(True Cloudy, y=1)", fontsize=9.5, fontweight="bold")

    fig.suptitle(
        "Test-Set Failure Gallery (Seed 0 FP32 vs. CPU ORT INT8; 224×224 RGB, γ=2.2 Display Stretch)",
        fontsize=12.0,
        y=0.99,
    )
    fig.subplots_adjust(hspace=0.38, wspace=0.08, top=0.87)
    plt.savefig(out_png, dpi=200, bbox_inches="tight")
    plt.close()


def main():
    verify_phase4_weights(["results/ckpt_seed0.pt"])
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False

    X_raw = np.load("data/processed/test_X.npy", mmap_mode="r")
    y_test = np.load("data/processed/test_y.npy").astype(np.int64)

    with open("results/manifest_export.csv", encoding="utf-8") as f:
        test_rows = [r for r in csv.DictReader(f) if r["split"] == "test"]
    assert len(test_rows) == len(y_test)
    assert np.array_equal(np.array([int(r["label"]) for r in test_rows]), y_test)

    meta = []
    for r in test_rows:
        meta.append({
            "roi_id": r["roi_id"],
            "cloud_pct": float(r["cloud_pct"]),
            "clear_pct": float(r["clear_pct"]),
            "shadow_pct": float(r.get("shadow_pct", 0.0)),
            "label": int(r["label"]),
            "orig_idx": r.get("idx", ""),
        })

    cloud_pct = np.array([m["cloud_pct"] for m in meta])
    shadow_pct = np.array([m["shadow_pct"] for m in meta])
    roi_ids = np.array([m["roi_id"] for m in meta])
    print(f"Loaded {len(y_test)} test tiles across {len(set(roi_ids))} unique ROIs.")

    # Load saved Phase 4 test probabilities
    probs = {}
    for prec in ["fp32", "fp16", "int8"]:
        for s in [0, 1, 2]:
            z = np.load(f"results/test_logits_{prec}_seed{s}.npy")
            probs[(prec, s)] = softmax_p(z)

    # Regenerate clean Phase 5 Sim-INT8 Seed-0 test logits for McNemar comparison
    X_gpu, y_np, cp_np, cl_np, _ = load_test_bundle("cuda")
    model = build_resnet18().cuda().eval()
    clean_state = torch.load("results/ckpt_seed0.pt", map_location="cpu")
    target_names = get_target_weight_names(model)
    int8_cache = precompute_int8_cache(clean_state, target_names)
    st_sim, _ = inject_int8_flips(clean_state, int8_cache, target_names, np.random.default_rng(0), n_flips=0)
    res_sim = evaluate_state(model, st_sim, X_gpu, y_np, cp_np, cl_np)
    probs[("sim_int8", 0)] = softmax_p(res_sim["logits"])
    del model, X_gpu
    torch.cuda.empty_cache()

    # ==========================================================
    # 5E. Failure Gallery & Test Error Breakdown by Cloud/Shadow
    # ==========================================================
    p32_s0 = probs[("fp32", 0)]
    pi8_s0 = probs[("int8", 0)]

    keep_idx = np.where(y_test == 0)[0]
    cloud_idx = np.where(y_test == 1)[0]

    worst_fd_idx = keep_idx[np.argsort(-p32_s0[keep_idx])]       # highest p_cloudy among y=0
    worst_leak_idx = cloud_idx[np.argsort(p32_s0[cloud_idx])]    # lowest p_cloudy among y=1

    plot_failure_gallery(X_raw, worst_fd_idx, worst_leak_idx, meta, p32_s0, pi8_s0, "results/day33_failure_gallery.png")

    fail_rows = []
    for rank, idx in enumerate(worst_fd_idx[:15], 1):
        m = meta[idx]
        fail_rows.append({
            "type": "false_deletion",
            "rank": rank,
            "test_idx": int(idx),
            "roi_id": m["roi_id"],
            "cloud_pct": round(m["cloud_pct"], 2),
            "clear_pct": round(m["clear_pct"], 2),
            "shadow_pct": round(m["shadow_pct"], 2),
            "p_fp32": round(float(p32_s0[idx]), 4),
            "p_ort_int8": round(float(pi8_s0[idx]), 4),
            "fp32_disc_08": bool(p32_s0[idx] >= 0.8),
            "int8_disc_08": bool(pi8_s0[idx] >= 0.8),
        })
    for rank, idx in enumerate(worst_leak_idx[:15], 1):
        m = meta[idx]
        fail_rows.append({
            "type": "cloud_leak",
            "rank": rank,
            "test_idx": int(idx),
            "roi_id": m["roi_id"],
            "cloud_pct": round(m["cloud_pct"], 2),
            "clear_pct": round(m["clear_pct"], 2),
            "shadow_pct": round(m["shadow_pct"], 2),
            "p_fp32": round(float(p32_s0[idx]), 4),
            "p_ort_int8": round(float(pi8_s0[idx]), 4),
            "fp32_disc_08": bool(p32_s0[idx] >= 0.8),
            "int8_disc_08": bool(pi8_s0[idx] >= 0.8),
        })
    with open("results/day33_failure_cases.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(fail_rows[0].keys()))
        w.writeheader()
        w.writerows(fail_rows)

    # Test error breakdown by cloud_pct bin & shadow >= 20%
    bin_defs = [
        ("0 to 10%", (cloud_pct >= 0) & (cloud_pct < 10)),
        ("10 to 30%", (cloud_pct >= 10) & (cloud_pct < 30)),
        ("30 to 70%", (cloud_pct >= 30) & (cloud_pct < 70)),
        ("70 to 90%", (cloud_pct >= 70) & (cloud_pct < 90)),
        ("90 to 100%", (cloud_pct >= 90) & (cloud_pct <= 100.1)),
        ("shadow >= 20%", shadow_pct >= 20.0),
    ]
    bin_rows = []
    print("\n=== 5E Test Error Rate by Cloud Bin & Heavy Shadow (3-Seed Mean) ===")
    for name, mask in bin_defs:
        n_m = int(mask.sum())
        e_fp32 = [float(((probs[("fp32", s)][mask] >= 0.5) != y_test[mask]).mean()) for s in [0, 1, 2]]
        e_int8 = [float(((probs[("int8", s)][mask] >= 0.5) != y_test[mask]).mean()) for s in [0, 1, 2]]
        fd08_fp32_s0 = float((probs[("fp32", 0)][mask] >= 0.8).mean())
        row = {
            "group": name,
            "count": n_m,
            "fp32_err_s0": round(e_fp32[0], 4),
            "fp32_err_s1": round(e_fp32[1], 4),
            "fp32_err_s2": round(e_fp32[2], 4),
            "fp32_err_mean": round(float(np.mean(e_fp32)), 4),
            "int8_err_mean": round(float(np.mean(e_int8)), 4),
            "fp32_s0_disc_at_08": round(fd08_fp32_s0, 4),
        }
        bin_rows.append(row)
        print(
            f"  {name:14s} (n={n_m:3d}): FP32 err={row['fp32_err_mean']:.4f} | "
            f"ORT-INT8 err={row['int8_err_mean']:.4f} | FP32 s0 disc@0.8={row['fp32_s0_disc_at_08']:.4f}"
        )

    with open("results/day33_test_error_bins.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(bin_rows[0].keys()))
        w.writeheader()
        w.writerows(bin_rows)

    # Print top-6 worst false deletions & cloud leaks
    print("\n=== 5E Top-6 Worst False Deletions (Seed 0 FP32, True y=0) ===")
    for r in fail_rows[:6]:
        print(
            f"  #{r['rank']} idx={r['test_idx']:3d} roi={r['roi_id']} | "
            f"cld={r['cloud_pct']:5.1f}% clr={r['clear_pct']:5.1f}% shd={r['shadow_pct']:5.1f}% | "
            f"p_fp32={r['p_fp32']:.4f} p_int8={r['p_ort_int8']:.4f}"
        )
    print("\n=== 5E Top-6 Worst Cloud Leaks (Seed 0 FP32, True y=1) ===")
    for r in fail_rows[15:21]:
        print(
            f"  #{r['rank']} idx={r['test_idx']:3d} roi={r['roi_id']} | "
            f"cld={r['cloud_pct']:5.1f}% clr={r['clear_pct']:5.1f}% shd={r['shadow_pct']:5.1f}% | "
            f"p_fp32={r['p_fp32']:.4f} p_int8={r['p_ort_int8']:.4f}"
        )

    # ==========================================================
    # 5F. Cluster Bootstrap 95% CIs (B=2000) & Exact McNemar
    # ==========================================================
    ci_rows = []
    print("\n=== 5F ROI-Cluster Bootstrap 95% CIs (B=2000, Test N=975) ===")
    for prec in ["fp32", "fp16", "int8"]:
        for s in [0, 1, 2]:
            st = cluster_bootstrap_metrics(y_test, probs[(prec, s)], roi_ids, B=2000, seed=42 + s)
            st["precision"] = prec
            st["seed"] = s
            ci_rows.append(st)
            if s == 0:
                print(
                    f"  {prec.upper():4s} s0: Acc={st['acc']:.4f} [{st['acc_ci95_lo']:.4f}, {st['acc_ci95_hi']:.4f}] | "
                    f"F1={st['f1']:.4f} [{st['f1_ci95_lo']:.4f}, {st['f1_ci95_hi']:.4f}] | "
                    f"AUROC={st['auroc']:.4f} [{st['auroc_ci95_lo']:.4f}, {st['auroc_ci95_hi']:.4f}] | "
                    f"ECE={st['ece']:.4f} [{st['ece_ci95_lo']:.4f}, {st['ece_ci95_hi']:.4f}]"
                )
    # Also add Sim-INT8 Seed 0
    st_sim = cluster_bootstrap_metrics(y_test, probs[("sim_int8", 0)], roi_ids, B=2000, seed=42)
    st_sim["precision"] = "sim_int8"
    st_sim["seed"] = 0
    ci_rows.append(st_sim)

    with open("results/day33_bootstrap_ci.csv", "w", newline="", encoding="utf-8") as f:
        fields = ["precision", "seed", "acc", "acc_ci95_lo", "acc_ci95_hi",
                  "f1", "f1_ci95_lo", "f1_ci95_hi", "auroc", "auroc_ci95_lo", "auroc_ci95_hi",
                  "ece", "ece_ci95_lo", "ece_ci95_hi"]
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(ci_rows)

    # Exact McNemar tests
    mc_rows = []
    comparisons = [
        ("acc_thr05_seed0", "fp32", 0, "int8", 0, "acc", 0.5),
        ("acc_thr05_seed1", "fp32", 1, "int8", 1, "acc", 0.5),
        ("acc_thr05_seed2", "fp32", 2, "int8", 2, "acc", 0.5),
        ("acc_thr05_sim_seed0", "fp32", 0, "sim_int8", 0, "acc", 0.5),
        ("acc_thr05_fp16_seed0", "fp32", 0, "fp16", 0, "acc", 0.5),
        ("fd_tau08_keep_seed0", "fp32", 0, "int8", 0, "fd_keep", 0.8),
        ("fd_tau08_keep_sim_seed0", "fp32", 0, "sim_int8", 0, "fd_keep", 0.8),
        ("disc_tau08_all_seed0", "fp32", 0, "int8", 0, "disc_all", 0.8),
    ]

    print("\n=== 5F Exact Two-Sided McNemar Tests (Paired Test Predictions) ===")
    for cname, pa, sa, pb, sb, mode, thr in comparisons:
        pA = probs[(pa, sa)]
        pB = probs[(pb, sb)]
        if mode == "acc":
            fA = (pA >= thr).astype(np.int64) == y_test
            fB = (pB >= thr).astype(np.int64) == y_test
            n_eval = len(y_test)
        elif mode == "fd_keep":
            keep = y_test == 0
            fA = pA[keep] >= thr
            fB = pB[keep] >= thr
            n_eval = int(keep.sum())
        else:
            fA = pA >= thr
            fB = pB >= thr
            n_eval = len(y_test)

        b_cnt, c_cnt, p_val, chi2_cc = mcnemar_exact(fA, fB)
        row = {
            "comparison": cname,
            "model_a": f"{pa}_s{sa}",
            "model_b": f"{pb}_s{sb}",
            "metric": mode,
            "tau": thr,
            "n_eval": n_eval,
            "a_only_b": b_cnt,
            "b_only_c": c_cnt,
            "discordant": b_cnt + c_cnt,
            "exact_p_two_sided": round(p_val, 6),
            "chi2_cc": round(chi2_cc, 4),
        }
        mc_rows.append(row)
        print(
            f"  {cname:24s}: b={b_cnt:2d}, c={c_cnt:2d} (disc={b_cnt+c_cnt:2d}) | "
            f"exact p={p_val:.4f} | chi2_cc={chi2_cc:.3f}"
        )

    with open("results/day33_mcnemar_tests.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(mc_rows[0].keys()))
        w.writeheader()
        w.writerows(mc_rows)

    print("\nSaved results/day33_failure_gallery.png, day33_failure_cases.csv, day33_test_error_bins.csv, day33_bootstrap_ci.csv, day33_mcnemar_tests.csv")


if __name__ == "__main__":
    main()