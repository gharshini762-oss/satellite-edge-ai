import os
import csv
import numpy as np
import matplotlib.pyplot as plt


def softmax_p(z):
    e = np.exp(z - z.max(axis=1, keepdims=True))
    return (e / e.sum(axis=1, keepdims=True))[:, 1]


def policy(p, cloud_pct, clear_pct, tau, T=70.0):
    discard = p >= tau
    keep_worthy = cloud_pct < T
    n_keep = max(int(keep_worthy.sum()), 1)
    n_cloud = max(int((~keep_worthy).sum()), 1)
    return {
        "tau": tau,
        "bandwidth_saved": float(discard.mean()),
        "false_deletion": float((discard & keep_worthy).sum() / n_keep),
        "cloud_leakage": float(((~discard) & (~keep_worthy)).sum() / n_cloud),
        "clear_ground_lost": float(clear_pct[discard].sum() / max(clear_pct.sum(), 1e-9)),
    }


def load_split_manifest(split_name):
    manifest_path = "results/manifest_export.csv" if os.path.exists("results/manifest_export.csv") else "manifest_export.csv"
    rows = []
    with open(manifest_path, "r", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            if r["split"] == split_name:
                rows.append(r)
    roi_id = np.array([r["roi_id"] for r in rows])
    cloud_pct = np.array([float(r["cloud_pct"]) for r in rows])
    clear_pct = np.array([float(r["clear_pct"]) for r in rows])
    return roi_id, cloud_pct, clear_pct


def cluster_ci(roi_id, p, cloud_pct, clear_pct, tau, B=1000, seed=0):
    rng = np.random.default_rng(seed)
    unique_rois = np.unique(roi_id)
    groups = {r: np.where(roi_id == r)[0] for r in unique_rois}
    bw_vals, fd_vals, cg_vals = [], [], []
    for _ in range(B):
        sampled_rois = rng.choice(unique_rois, size=len(unique_rois), replace=True)
        idx = np.concatenate([groups[r] for r in sampled_rois])
        res = policy(p[idx], cloud_pct[idx], clear_pct[idx], tau)
        bw_vals.append(res["bandwidth_saved"])
        fd_vals.append(res["false_deletion"])
        cg_vals.append(res["clear_ground_lost"])
    return {
        "bw_ci": np.percentile(bw_vals, [2.5, 97.5]),
        "fd_ci": np.percentile(fd_vals, [2.5, 97.5]),
        "cg_ci": np.percentile(cg_vals, [2.5, 97.5]),
    }


def main():
    taus = [0.5, 0.8, 0.9, 0.95, 0.99, 0.999]
    val_roi, val_cloud, val_clear = load_split_manifest("validation")
    test_roi, test_cloud, test_clear = load_split_manifest("test")

    specs = [
        ("fp32", "results/val_logits_fp32_seed0.npy", "results/test_logits_fp32_seed0.npy"),
        ("fp16", "results/val_logits_fp16_seed0.npy", "results/test_logits_fp16_seed0.npy"),
        ("int8", "results/val_logits_int8_seed0.npy", "results/test_logits_int8_seed0.npy"),
    ]

    val_rows = []
    chosen = {}

    for prec, v_path, _ in specs:
        pv = softmax_p(np.load(v_path))
        chosen[prec] = {"fd_le_1pct": None, "fd_le_5pct": None}
        for tau in taus:
            res = policy(pv, val_cloud, val_clear, tau)
            val_rows.append({
                "precision": prec,
                "split": "validation",
                "tau": tau,
                "bandwidth_saved": round(res["bandwidth_saved"], 6),
                "false_deletion": round(res["false_deletion"], 6),
                "cloud_leakage": round(res["cloud_leakage"], 6),
                "clear_ground_lost": round(res["clear_ground_lost"], 6),
            })
            # Because false_deletion decreases as tau increases, pick smallest tau (largest bandwidth) meeting the cap
            if res["false_deletion"] <= 0.01 and chosen[prec]["fd_le_1pct"] is None:
                chosen[prec]["fd_le_1pct"] = tau
            if res["false_deletion"] <= 0.05 and chosen[prec]["fd_le_5pct"] is None:
                chosen[prec]["fd_le_5pct"] = tau

    with open("results/day25_val_policy_sweep.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(val_rows[0].keys()))
        writer.writeheader()
        writer.writerows(val_rows)

    print("Chosen validation operating taus:", chosen)

    test_rows = []
    plt.figure(figsize=(8, 5.5))
    styles = {
        "fp32": ("#1f77b4", "o"),
        "fp16": ("#ff7f0e", "s"),
        "int8": ("#2ca02c", "^"),
    }

    for prec, _, t_path in specs:
        pt = softmax_p(np.load(t_path))
        color, marker = styles[prec]
        xs, ys = [], []
        for tau in taus:
            res = policy(pt, test_cloud, test_clear, tau)
            ci = cluster_ci(test_roi, pt, test_cloud, test_clear, tau, B=1000, seed=0)
            test_rows.append({
                "precision": prec,
                "split": "test",
                "tau": tau,
                "chosen_1pct": tau == chosen[prec]["fd_le_1pct"],
                "chosen_5pct": tau == chosen[prec]["fd_le_5pct"],
                "bandwidth_saved": round(res["bandwidth_saved"], 6),
                "bw_ci95_lo": round(float(ci["bw_ci"][0]), 6),
                "bw_ci95_hi": round(float(ci["bw_ci"][1]), 6),
                "false_deletion": round(res["false_deletion"], 6),
                "fd_ci95_lo": round(float(ci["fd_ci"][0]), 6),
                "fd_ci95_hi": round(float(ci["fd_ci"][1]), 6),
                "cloud_leakage": round(res["cloud_leakage"], 6),
                "clear_ground_lost": round(res["clear_ground_lost"], 6),
            })
            xs.append(res["bandwidth_saved"] * 100.0)
            ys.append(res["false_deletion"] * 100.0)

        plt.plot(xs, ys, marker=marker, color=color, linewidth=2, label=prec.upper())

    plt.axhline(1.0, color="crimson", linestyle="--", alpha=0.65, label="1% False-Deletion Cap")
    plt.axhline(5.0, color="darkorange", linestyle=":", alpha=0.75, label="5% False-Deletion Cap")
    plt.xlabel("Bandwidth Saved (% of Tiles Discarded)")
    plt.ylabel("False-Deletion Rate (% Keep-Worthy Tiles Discarded)")
    plt.title("Test Trade-Off: Bandwidth Saved vs. False-Deletion Rate")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig("results/day25_policy_tradeoff.png", dpi=200)

    with open("results/day25_test_policy_ci.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(test_rows[0].keys()))
        writer.writeheader()
        writer.writerows(test_rows)


if __name__ == "__main__":
    main()