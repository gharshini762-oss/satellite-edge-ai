import os
import sys
import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.dataset import CloudDS, load_norm_stats
from src.models import build_resnet18
from src.train import predict
from src.metrics import auroc, ece, basic

def plot_reliability_diagrams(y, probs_by_seed, eces_by_seed, out_path="results/day18_val_reliability.png", bins=10):
    fig, axes = plt.subplots(1, 3, figsize=(13, 4), sharey=True)
    edges = np.linspace(0, 1, bins + 1)

    for ax, (seed, p) in zip(axes, probs_by_seed.items()):
        bin_confs, bin_accs, bin_counts = [], [], []
        for lo, hi in zip(edges[:-1], edges[1:]):
            m = (p >= lo) & ((p < hi) if hi < 1 else (p <= hi))
            if m.any():
                bin_confs.append(p[m].mean())
                bin_accs.append(y[m].mean())
                bin_counts.append(int(m.sum()))

        ax.plot([0, 1], [0, 1], "k--", alpha=0.6, label="Perfect calibration")
        ax.plot(bin_confs, bin_accs, marker="o", color="tab:blue", linewidth=2, label="Model (10 bins)")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_xlabel("Mean Predicted Probability")
        if seed == 0:
            ax.set_ylabel("Empirical Fraction Cloudy (y=1)")
        ax.set_title(f"Seed {seed} (Val ECE = {eces_by_seed[seed]:.4f})")
        ax.grid(True, alpha=0.3)
        ax.legend(loc="upper left", fontsize=8.5)

    plt.suptitle("Validation Reliability Diagrams (10 Bins, N=535; individual bins have sampling noise)", fontsize=11)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"Saved reliability diagram: {out_path}")

def save_failure_gallery(val_ds, val_man, y, p, out_path="results/day18_val_failure_gallery.png"):
    mean, std = load_norm_stats()
    mean_t = torch.tensor(mean).view(3, 1, 1)
    std_t = torch.tensor(std).view(3, 1, 1)

    pred = (p >= 0.5).astype(int)
    wrong_idx = np.where(pred != y)[0]
    conf_err = np.abs(y[wrong_idx] - p[wrong_idx])
    worst_order = wrong_idx[np.argsort(-conf_err)][:12]

    fig, axes = plt.subplots(3, 4, figsize=(13, 10))
    for ax, i in zip(axes.flat, worst_order):
        x_norm, label = val_ds[int(i)]
        rgb = (x_norm * std_t + mean_t).clamp(0.0, 1.0).permute(1, 2, 0).numpy()
        rgb_vis = np.clip(rgb * 2.5, 0.0, 1.0)
        row_meta = val_man.iloc[int(i)]
        ax.imshow(rgb_vis)
        ax.set_title(
            f"row={int(i)} | y={label}, p={p[i]:.3f}\ncloud={row_meta['cloud_pct']:.1f}%, shad={row_meta['shadow_pct']:.1f}%",
            fontsize=8.5
        )
        ax.axis("off")

    plt.suptitle("Validation Failure Gallery: 12 Worst Errors (Seed 0, Brightened x2.5)", fontsize=12)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"Saved 12-worst failure gallery: {out_path}")

if __name__ == "__main__":
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    mean, std = load_norm_stats()
    val_ds = CloudDS("validation", mean, std, train=False)
    val_loader = DataLoader(val_ds, batch_size=64, shuffle=False, num_workers=0)

    man = pd.read_csv("results/manifest_export.csv")
    val_man = man[man["split"] == "validation"].reset_index(drop=True)

    rows = []
    probs_by_seed = {}
    eces_by_seed = {}

    for seed in [0, 1, 2]:
        m = build_resnet18().to(dev)
        m.load_state_dict(torch.load(f"results/ckpt_seed{seed}.pt", map_location=dev))
        z, yv = predict(m, val_loader, dev)

        assert np.array_equal(val_man["label"].to_numpy(), yv), "Validation manifest label mismatch!"

        p = torch.softmax(torch.tensor(z), dim=1)[:, 1].numpy()
        probs_by_seed[seed] = p

        b = basic(yv, p, thr=0.5)
        auc = auroc(yv, p)
        e = ece(yv, p, bins=10)
        eces_by_seed[seed] = e

        rows.append(dict(seed=seed, acc=b["acc"], precision=b["precision"], recall=b["recall"],
                         f1=b["f1"], auroc=auc, ece=e, tp=b["tp"], fp=b["fp"], fn=b["fn"], tn=b["tn"]))

    df_metrics = pd.DataFrame(rows)
    df_metrics.to_csv("results/day18_val_metrics.csv", index=False)

    print("\n=== Day 18: Validation Metrics per Seed (N=535) ===")
    print(df_metrics.to_string(index=False))

    print("\n=== Day 18: Validation Mean ± Std Across 3 Seeds ===")
    for col in ["acc", "precision", "recall", "f1", "auroc", "ece"]:
        print(f"{col:10s}: {df_metrics[col].mean():.4f} ± {df_metrics[col].std(ddof=1):.4f}")

    plot_reliability_diagrams(yv, probs_by_seed, eces_by_seed)

    bins_def = [
        ("0 to 10%",   (val_man["cloud_pct"] >= 0)  & (val_man["cloud_pct"] < 10)),
        ("10 to 30%",  (val_man["cloud_pct"] >= 10) & (val_man["cloud_pct"] < 30)),
        ("30 to 70%",  (val_man["cloud_pct"] >= 30) & (val_man["cloud_pct"] < 70)),
        ("70 to 90%",  (val_man["cloud_pct"] >= 70) & (val_man["cloud_pct"] < 90)),
        ("90 to 100%", (val_man["cloud_pct"] >= 90) & (val_man["cloud_pct"] <= 100)),
        ("shadow >= 20%", val_man["shadow_pct"] >= 20.0),]

    bin_rows = []
    for name, mask in bins_def:
        mask_np = mask.to_numpy()
        n_patches = int(mask_np.sum())
        if n_patches == 0:
            continue
        errs = [float((((probs_by_seed[s][mask_np] >= 0.5).astype(int)) != yv[mask_np]).mean()) for s in [0, 1, 2]]
        bin_rows.append(dict(
            group=name, count=n_patches,
            err_seed0=round(errs[0], 4), err_seed1=round(errs[1], 4), err_seed2=round(errs[2], 4),
            err_mean=round(float(np.mean(errs)), 4)
        ))

    df_bins = pd.DataFrame(bin_rows)
    df_bins.to_csv("results/day18_val_error_bins.csv", index=False)
    print("\n=== Day 18: Validation Error Rate by Cloud Bin & Heavy Shadow ===")
    print(df_bins.to_string(index=False))

    save_failure_gallery(val_ds, val_man, yv, probs_by_seed[0])