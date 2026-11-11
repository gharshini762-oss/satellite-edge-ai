# src/check_loaders.py
import os
import sys
import time
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.dataset import CloudDS, load_norm_stats

def save_stratified_grid(dataset, manifest_split_df, out_path, title, seed=42):
    mean, std = load_norm_stats()
    mean_t = torch.tensor(mean).view(3, 1, 1)
    std_t = torch.tensor(std).view(3, 1, 1)

    rng = np.random.default_rng(seed)
    idx_0 = np.where(manifest_split_df["label"].to_numpy() == 0)[0]
    idx_1 = np.where(manifest_split_df["label"].to_numpy() == 1)[0]
    pick_0 = rng.choice(idx_0, size=13, replace=False)
    pick_1 = rng.choice(idx_1, size=12, replace=False)
    picks = np.concatenate([pick_1, pick_0])  # Top 12 are y=1, bottom 13 are y=0

    fig, axes = plt.subplots(5, 5, figsize=(12, 12))
    for ax, i in zip(axes.flat, picks):
        x_norm, y = dataset[int(i)]
        rgb = (x_norm * std_t + mean_t).clamp(0.0, 1.0).permute(1, 2, 0).numpy()
        rgb_vis = np.clip(rgb * 2.5, 0.0, 1.0)  # 2.5x display gain for dark TOA reflectance
        cloud_pct = manifest_split_df.iloc[int(i)]["cloud_pct"]
        ax.imshow(rgb_vis)
        ax.set_title(f"row={int(i)} | y={y} ({cloud_pct:.0f}%)", fontsize=8.5)
        ax.axis("off")

    plt.suptitle(f"{title}\n(Stratified sample: 12 y=1, 13 y=0 | Display brightened x2.5)", fontsize=12)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"Saved stratified visual grid: {out_path}")

if __name__ == "__main__":
    torch.manual_seed(0)
    mean, std = load_norm_stats()
    train_ds = CloudDS("train", mean, std, train=True)
    val_ds   = CloudDS("validation", mean, std, train=False)
    test_ds  = CloudDS("test", mean, std, train=False)

    train_loader = DataLoader(train_ds, batch_size=64, shuffle=True, num_workers=0)
    val_loader   = DataLoader(val_ds, batch_size=64, shuffle=False, num_workers=0)
    test_loader  = DataLoader(test_ds, batch_size=64, shuffle=False, num_workers=0)

    x, y = next(iter(train_loader))
    print("=== Batch Check ===")
    print("Shape:", tuple(x.shape), "| Dtype:", x.dtype)
    print("All finite:", bool(torch.isfinite(x).all()))
    print("Label counts (0=keep, 1=cloudy):", torch.bincount(y).tolist(), f"(cloudy frac={y.float().mean():.3f})")
    print("Batch per-channel mean:", x.mean(dim=(0, 2, 3)).round(decimals=3).tolist())
    print("Batch per-channel std :", x.std(dim=(0, 2, 3)).round(decimals=3).tolist())

    manifest = pd.read_csv("results/manifest_export.csv")
    train_man = manifest[manifest["split"] == "train"].reset_index(drop=True)
    val_man   = manifest[manifest["split"] == "validation"].reset_index(drop=True)

    train_ds_eval = CloudDS("train", mean, std, train=False)
    save_stratified_grid(train_ds_eval, train_man, "results/day14_train_grid.png", "Train 5x5 Stratified Grid")
    save_stratified_grid(val_ds, val_man, "results/day14_val_grid.png", "Validation 5x5 Stratified Grid")

    t0 = time.perf_counter()
    n_imgs = sum(len(bx) for bx, _ in train_loader)
    elapsed = time.perf_counter() - t0
    print(f"=== Loader Speed: {n_imgs / elapsed:.1f} images/sec ({elapsed:.2f}s for {n_imgs} images) ===")