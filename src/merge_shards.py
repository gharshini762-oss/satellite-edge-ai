# src/merge_shards.py
import glob
import json
import numpy as np
import pandas as pd

export_df = pd.read_csv("results/manifest_export.csv")
updated_splits = []

for split in ["test", "validation", "train"]:
    files = sorted(glob.glob(f"data/processed/shards/{split}_*.npz"))
    assert len(files) > 0, f"No shards found for {split} in data/processed/shards/"
    parts = [np.load(f) for f in files]

    X = np.concatenate([p["X"] for p in parts], axis=0)
    y = np.concatenate([p["y"] for p in parts], axis=0).astype(np.int64)
    idx = np.concatenate([p["idx"] for p in parts], axis=0)

    sub_df = export_df[export_df["split"] == split].reset_index(drop=True).copy()
    assert len(X) == len(sub_df), f"Length mismatch in {split}: {len(X)} vs {len(sub_df)}"
    assert X.shape[1:] == (3, 224, 224) and X.dtype == np.int16, f"Unexpected shape/dtype: {X.shape}, {X.dtype}"
    assert np.array_equal(idx, sub_df["idx"].to_numpy()), f"idx alignment mismatch in {split}!"
    assert np.array_equal(y, sub_df["label"].to_numpy()), f"label alignment mismatch in {split}!"

    # Verify v2 position and mask_cloud_pct fields
    if "pos" in parts[0] and "mask_cloud_pct" in parts[0]:
        pos = np.concatenate([p["pos"] for p in parts], axis=0)
        mask_pct = np.concatenate([p["mask_cloud_pct"] for p in parts], axis=0)
        sub_df["pos"] = pos
        sub_df["mask_cloud_pct"] = np.round(mask_pct, 4)
        abs_diff = np.abs(mask_pct - sub_df["cloud_pct"].to_numpy())
        assert abs_diff.max() <= 3.0, f"Mask diff > 3% in {split}: {abs_diff.max():.2f}%"
        diff_str = f"mean_diff={abs_diff.mean():.2f}%, max_diff={abs_diff.max():.2f}%"
    else:
        diff_str = "WARNING: v1 shard detected (no mask_cloud_pct field)"

    updated_splits.append(sub_df)
    np.save(f"data/processed/{split}_X.npy", X)
    np.save(f"data/processed/{split}_y.npy", y)
    print(f"{split:10s} | shape: {str(X.shape):18s} | cloudy: {y.mean():.6f} | {diff_str}")

# Save manifest_export.csv with verified pos and mask_cloud_pct columns
pd.concat(updated_splits, ignore_index=True).to_csv("results/manifest_export.csv", index=False)

# Compute per-channel mean and std from v2 training data ONLY and save to results/norm_stats.json
X_tr = np.load("data/processed/train_X.npy", mmap_mode="r")
s = np.clip(np.asarray(X_tr[::10], dtype=np.float32) / 10000.0, 0.0, 1.0)
mean = s.mean(axis=(0, 2, 3)).round(6).tolist()
std = s.std(axis=(0, 2, 3)).round(6).tolist()

stats = {"mean": mean, "std": std}
with open("results/norm_stats.json", "w") as f:
    json.dump(stats, f, indent=2)

print("\nSaved v2 results/norm_stats.json:", stats)