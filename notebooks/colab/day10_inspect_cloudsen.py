import tacoreader

ds = tacoreader.load("hf://datasets/tacofoundation/cloudsen12-11c")
print(f"Total rows in dataset: {len(ds)}")
print(ds.columns.tolist())

import os
import tacoreader
import rasterio as rio
import numpy as np
import matplotlib.pyplot as plt

ds = tacoreader.load("tacofoundation:cloudsen12-11c")
os.makedirs("results", exist_ok=True)

fig, axes = plt.subplots(5, 4, figsize=(12, 15))

for idx in range(5):
   
    s = ds.read(idx)
    
    
    with rio.open(s.read(0)) as a, rio.open(s.read(1)) as b:
       
        img = a.read().astype("float32") * 0.0001 
        lab = b.read(1)
        
        rgb = img[[3, 2, 1]].transpose(1, 2, 0)
        
        p2, p98 = np.percentile(rgb, (2, 98))
        rgb_stretched = np.clip((rgb - p2) / (p98 - p2 + 1e-6), 0.0, 1.0)
        
        # Plot Image
        ax_img = axes[idx, 0]
        ax_img.imshow(rgb_stretched)
        ax_img.set_title(f"Sample {idx} RGB")
        ax_img.axis("off")
        
        # Plot Mask
        ax_lab = axes[idx, 1]
        im = ax_lab.imshow(lab, vmin=0, vmax=3, cmap="viridis")
        ax_lab.set_title(f"Sample {idx} Mask")
        ax_lab.axis("off")

plt.tight_layout()
plt.savefig("results/day10_cloudsen_samples.png", dpi=300)
print("Saved 10-sample visualization grid to results/day10_cloudsen_samples.png")


import tacoreader.v1 as tacoreader
import rasterio as rio
import numpy as np

ds = tacoreader.load("tacofoundation:cloudsen12-l1c")
s = ds.read(0)

with rio.open(s.read(0)) as a, rio.open(s.read(1)) as b:
    img = a.read()
    lab = b.read(1)

    row_sums = np.sum(img, axis=(0, 2))
    zero_rows = np.where(row_sums == 0)[0]
    col_sums = np.sum(img, axis=(0, 1))
    zero_cols = np.where(col_sums == 0)[0]

    print(f"Zero-padded rows count: {len(zero_rows)}")
    print(f"Zero-padded cols count: {len(zero_cols)}")

    if len(zero_rows) > 0:
    
        padding_labels = lab[zero_rows[0], :]
        print(f"Label values in zero-padded strip: {np.unique(padding_labels, return_counts=True)}")


try:
    import tacoreader.v1 as tacoreader
except ModuleNotFoundError:
    import subprocess
    import sys

    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "tacoreader", "rasterio"])
    import tacoreader.v1 as tacoreader

ds = tacoreader.load("tacofoundation:cloudsen12-l1c")
print(ds.columns.tolist())

print(ds["label_type"].value_counts())
print(ds["tortilla:data_split"].value_counts())
print(ds["real_proj_shape"].astype(str).value_counts().head())

cols = ["roi_id", "label_type", "tortilla:data_split", "stac:time_start",
        "clear_percentage", "thick_percentage", "thin_percentage", "cloud_shadow_percentage"]
print(ds[cols].head(5))

n_splits = ds.groupby("roi_id")["tortilla:data_split"].nunique()
print("roi_ids in more than one split:", (n_splits > 1).sum(), "of", len(n_splits))