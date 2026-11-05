# Run in Google Colab to generate 224x224 RGB patches
import os, zipfile
import tacoreader
import rasterio as rio
import numpy as np
import pandas as pd
from PIL import Image

ds = tacoreader.load("tacofoundation:cloudsen12-11c")
hq = ds[(ds["label_type"] == "high") & (ds["real_proj_shape"] == 509)].copy()
hq["cloud_pct"] = hq["thick_percentage"] + hq["thin_percentage"]

OUT_DIR = "data/patches_224"
os.makedirs(f"{OUT_DIR}/images", exist_ok=True)

sample_df = hq.groupby("tortilla:data_split", group_keys=False).apply(
    lambda x: x.sample(n=min(len(x), 300), random_state=42)
)

records = []
for idx, row in sample_df.iterrows():
    try:
        s = ds.read(int(idx))
        with rio.open(s.read(0)) as src:
            rgb = src.read([4, 3, 2])[:, :509, :509].astype(np.float32) * 0.0001

        rgb_u8 = np.clip(rgb * 255.0, 0, 255).astype(np.uint8)
        rgb_hwc = np.transpose(rgb_u8, (1, 2, 0)) # (509, 509, 3)
        
        img_224 = Image.fromarray(rgb_hwc).resize((224, 224), Image.BILINEAR)
        
        fname = f"patch_{int(idx):05d}.png"
        img_224.save(f"{OUT_DIR}/images/{fname}")
        
        c_pct = float(row["cloud_pct"])
        sh_pct = float(row.get("shadow_percentage", 0.0))
        
        if sh_pct >= 50.0:
            label, subset_role = 0, "heavy_shadow_eval_only"
        elif c_pct <= 10.0:
            label, subset_role = 0, "clear"
        elif c_pct >= 70.0:
            label, subset_role = 1, "cloudy"
        else:
            label, subset_role = -1, "ambiguous_eval_only"
            
        records.append({
            "patch_id": int(idx),
            "filename": fname,
            "split": row["tortilla:data_split"],
            "roi_id": row.get("roi_id", f"scene_{int(idx)}"),
            "cloud_pct": round(c_pct, 2),
            "thick_pct": round(float(row["thick_percentage"]), 2),
            "thin_pct": round(float(row["thin_percentage"]), 2),
            "shadow_pct": round(sh_pct, 2),
            "label": label,
            "role": subset_role
        })
    except Exception as e:
        print(f"Skipped {idx}: {e}")

meta_df = pd.DataFrame(records)
meta_df.to_csv(f"{OUT_DIR}/patches_metadata.csv", index=False)
print("Saved patches! Breakdown by role:")
print(meta_df["role"].value_counts())

with zipfile.ZipFile("day11_patches_224.zip", "w", zipfile.ZIP_DEFLATED) as zf:
    for root, _, files in os.walk(OUT_DIR):
        for f in files:
            zf.write(os.path.join(root, f))
print("Ready to download: day11_patches_224.zip")