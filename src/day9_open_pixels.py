import rasterio
from rasterio.enums import Resampling
import numpy as np
import matplotlib.pyplot as plt

def load_band(path, scale=4):
    """Open a JP2 band and downsample by 4x so it uses minimal RAM."""
    with rasterio.open(path) as src:
        new_h = src.height // scale
        new_w = src.width // scale
        dn = src.read(
            1,
            out_shape=(new_h, new_w),
            resampling=Resampling.bilinear
        ).astype(np.float32)
    return dn

b04_dn = load_band("data/B04.jp2")
b03_dn = load_band("data/B03.jp2")
b02_dn = load_band("data/B02.jp2")

rgb_dn = np.stack([b04_dn, b03_dn, b02_dn], axis=-1)
rgb_toa = (rgb_dn - 1000.0) / 10000.0
rgb_toa = np.clip(rgb_toa, 0.0, 1.0)


p2, p98 = np.percentile(rgb_toa, (2, 98))
rgb_stretched = np.clip((rgb_toa - p2) / (p98 - p2 + 1e-6), 0.0, 1.0)

plt.imsave("data/day9_rgb_preview.png", rgb_stretched)
print(f"Saved preview! Shape: {rgb_stretched.shape}, TOA min/max: {rgb_toa.min():.3f}/{rgb_toa.max():.3f}")
valid = (b04_dn > 0) & (b03_dn > 0) & (b02_dn > 0)   
print("no-data fraction:", 1 - valid.mean())
print("raw DN min/max (valid):", rgb_dn[valid].min(), rgb_dn[valid].max())
p2, p98 = np.percentile(rgb_toa[valid], (2, 98))
rgb_stretched[~valid] = 0
toa_raw = (rgb_dn - 1000.0) / 10000.0
v = toa_raw[valid]
print("fraction < 0:", (v < 0).mean(), "| fraction > 1:", (v > 1).mean(), "| max:", v.max())
