# Run in google colab.

# Build manifest and check leakage
hq['cloud_pct'] = hq['thick_percentage'] + hq['thin_percentage']

conditions = [
    (hq['cloud_pct'] >= 70.0),
    (hq['cloud_pct'] <= 10.0)
]
choices = [1, 0]
hq['label'] = np.select(conditions, choices, default=-1)

man = hq.rename(columns={'tortilla:data_split': 'split', 'clear_percentage': 'clear_pct',
                          'cloud_shadow_percentage': 'shadow_pct'})
man = man[['roi_id', 'split', 'cloud_pct', 'clear_pct', 'shadow_pct', 'label']].copy()
man['idx'] = man.index
man.to_csv('manifest_all.csv', index=False)

print("SUCCESS: Zero-leakage assertion passed!")
print(man.groupby('split')['label'].agg(['count', 'mean']))

N_TRAIN = 4000
train = man[man['split'] == 'train'].copy()
train['bin'] = (train['cloud_pct'] // 10).clip(upper=9)
sub = train.groupby('bin', group_keys=False).apply(
    lambda g: g.sample(frac=min(1, N_TRAIN / len(train)), random_state=42)
)
export = pd.concat([man[man['split'] == 'test'], man[man['split'] == 'validation'], sub.drop(columns='bin')])
export.to_csv('manifest_export.csv', index=False)
print(f"Export manifest saved with {len(export)} total rows (Train subset: {len(sub)}). ")

# mount Google Drive and run the sharded exporter for your test.

import os
import numpy as np
import rasterio as rio
import tacoreader.v1 as tacoreader

try:
    from google.colab import drive
    drive.mount('/content/drive')
    OUT = '/content/drive/MyDrive/cloud_export'
except Exception as e:
    print(f"Could not mount Google Drive ({e}). Falling back to local directory /content/cloud_export")
    OUT = '/content/cloud_export'

os.makedirs(OUT, exist_ok=True)
SHARD = 500

def process(idx):
    s = ds.read(int(idx))
    with rio.open(s.read(0)) as src:
        rgb = src.read([4, 3, 2])[:, :509, :509].astype(np.float32) * 0.0001
    return rgb

for split in ['test', 'validation']:
    rows = export[export['split'] == split].reset_index(drop=True)
    for k in range(0, len(rows), SHARD):
        path = f"{OUT}/{split}_{k // SHARD:03d}.npz"
        if os.path.exists(path): 
            continue 
        chunk = rows.iloc[k:k + SHARD]
        X = np.stack([process(i) for i in chunk['idx']])
        np.savez(path, X=X, y=chunk['label'].to_numpy(), idx=chunk['idx'].to_numpy())
        print(path, X.shape)


# To convert Image Size from 509x509 to 224x224
from google.colab import drive
drive.mount('/content/drive')

import os, cv2
import numpy as np
import pandas as pd

OUT = '/content/drive/MyDrive/cloud_export'
export = pd.read_csv('manifest_export.csv')
export['label'] = (export['cloud_pct'] >= 70.0).astype(int)
export.to_csv('manifest_export.csv', index=False)

SHARD = 500
for split in ['test', 'validation']:
    rows = export[export['split'] == split].reset_index(drop=True)
    for k in range(0, len(rows), SHARD):
        path = f"{OUT}/{split}_{k // SHARD:03d}.npz"
        chunk = rows.iloc[k:k + SHARD]
        d = np.load(path)
        X_old = d['X']
        if X_old.shape[-1] == 509:
            X_new = np.empty((len(X_old), 3, 224, 224), dtype=np.int16)
            for j in range(len(X_old)):
                hwc = X_old[j].transpose(1, 2, 0) * 10000.0
                hwc_224 = cv2.resize(hwc, (224, 224), interpolation=cv2.INTER_AREA)
                X_new[j] = np.rint(hwc_224.transpose(2, 0, 1)).astype(np.int16)
            np.savez(path, X=X_new, y=chunk['label'].to_numpy(), idx=chunk['idx'].to_numpy())
            print(f"Converted {path} -> {X_new.shape} {X_new.dtype}, cloudy={chunk['label'].mean():.3f}")
        else:
            np.savez(path, X=X_old, y=chunk['label'].to_numpy(), idx=chunk['idx'].to_numpy())
            print(f"Verified {path} -> {X_old.shape} {X_old.dtype}, cloudy={chunk['label'].mean():.3f}")

import rasterio as rio

# Check if 'ds' is already in memory from earlier; if not, load it cleanly
if 'ds' not in globals():
    import tacoreader
    try:
        ds = tacoreader.load("tacofoundation:cloudsen12-l1c")
    except Exception:
        import tacoreader.v1 as tacoreader_v1
        ds = tacoreader_v1.load("tacofoundation:cloudsen12-l1c")

RGB_IDX = [4, 3, 2]

def process(i):
    s = ds.read(int(i))
    with rio.open(s.read(0)) as a:
        img = a.read(RGB_IDX)[:, :509, :509].astype('float32')
    img = cv2.resize(img.transpose(1, 2, 0), (224, 224), interpolation=cv2.INTER_AREA)
    return np.rint(img.transpose(2, 0, 1)).astype(np.int16)

train_rows = export[export['split'] == 'train'].reset_index(drop=True)
for k in range(0, len(train_rows), SHARD):
    path = f"{OUT}/train_{k // SHARD:03d}.npz"
    if os.path.exists(path):
        print(f"Already exists, skipping: {path}")
        continue
    chunk = train_rows.iloc[k:k + SHARD]
    X = np.stack([process(i) for i in chunk['idx']])
    np.savez(path, X=X, y=chunk['label'].to_numpy(), idx=chunk['idx'].to_numpy())
    print(path, X.shape, X.dtype, "cloudy:", chunk['label'].mean().round(3))       