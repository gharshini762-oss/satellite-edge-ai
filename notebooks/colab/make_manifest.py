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

assert (man.groupby('roi_id')['split'].nunique() == 1).all()
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

import cv2

OUT_V2 = '/content/drive/MyDrive/cloud_export_v2'
os.makedirs(OUT_V2, exist_ok=True)

pos_of = {lab: p for p, lab in enumerate(ds.index)} 

hq_v2 = ds[(ds['label_type'] == 'high') & (ds['real_proj_shape'] == 509)].copy()
hq_v2['cloud_pct'] = hq_v2['thick_percentage'] + hq_v2['thin_percentage']
hq_v2['label'] = (hq_v2['cloud_pct'] >= 70.0).astype(int)
hq_v2['idx'] = hq_v2.index
hq_v2['pos'] = [pos_of[lab] for lab in hq_v2.index]

man_v2 = hq_v2.rename(columns={
    'tortilla:data_split': 'split',
    'clear_percentage': 'clear_pct',
    'cloud_shadow_percentage': 'shadow_pct'
})[['roi_id', 'split', 'cloud_pct', 'clear_pct', 'shadow_pct', 'label', 'idx', 'pos']].copy()

assert (man_v2.groupby('roi_id')['split'].nunique() == 1).all()
man_v2.to_csv('manifest_all.csv', index=False)

N_TRAIN = 4000
train_v2 = man_v2[man_v2['split'] == 'train'].copy()
train_v2['bin'] = (train_v2['cloud_pct'] // 10).clip(upper=9)
sub_v2 = train_v2.groupby('bin', group_keys=False).apply(
    lambda g: g.sample(frac=min(1, N_TRAIN / len(train_v2)), random_state=42)
)
export_v2 = pd.concat([
    man_v2[man_v2['split'] == 'test'],
    man_v2[man_v2['split'] == 'validation'],
    sub_v2.drop(columns='bin')
]).reset_index(drop=True)
export_v2.to_csv('manifest_export.csv', index=False)

RGB_IDX = [4, 3, 2]

def process_and_verify_v2(pos, expected_cloud_pct):
    s = ds.read(int(pos))
    with rio.open(s.read(0)) as a:
        img = a.read(RGB_IDX)[:, :509, :509].astype('float32')
    with rio.open(s.read(1)) as b:
        m = b.read(1)[:509, :509]
    f = np.bincount(m.ravel(), minlength=4) / m.size * 100.0
    mask_cloud_pct = float(f[1] + f[2])
    diff = abs(mask_cloud_pct - float(expected_cloud_pct))
    assert diff <= 3.0, f"Mask mismatch at pos={pos}: mask={mask_cloud_pct:.2f}% vs col={expected_cloud_pct:.2f}%"
    
    img_224 = cv2.resize(img.transpose(1, 2, 0), (224, 224), interpolation=cv2.INTER_AREA)
    return np.rint(img_224.transpose(2, 0, 1)).astype(np.int16), np.float32(mask_cloud_pct)

SHARD = 500
for split in ['test', 'validation', 'train']:
    rows = export_v2[export_v2['split'] == split].reset_index(drop=True)
    for k in range(0, len(rows), SHARD):
        path = f"{OUT_V2}/{split}_{k // SHARD:03d}.npz"
        if os.path.exists(path): continue
        chunk = rows.iloc[k:k + SHARD]
        out_pairs = [process_and_verify_v2(p, c) for p, c in zip(chunk['pos'], chunk['cloud_pct'])]
        X = np.stack([pair[0] for pair in out_pairs])
        mask_pct = np.array([pair[1] for pair in out_pairs], dtype=np.float32)
        np.savez(
            path,
            X=X, y=chunk['label'].to_numpy(),
            idx=chunk['idx'].to_numpy(), pos=chunk['pos'].to_numpy(),
            col_cloud_pct=chunk['cloud_pct'].to_numpy(dtype=np.float32),
            mask_cloud_pct=mask_pct
        )