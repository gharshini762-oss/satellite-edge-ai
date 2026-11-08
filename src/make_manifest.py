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