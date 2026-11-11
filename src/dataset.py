import os
import json
import pandas as pd 
import torch
import numpy as np

def load_norm_stats(path="results/norm_stats.json"):
    with open(path, "r") as f:
        d = json.load(f)
    return d["mean"], d["std"]

class CloudDS(torch.utils.data.Dataset):
    def __init__(self, split, mean=None, std=None, train=False, root="data/processed", stats_path="results/norm_stats.json"):
        if mean is None or std is None:
            mean, std = load_norm_stats(stats_path)
        self.x = np.load(f"{root}/{split}_X.npy", mmap_mode="r")
        self.y = np.load(f"{root}/{split}_y.npy")
        self.mean = torch.tensor(mean, dtype=torch.float32).view(3, 1, 1)
        self.std = torch.tensor(std, dtype=torch.float32).view(3, 1, 1)
        self.train = train

    def __len__(self):
        return len(self.y)

    def __getitem__(self, i):
        x = torch.from_numpy(self.x[i].astype(np.float32)) / 10000.0
        x = x.clamp(0.0, 1.0)
        if self.train:
            if torch.rand(1) < 0.5:
                x = torch.flip(x, dims=[2])
            if torch.rand(1) < 0.5:
                x = torch.flip(x, dims=[1])
            x = torch.rot90(x, int(torch.randint(0, 4, (1,))), dims=[1, 2])
        return (x - self.mean) / self.std, int(self.y[i])

if __name__ == "__main__":
    mean, std = load_norm_stats()
    for sp in ["train", "validation", "test"]:
        ds = CloudDS(sp, mean, std, train=(sp == "train"))
        x0, y0 = ds[0]
        print(f"{sp:10s} | len={len(ds):4d} | item[0] shape={tuple(x0.shape)} | dtype={x0.dtype} | label={y0}")