import os
import sys
import pandas as pd
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.dataset import CloudDS, load_norm_stats
from src.models import build_resnet18

if __name__ == "__main__":
    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)

    mean, std = load_norm_stats()
    train_ds = CloudDS("train", mean, std, train=True)
    train_loader = DataLoader(train_ds, batch_size=64, shuffle=True, num_workers=0)

    x, y = next(iter(train_loader))
    x, y = x.cuda(), y.cuda()

    m = build_resnet18().cuda()
    m.train()
    opt = torch.optim.AdamW(m.parameters(), lr=1e-3)

    torch.cuda.reset_peak_memory_stats()
    print("=== Day 15: 150-Step Single-Batch Overfit Test (batch_size=64, lr=1e-3) ===")
    for step in range(150):
        opt.zero_grad()
        out = m(x)
        loss = torch.nn.functional.cross_entropy(out, y)
        loss.backward()
        opt.step()

        if step == 0:
            mem_step1_bytes = torch.cuda.max_memory_allocated()
            print(f"After 1st training step -> max_memory_allocated: {mem_step1_bytes:,} bytes ({mem_step1_bytes / (1024**2):.2f} MiB)")

        if step % 25 == 0 or step == 149:
            with torch.no_grad():
                acc = (out.argmax(dim=1) == y).float().mean().item()
            print(f"step {step:3d} | loss: {loss.item():.6f} | batch_acc: {acc:.4f}")