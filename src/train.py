import os
import sys
import csv
import time
import random
import argparse
import pandas as pd
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.dataset import CloudDS, load_norm_stats
from src.models import build_resnet18

def set_seed(s):
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)
    torch.cuda.manual_seed_all(s)

@torch.no_grad()
def predict(model, loader, dev):
    model.eval()
    zs, ys = [], []
    for x, y in loader:
        zs.append(model(x.to(dev)).float().cpu())
        ys.append(y)
    return torch.cat(zs).numpy(), torch.cat(ys).numpy()  

def nll(logits, y):
    p = torch.softmax(torch.tensor(logits), dim=1)[:, 1].numpy()
    return float(-(y * np.log(p + 1e-7) + (1 - y) * np.log(1 - p + 1e-7)).mean())

def save_training_curve(csv_path, out_png, seed, best_epoch):
    df = pd.read_csv(csv_path)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

    ax1.plot(df["epoch"], df["train_loss"], marker="o", label="Train CE Loss")
    ax1.plot(df["epoch"], df["val_loss"], marker="s", label="Val NLL Loss")
    ax1.axvline(best_epoch, color="gray", linestyle="--", label=f"Best Epoch ({best_epoch})")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.set_title(f"Seed {seed}: Loss Curve")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2.plot(df["epoch"], df["val_acc"], marker="s", color="green", label="Val Accuracy")
    ax2.axvline(best_epoch, color="gray", linestyle="--", label=f"Best Epoch ({best_epoch})")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Validation Accuracy")
    ax2.set_title(f"Seed {seed}: Validation Accuracy")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(out_png, dpi=150)
    plt.close()

def train_one_seed(seed, max_epochs=15, smoke=False):
    set_seed(seed)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    mean, std = load_norm_stats()
    train_ds = CloudDS("train", mean, std, train=True)
    val_ds   = CloudDS("validation", mean, std, train=False)

    g = torch.Generator()
    g.manual_seed(seed)
    train_loader = DataLoader(train_ds, batch_size=64, shuffle=True, num_workers=0, generator=g)
    val_loader   = DataLoader(val_ds, batch_size=64, shuffle=False, num_workers=0)

    model = build_resnet18().to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)

    tag = f"smoke_seed{seed}" if smoke else f"seed{seed}"
    csv_path = f"results/train_log_{tag}.csv"
    ckpt_path = f"results/ckpt_{tag}.pt"

    best, best_epoch, best_acc, patience = 1e9, -1, 0.0, 0
    print(f"\n=== Training {tag} (max_epochs={max_epochs}, dev={dev}) ===")

    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "train_loss", "val_loss", "val_acc", "seconds"])

        for epoch in range(max_epochs):
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            t0 = time.perf_counter()

            model.train()
            running_loss, n_seen = 0.0, 0
            for x, y in train_loader:
                x, y = x.to(dev), y.to(dev)
                opt.zero_grad(set_to_none=True)
                logits = model(x)
                loss = F.cross_entropy(logits, y)
                loss.backward()
                opt.step()
                running_loss += loss.item() * len(y)
                n_seen += len(y)

            train_loss = running_loss / n_seen
            z, yv = predict(model, val_loader, dev)
            v = nll(z, yv)
            acc = float(((z[:, 1] > z[:, 0]) == yv).mean())

            if torch.cuda.is_available():
                torch.cuda.synchronize()
            elapsed = time.perf_counter() - t0

            writer.writerow([epoch, f"{train_loss:.6f}", f"{v:.6f}", f"{acc:.6f}", f"{elapsed:.2f}"])
            f.flush()

            status = ""
            if v < best - 1e-4:
                best, best_epoch, best_acc, patience = v, epoch, acc, 0
                torch.save(model.state_dict(), ckpt_path)
                status = " [saved best]"
            else:
                patience += 1
                status = f" [patience {patience}/3]"

            print(f"epoch {epoch:2d} | train_loss={train_loss:.4f} | val_loss={v:.4f} | val_acc={acc:.4f} | {elapsed:.2f}s{status}")

            if patience >= 3:
                print(f"Early stopping triggered at epoch {epoch} (best epoch={best_epoch}, best_val_loss={best:.4f}, best_val_acc={best_acc:.4f})")
                break

    if not smoke:
        curve_png = f"results/curve_seed{seed}.png"
        save_training_curve(csv_path, curve_png, seed, best_epoch)
        print(f"Saved training curve: {curve_png}")

    return dict(seed=seed, best_epoch=best_epoch, stop_epoch=epoch, best_val_loss=best, best_val_acc=best_acc)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, nargs="+", default=[0])
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()

    summaries = []
    for s in args.seeds:
        summaries.append(train_one_seed(s, max_epochs=args.epochs, smoke=args.smoke))

    print("\n=== Summary ===")
    for r in summaries:
        print(f"Seed {r['seed']} | Best Epoch: {r['best_epoch']} (Stopped: {r['stop_epoch']}) | Best Val Loss: {r['best_val_loss']:.4f} | Best Val Acc: {r['best_val_acc']:.4f}")