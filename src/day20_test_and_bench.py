import os
import sys
import time
import argparse
from datetime import datetime
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
import pynvml  # type: ignore[import]

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.dataset import CloudDS, load_norm_stats
from src.models import build_resnet18
from src.train import predict
from src.metrics import auroc, ece, basic

def get_gpu_temp_and_vram(handle):
    temp = float(pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU))
    mem = pynvml.nvmlDeviceGetMemoryInfo(handle)
    used_mib = round(mem.used / (1024 ** 2), 1)
    return temp, used_mib

@torch.no_grad()
def benchmark_cuda_events(model, batch_size, handle, warmup=50, iters=500):
    model.eval()
    x = torch.randn(batch_size, 3, 224, 224, device="cuda", dtype=torch.float32)

    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    temp_start, _ = get_gpu_temp_and_vram(handle)

    for _ in range(warmup):
        _ = model(x)
    torch.cuda.synchronize()

    starts = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    ends = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]

    for i in range(iters):
        starts[i].record()
        _ = model(x)
        ends[i].record()
    torch.cuda.synchronize()

    times_ms = [s.elapsed_time(e) for s, e in zip(starts, ends)]
    peak_alloc_mib = round(torch.cuda.max_memory_allocated() / (1024 ** 2), 2)
    _, pynvml_used_mib = get_gpu_temp_and_vram(handle)

    return dict(
        temp_start_c=temp_start,
        median_ms=round(float(np.median(times_ms)), 4),
        p95_ms=round(float(np.percentile(times_ms, 95)), 4),
        peak_vram_mb=peak_alloc_mib,
        pynvml_vram_mb=pynvml_used_mib,
    )

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--power-mode", type=str, default="Balanced")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--cooldown", type=int, default=60)
    parser.add_argument("--plugged-in", type=bool, default=True)
    args = parser.parse_args()

    # Strict FP32 evaluation and benchmarking (TF32 disabled)
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    mean, std = load_norm_stats()
    val_ds = CloudDS("validation", mean, std, train=False)
    test_ds = CloudDS("test", mean, std, train=False)

    val_loader = DataLoader(val_ds, batch_size=64, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=64, shuffle=False, num_workers=0)

    man = pd.read_csv("results/manifest_export.csv")
    val_man = man[man["split"] == "validation"].reset_index(drop=True)
    test_man = man[man["split"] == "test"].reset_index(drop=True)

    print("=== Step 1: Evaluating Frozen Seeds Once on Val & Test (TF32=False) ===")
    test_rows = []
    test_metrics_by_seed = {}

    for seed in [0, 1, 2]:
        ckpt_path = f"results/ckpt_seed{seed}.pt"
        m = build_resnet18().to(dev)
        m.load_state_dict(torch.load(ckpt_path, map_location=dev))

        z_val, y_val = predict(m, val_loader, dev)
        z_test, y_test = predict(m, test_loader, dev)

        assert np.array_equal(val_man["label"].to_numpy(), y_val), "Val label mismatch!"
        assert np.array_equal(test_man["label"].to_numpy(), y_test), "Test label mismatch!"

        np.save(f"results/val_logits_fp32_seed{seed}.npy", z_val)
        np.save(f"results/test_logits_fp32_seed{seed}.npy", z_test)

        p_test = torch.softmax(torch.tensor(z_test), dim=1)[:, 1].numpy()
        b = basic(y_test, p_test, thr=0.5)
        auc = auroc(y_test, p_test)
        e = ece(y_test, p_test, bins=10)

        row_m = dict(
            seed=seed,
            test_acc=round(b["acc"], 6),
            test_precision=round(b["precision"], 6),
            test_recall=round(b["recall"], 6),
            test_f1=round(b["f1"], 6),
            test_auroc=round(auc, 6),
            test_ece=round(e, 6),
            tp=b["tp"], fp=b["fp"], fn=b["fn"], tn=b["tn"],
        )
        test_rows.append(row_m)
        test_metrics_by_seed[seed] = row_m
        print(
            f"Seed {seed} Test (N={len(y_test)}) -> acc={b['acc']:.4f} | "
            f"f1={b['f1']:.4f} | auroc={auc:.4f} | ece={e:.4f} | saved val/test logits"
        )

    df_test = pd.DataFrame(test_rows)
    df_test.to_csv("results/day20_test_metrics.csv", index=False)

    print("\n=== Test Set Mean ± Std Across 3 Seeds (N=975) ===")
    for col in ["test_acc", "test_precision", "test_recall", "test_f1", "test_auroc", "test_ece"]:
        print(f"{col:15s}: {df_test[col].mean():.4f} ± {df_test[col].std(ddof=1):.4f}")

    # Step 2: Strict FP32 GPU Benchmark on Seed 0
    pynvml.nvmlInit()
    handle = pynvml.nvmlDeviceGetHandleByIndex(0)

    m_bench = build_resnet18().to(dev)
    m_bench.load_state_dict(torch.load("results/ckpt_seed0.pt", map_location=dev))
    m_bench.eval()

    n_params = sum(p.numel() for p in m_bench.parameters())
    size_mb = round(os.path.getsize("results/ckpt_seed0.pt") / (1024 ** 2), 2)
    today_str = datetime.now().strftime("%Y-%m-%d")

    print(f"\n=== Step 2: Strict FP32 GPU Benchmark (params={n_params:,}, size_mb={size_mb} MiB) ===")
    bench_rows = []

    for batch in [1, 8, 32]:
        for rep in range(1, args.repeats + 1):
            if not (batch == 1 and rep == 1):
                print(f"Resting {args.cooldown}s before batch={batch} repeat {rep}/{args.repeats}...")
                time.sleep(args.cooldown)

            res = benchmark_cuda_events(m_bench, batch, handle, warmup=50, iters=500)
            s0 = test_metrics_by_seed[0]

            row = dict(
                date=today_str,
                model="resnet18",
                precision="fp32",
                seed=0,
                repeat=rep,
                batch=batch,
                median_ms=res["median_ms"],
                p95_ms=res["p95_ms"],
                peak_vram_mb=res["peak_vram_mb"],
                pynvml_vram_mb=res["pynvml_vram_mb"],
                params=n_params,
                size_mb=size_mb,
                temp_start_c=res["temp_start_c"],
                plugged_in=args.plugged_in,
                power_mode=args.power_mode,
                test_acc=s0["test_acc"],
                test_f1=s0["test_f1"],
                test_auroc=s0["test_auroc"],
            )
            bench_rows.append(row)
            print(
                f"batch={batch:2d} | rep={rep} | start={res['temp_start_c']:.0f}C | "
                f"median={res['median_ms']:.3f} ms | p95={res['p95_ms']:.3f} ms | "
                f"peak_alloc={res['peak_vram_mb']:.1f} MiB | pynvml_used={res['pynvml_vram_mb']:.1f} MiB"
            )

    # Append summary rows for seed 1 and seed 2 test metrics with NaN for unbenchmarked latency fields
    for seed in [1, 2]:
        sm = test_metrics_by_seed[seed]
        sz = round(os.path.getsize(f"results/ckpt_seed{seed}.pt") / (1024 ** 2), 2)
        bench_rows.append(dict(
            date=today_str,
            model="resnet18",
            precision="fp32",
            seed=seed,
            repeat=np.nan,
            batch=np.nan,
            median_ms=np.nan,
            p95_ms=np.nan,
            peak_vram_mb=np.nan,
            pynvml_vram_mb=np.nan,
            params=n_params,
            size_mb=sz,
            temp_start_c=np.nan,
            plugged_in=args.plugged_in,
            power_mode=args.power_mode,
            test_acc=sm["test_acc"],
            test_f1=sm["test_f1"],
            test_auroc=sm["test_auroc"],
        ))

    pynvml.nvmlShutdown()
    df_bench = pd.DataFrame(bench_rows)
    df_bench.to_csv("results/baseline_results.csv", index=False)
    print("\nSaved benchmark table: results/baseline_results.csv")