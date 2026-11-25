import os
import sys
import csv
import cv2
import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.policy import softmax_p


def choose_lossless_encoder(X):
    sample = np.clip(X[0].transpose(1, 2, 0), 0, 65535).astype(np.uint16)
    candidates = [
        (".jp2", [int(cv2.IMWRITE_JPEG2000_COMPRESSION_X1000), 1000]),
        (".png", [int(cv2.IMWRITE_PNG_COMPRESSION), 3]),
    ]
    for ext, params in candidates:
        try:
            ok, buf = cv2.imencode(ext, sample, params)
            if not ok or buf is None or len(buf) == 0:
                continue
            decoded = cv2.imdecode(buf, cv2.IMREAD_UNCHANGED)
            if decoded is not None and np.array_equal(decoded, sample):
                return ext, params
            else:
                print(f"Skipping {ext}: round-trip decode was not bit-exact lossless.")
        except Exception as e:
            print(f"Skipping {ext}: {e}")
    raise RuntimeError("No bit-exact 16-bit lossless encoder worked in OpenCV.")


def main():
    X = np.load("data/processed/test_X.npy", mmap_mode="r")
    ext, params = choose_lossless_encoder(X)
    raw_bytes_per_tile = X[0].nbytes
    total_raw_mb = (len(X) * raw_bytes_per_tile) / 1e6

    comp_sizes = np.empty(len(X), dtype=np.int64)
    for i in range(len(X)):
        hwc = np.clip(X[i].transpose(1, 2, 0), 0, 65535).astype(np.uint16)
        ok, buf = cv2.imencode(ext, hwc, params)
        assert ok and len(buf) > 0, f"Encoding failed on tile {i}"
        comp_sizes[i] = len(buf)

    print(f"Verified bit-exact lossless 16-bit encoder: {ext}")
    print(f"Sample tile bytes (first 3): {comp_sizes[:3].tolist()} vs raw {raw_bytes_per_tile}")
    print(f"Total raw: {total_raw_mb:.3f} MB | Compress-only: {comp_sizes.sum()/1e6:.3f} MB "
          f"(ratio={total_raw_mb/(comp_sizes.sum()/1e6):.2f}x)")

    with open("results/day25_test_policy_ci.csv", "r", encoding="utf-8") as f:
        pol_rows = list(csv.DictReader(f))

    out_rows = []
    for prec in ["fp32", "fp16", "int8"]:
        logit_path = (
            "results/test_logits_fp32_seed0.npy"
            if prec == "fp32"
            else f"results/test_logits_{prec}_seed0.npy"
        )
        p_test = softmax_p(np.load(logit_path))

        for cap_col, cap_label in [("chosen_1pct", "val_fd_le_1pct"), ("chosen_5pct", "val_fd_le_5pct")]:
            tau = None
            for r in pol_rows:
                if r["precision"] == prec and r[cap_col] == "True":
                    tau = float(r["tau"])
                    break
            if tau is None:
                continue

            discard = p_test >= tau
            scenarios = [
                ("raw", total_raw_mb),
                ("compress_only", comp_sizes.sum() / 1e6),
                ("filter_only", ((~discard).sum() * raw_bytes_per_tile) / 1e6),
                ("filter_plus_compress", comp_sizes[~discard].sum() / 1e6),
            ]
            for s_name, mb in scenarios:
                out_rows.append({
                    "precision": prec,
                    "operating_cap": cap_label,
                    "tau": tau,
                    "encoder": ext,
                    "scenario": s_name,
                    "megabytes": round(float(mb), 4),
                    "pct_of_raw": round(float(mb / total_raw_mb * 100.0), 2),
                    "kept_tiles": int((~discard).sum()) if "filter" in s_name else len(X),
                })

    with open("results/day27_compression_scenarios.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(out_rows[0].keys()))
        w.writeheader()
        w.writerows(out_rows)

    # Plot 4 bars for FP32 at the strict <=1% validation operating point
    plot_rows = [r for r in out_rows if r["precision"] == "fp32" and r["operating_cap"] == "val_fd_le_1pct"]
    labels = [
        "Raw\n(All Tiles)",
        f"Compress Only\n(Lossless 16-bit {ext})",
        f"Filter Only\n(FP32, tau={plot_rows[0]['tau']})",
        f"Filter + Compress\n(FP32, tau={plot_rows[0]['tau']})",
    ]
    mbs = [r["megabytes"] for r in plot_rows]
    pcts = [r["pct_of_raw"] for r in plot_rows]

    plt.figure(figsize=(8, 5.2))
    bars = plt.bar(labels, mbs, color=["#7f7f7f", "#1f77b4", "#ff7f0e", "#2ca02c"], width=0.55)
    for bar, mb, pct in zip(bars, mbs, pcts):
        plt.text(
            bar.get_x() + bar.get_width() / 2.0,
            bar.get_height() + max(mbs) * 0.02,
            f"{mb:.1f} MB\n({pct:.1f}%)",
            ha="center",
            fontsize=9.5,
        )
    plt.ylim(0, max(mbs) * 1.18)
    plt.ylabel("Downlink Volume Across 975 Test Tiles (MB)")
    plt.title("Four-Scenario Downlink Comparison (224x224 RGB Thumbnails Only)")
    plt.figtext(
        0.5, 0.01,
        "Limitation: 224x224 3-band RGB thumbnails (not full-res 13-band Sentinel-2); only relative percentages apply.",
        ha="center", fontsize=8.5, style="italic"
    )
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout(rect=[0, 0.04, 1, 1])
    plt.savefig("results/day27_compression_bars.png", dpi=200)
    print("Saved results/day27_compression_scenarios.csv and results/day27_compression_bars.png")


if __name__ == "__main__":
    main()