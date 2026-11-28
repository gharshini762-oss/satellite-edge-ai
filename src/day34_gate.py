import os
import sys
import csv
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.verify_weights import verify_phase4_weights


def read_csv(path):
    with open(path, encoding="utf-8") as f:
        return list(csv.DictReader(f))


def main():
    verify_phase4_weights()

    master = []

    # 1. 5A Energy & Throughput (RTX 5060, 3 repeats, ddof=1)
    e_rows = read_csv("results/day29_energy_and_throughput.csv")
    for prec in ["fp32", "fp16"]:
        for batch in ["1", "8", "32"]:
            sub = [r for r in e_rows if r["precision"] == prec and r["batch"] == batch]
            pps = [float(r["throughput_pps"]) for r in sub]
            pw = [float(r["mean_power_w"]) for r in sub]
            tot = [float(r["total_energy_mj_per_patch"]) for r in sub]
            net = [float(r["net_energy_mj_per_patch"]) for r in sub]
            master.append({
                "section": "5A_energy",
                "precision": prec,
                "setting": f"gpu_batch={batch}",
                "metric_primary": f"{np.mean(pps):.1f} ± {np.std(pps, ddof=1):.1f} p/s",
                "metric_secondary": f"total {np.mean(tot):.2f} ± {np.std(tot, ddof=1):.2f} mJ/p | net {np.mean(net):.2f} ± {np.std(net, ddof=1):.2f} mJ/p",
                "notes": f"mean power {np.mean(pw):.2f} W (idle {sub[0]['idle_power_w']} W), T={min(int(r['start_temp_c']) for r in sub)}-{max(int(r['end_temp_c']) for r in sub)}C",
            })

    # 2. 5B 10-Minute Sustained Thermal Drift
    t_rows = read_csv("results/day30_thermal_10min.csv")
    for prec in ["fp32", "fp16"]:
        sub = [r for r in t_rows if r["precision"] == prec]
        early = [r for r in sub if float(r["elapsed_s"]) <= 60.0]
        late = [r for r in sub if float(r["elapsed_s"]) >= 540.0]
        e_lat = float(np.mean([float(r["median_ms"]) for r in early]))
        l_lat = float(np.mean([float(r["median_ms"]) for r in late]))
        drift = (l_lat - e_lat) / e_lat * 100.0
        e_pps = float(np.mean([float(r["throughput_pps"]) for r in early]))
        l_pps = float(np.mean([float(r["throughput_pps"]) for r in late]))
        max_t = max(int(r["temp_c"]) for r in sub)
        min_clk = min(int(r["sm_clock_mhz"]) for r in sub)
        peak_clk = max(int(r["sm_clock_mhz"]) for r in sub)
        first_th = next((float(r["elapsed_s"]) for r in sub if int(r["throttle_mask"]) == 32), -1.0)
        master.append({
            "section": "5B_thermal",
            "precision": prec,
            "setting": "gpu_batch=32_10min",
            "metric_primary": f"lat {e_lat:.4f} -> {l_lat:.4f} ms ({drift:+.2f}%)",
            "metric_secondary": f"pps {e_pps:.1f} -> {l_pps:.1f}",
            "notes": f"max {max_t}C, SM {peak_clk}->{min_clk} MHz, mask=32 at t={first_th:.2f}s",
        })

    # 3. 5D Equal-BER & Equal-Flips Sweeps
    s31 = read_csv("results/day31_ber_sweep_summary.csv")
    for r in s31:
        master.append({
            "section": f"5D_{r['mode']}",
            "precision": r["precision"],
            "setting": f"lvl={r['level']} (flips={r['mean_flips']})",
            "metric_primary": f"acc {r['acc_mean']} ± {r['acc_ci95']} (med {r['acc_median']}, min {r['acc_min']})",
            "metric_secondary": f"fd@0.8 {r['fd08_mean']} (max {r['fd08_max']})",
            "notes": f"col<85%={r['collapse_acc_lt_85']}/30, nf={r['non_finite_trials']}/30, catFD>10%={r['catastrophic_fd08_gt_10pct']}/30",
        })

    # 4. 5D Key Bit-Position Results (K=1, K=10 b30 vs b7, K=100 groups)
    s32 = read_csv("results/day32_bit_position_summary.csv")
    keep_32 = {
        ("per_bit_k10", "fp32", "b30"),
        ("per_bit_k10", "fp32", "b29"),
        ("per_bit_k10", "fp32", "b31"),
        ("per_bit_k10", "sim_int8", "b7"),
    }
    for r in s32:
        if r["experiment"] in ("single_flip_k1", "field_groups_k100") or (r["experiment"], r["precision"], r["bit_group"]) in keep_32:
            master.append({
                "section": f"5D_{r['experiment']}",
                "precision": r["precision"],
                "setting": f"{r['bit_group']} (K={r['k_flips']})",
                "metric_primary": f"acc {r['acc_mean']} ± {r['acc_ci95']} (med {r['acc_median']}, min {r['acc_min']})",
                "metric_secondary": f"fd@0.8 {r['fd08_mean']} (max {r['fd08_max']})",
                "notes": f"col<85%={r['collapse_acc_lt_85']}/30, nf={r['non_finite_trials']}/30, catFD>10%={r['catastrophic_fd08_gt_10pct']}/30",
            })

    # 5. 5F Exact McNemar Tests
    mc = read_csv("results/day33_mcnemar_tests.csv")
    for r in mc:
        master.append({
            "section": "5F_mcnemar",
            "precision": f"{r['model_a']}_vs_{r['model_b']}",
            "setting": f"{r['metric']} (tau={r['tau']}, n={r['n_eval']})",
            "metric_primary": f"exact p={r['exact_p_two_sided']}",
            "metric_secondary": f"b={r['a_only_b']}, c={r['b_only_c']} (disc={r['discordant']})",
            "notes": f"chi2_cc={r['chi2_cc']}",
        })

    out_path = "results/day34_phase5_master.csv"
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(master[0].keys()))
        w.writeheader()
        w.writerows(master)
    print(f"Saved {out_path} ({len(master)} summary rows)")

    required = [
        "src/verify_weights.py",
        "src/day29_energy_and_feasibility.py",
        "src/day30_thermal_test.py",
        "src/fault_injection.py",
        "src/day31_ber_sweep.py",
        "src/day32_bit_position.py",
        "src/day33_error_and_stats.py",
        "results/day29_energy_and_throughput.csv",
        "results/day30_thermal_10min.csv",
        "results/day30_thermal_drift.png",
        "results/day31_ber_sweep_raw.csv",
        "results/day31_ber_sweep_summary.csv",
        "results/day31_ber_and_flips.png",
        "results/day32_bit_position_raw.csv",
        "results/day32_bit_position_summary.csv",
        "results/day32_bit_position.png",
        "results/day33_failure_gallery.png",
        "results/day33_failure_cases.csv",
        "results/day33_test_error_bins.csv",
        "results/day33_bootstrap_ci.csv",
        "results/day33_mcnemar_tests.csv",
        "results/day34_phase5_master.csv",
    ]
    missing = [p for p in required if not os.path.exists(p)]
    print(f"Phase 5 deliverable check: {len(required) - len(missing)}/{len(required)} present")
    for p in missing:
        print(f"[MISSING] {p}")


if __name__ == "__main__":
    main()