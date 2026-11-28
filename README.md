# Onboard Cloud Filtering with Quantized Neural Networks

## Goal
Compare FP32, FP16, and INT8 cloud-vs-clear classifiers on Sentinel-2
patches under satellite downlink limits.
Measurements: accuracy, model size, latency, and memory; the trade-off between bandwidth saved and wrongly deleted clear tiles under a high-confidence deletion threshold; and software bit-flip fault injection.
A laptop GPU stands in for onboard hardware, so results are relative comparisons, not in-orbit performance.

## Status (last updated 2026-11-26)
- Phase 1 (environment and tools): complete.
- Phase 2 (data): complete.
- Phase 3 (FP32 baseline): complete.
- Phase 4 (FP16/INT8 quantization): complete.
- Phase 5 (benchmarks and bit-flip tests): complete.
- Phase 6 (IEEE paper draft): in progress.

## Environment
- Device: Acer Nitro ANV15-52
- GPU: NVIDIA GeForce RTX 5060 Laptop GPU, 8 GB VRAM, driver 621.01
- Python 3.12.10
- PyTorch 2.11.0+cu128, CUDA 12.8 build
- Test conditions: `plugged_in=True` and NitroSense `Balanced` mode are self-reported CLI/script settings. Training used PyTorch default precision, whereas evaluation, latency/energy benchmarking, thermal stress, and fault injection enforced strict `TF32=False`.
- CPU: 13th Gen Intel(R) Core(TM) i7-13620H
- Packages: `torch 2.11.0+cu128`, `onnx 1.23.2`, `onnxruntime 1.32.0`, `psutil 7.2.2`, `cv2 5.0.0`, `numpy 2.5.4`

## Data
- Source: CloudSEN12+ v1.1.2, L1C, CC0 (Aybar et al. 2022, Scientific Data,
  doi 10.1038/s41597-022-01878-2; Aybar et al. 2024, Data in Brief,
  doi 10.1016/j.dib.2024.110852).
- Subset: high-quality labels, 509-px patches, official split. Exported
  5,510 patches: train 4,000 (stratified from 8,490), validation 535, test 975.
- Label: cloudy = 1 if thick + thin cloud >= 70%, else 0. The threshold was
  fixed before any model was trained, after viewing the cloud-fraction
  histogram. Shadow counts as ground.
- Input: RGB (B04, B03, B02), 3-px padding cropped, resized 509 to 224
  (area interpolation), value / 10,000.
- Alignment check: mask-derived cloud % vs. metadata cloud %, mean difference
  0.24 to 0.26 points, max 2.61 (train, validation, test).
- The data is not in the repo. Details are in dataset_card.md.

## What's here so far
- `src/verify_env.py`, `day4_image_basics.py`, `day5_gpu_benchmark.py`, `telemetry.py`, `day19_telemetry_compare.py`: Phase 1 environment checks, `time.perf_counter()` GPU timing, and `pynvml` vs. `nvidia-smi` telemetry comparison.
- `src/merge_shards.py`, `dataset.py`, `check_loaders.py`: shard merging, dataset class, and loader checks.
- `src/models.py`, `day15_overfit.py`, `train.py`, `metrics.py`, `eval_val.py`: ResNet-18 definition, 150-step single-batch overfit check, multi-seed training with early stopping, and validation diagnostics.
- `src/day20_test_and_bench.py`, `day21_gate.py`: one-time strict FP32 (`TF32=False`) evaluation, logit saving, GPU batch latency/VRAM benchmark (`warmup=50, iters=500`), and Phase 3 gate summary.
- `src/day22_fp16.py`, `day22_gpu_paired_bench.py`, `day23_onnx_cpu.py`, `day24_int8_ptq.py`, `day24_int8_multiseed.py`, `policy.py`, `day26_calibration.py`, `day27_compression.py`, `day28_gate.py`, `day28_replot_figures.py`: Phase 4 FP16/ONNX/INT8 quantization, same-session paired GPU & CPU benchmarks, threshold policy sweep, temperature scaling, lossless `.jp2` compression scenarios, and gate tables/figures.
- `src/verify_weights.py`, `day29_energy_and_feasibility.py`, `day30_thermal_test.py`, `fault_injection.py`, `day31_ber_sweep.py`, `day32_bit_position.py`, `day33_error_and_stats.py`, `day34_gate.py`: Phase 5 SHA-256 weight lock, GPU energy-per-patch and Sentinel-2 swath feasibility, 10-minute sustained thermal drift test, Monte Carlo bit-flip sweeps (equal BER, equal flip count, and bit-position breakdown), test failure gallery, ROI-cluster bootstrap CIs, exact two-sided McNemar tests, and master gate summary.

## Phase 1 measurements
- GPU timing: synchronized(Measures actual GPU compute time) v/s unsynchronized (Measures CPU handoff only). GPU calls return before the work finishes, so timing without synchronize() undercounts. 19.3, 20.0, and 19.1 ms plugged in, and 28.2 ms on battery.

- Telemetry : clock rose from about 200 MHz to about 1750 MHz; temperature rose from about 54 C to about 78 C. 
-  (`src/day19_telemetry_compare.py`): During a 20 s load in self-reported `Balanced` mode, early temperature/clock oscillations (`0-6 s`) occur as power briefly overshoots to `63.62 W` (`1740 MHz`) before settling at the active `~59.8-60.0 W` P0 limit (`1680-1747 MHz`, `78-81 C`). At `5 Hz` polling, `pynvml` repeats `21/99` consecutive samples (`21.2%`) and `nvidia-smi` repeats `17/97` (`17.5%`), indicating an underlying NVML sensor refresh rate of `~4 Hz` (`~243-254 ms`).

## Phase 2 pipeline & observations          
- Indexing mechanism: Discovered that tacoreader expects 0-based row positions (pos) rather than pandas index labels (idx). Passing labels initially caused a silent image-label mismatch, which was identified through control audits and resolved by enforcing position-based reads with automated semantic mask validation.

- Split integrity: Confirmed strict separation across region-of-interest identifiers (roi_id), (checking that no roi_id is shared between splits) between training, validation, and test subsets.

- Normalization approach: Channel means and standard deviations were computed exclusively from the training split to avoid introducing evaluation bias into validation or test sets.

- Memory mapping: Utilizing memory-mapped array access (mmap_mode='r') avoids loading the full dataset into RAM at once and allowed high data throughput during batch iteration.

## Phase 3 FP32 Baseline Results (ResNet-18, 3 Seeds)
- **Model size:** `11,177,538` parameters (`42.71 MiB` checkpoint).
- **Classification metrics (mean ± std across seeds 0, 1, 2 at threshold 0.5, computed from `TF32=False` saved logits):**
  - Validation (`N = 535`, selected by early stopping on val NLL): Accuracy `0.9364 ± 0.0131`, Precision `0.8600 ± 0.0580`, Recall `0.8519 ± 0.0131`, F1 `0.8549 ± 0.0236`, AUROC `0.9759 ± 0.0002`, 10-bin ECE `0.0283 ± 0.0094`.
  - Test (`N = 975`, evaluated once): Accuracy `0.9241 ± 0.0067`, Precision `0.8236 ± 0.0265`, Recall `0.8007 ± 0.0353`, F1 `0.8114 ± 0.0180`, AUROC `0.9637 ± 0.0047`, 10-bin ECE `0.0243 ± 0.0051`.
- **Error distribution (3-seed validation mean):** Validation errors concentrate in the `70-90%` thin/broken cloud bin (`15.42%`), `90-100%` bin (`13.51%`), tiles with `>= 20%` cloud shadow (`11.69%`), and `30-70%` broken-cloud bin (`8.59%`), vs. `0.89%` on `0-10%` clear sky.
- **Strict FP32 GPU inference benchmark (Seed 0, `TF32=False`, `warmup=50, iters=500`, 3 repeats, 60 s cooldown, self-reported Balanced mode):**
  - `batch = 1`: median `3.8511-4.1518 ms` (p95 `4.3441-4.8872 ms`; rep 1 started warm at `61 C` before cooldown, reps 2-3 at `46-48 C` ran in `3.8511-3.8707 ms`), peak allocated VRAM `144.81 MiB` (`pynvml` `565.1 MiB`).
  - `batch = 8`: median `8.3295-8.3677 ms` (p95 `8.6730-8.7718 ms`, `46-47 C`), peak allocated VRAM `152.85-153.76 MiB` (`pynvml` `617.1-639.1 MiB`).
  - `batch = 32`: median `20.2801-20.3666 ms` (p95 `20.9369-21.0784 ms`, `47-49 C`), peak allocated VRAM `312.13 MiB` (`pynvml` `705.1 MiB`).
  - *(Note: Phase 3 standalone VRAM includes prior test-pass memory overhead; Phase 4's same-session paired benchmark below uses isolated single-model loading and is the canonical FP32-vs-FP16 reference.)*

## Phase 4 Precision, Policy, Calibration & Compression Results (ResNet-18)

All means and standard deviations use sample std (`ddof=1`). Seed 0 uses `seed-0 v2` (`ckpt_seed0.pt`, SHA-256 `B7FDA259...D210`), synced across Phases 3, 4, and 5 (original `baseline-frozen` Seed-0 hash `FC79C352...A504` is preserved in git history and logged in `log.md`).

- **3-seed Test Metrics (`N = 975`, threshold `0.5`, Table II):**
  - FP32 (seeds 0, 1, 2): Accuracy `0.9241 ± 0.0067`, F1 `0.8114 ± 0.0180`, AUROC `0.9637 ± 0.0047`, 10-bin ECE `0.0243 ± 0.0051`.
  - FP16 (seeds 0, 1, 2; `1.0000` prediction agreement with FP32): Accuracy `0.9241 ± 0.0067`, F1 `0.8114 ± 0.0180`, AUROC `0.9637 ± 0.0047`, 10-bin ECE `0.0243 ± 0.0050`.
  - INT8 MinMax (CPU ORT W8A8, seeds 0, 1, 2; `98.67%–99.28%` test agreement): Accuracy `0.9214 ± 0.0016`, F1 `0.8116 ± 0.0069`, AUROC `0.9636 ± 0.0046`, 10-bin ECE `0.0240 ± 0.0063`.
- **Same-Session Paired GPU Benchmark (RTX 5060, `TF32=False`, 3 repeats, 60 s cooldown before every run, isolated model loading, Table III):**
  - Model size: `42.71 → 21.38 MiB` (`2.00x` smaller).
  - `batch=1`: `4.1455 ± 0.1013 → 2.6147 ± 0.0581 ms` (`1.59x` faster); peak VRAM `98.87 → 36.86 MiB` (`62.7%` drop).
  - `batch=8`: `8.6842 ± 0.0571 → 3.1590 ± 0.0267 ms` (`2.75x` faster); peak VRAM `106.91 → 61.66 MiB` (`42.3%` drop).
  - `batch=32`: `20.4855 ± 0.0255 → 10.1905 ± 0.0900 ms` (`2.01x` faster); peak VRAM `266.19 → 151.55 MiB` (`43.1%` drop).
- **CPU ONNX Runtime Benchmark (`i7-13620H`, `onnxruntime 1.32.0`, `batch=1`, Table III):**
  - ONNX size: `42.63 → 10.77 MiB` (`3.96x` smaller).
  - FP32 → INT8 MinMax: `26.5741 ± 0.0066 → 8.7145 ± 0.1355 ms` at 1T (`3.05x`), `13.8985 ± 0.0229 → 4.7895 ± 0.0545 ms` at 2T (`2.90x`), `8.3058 ± 0.1141 → 3.4573 ± 0.3160 ms` at 4T (`2.40x`).
- **Policy, Calibration & Compression (Seed 0, `224×224` RGB thumbnails):**
  - Validation-selected operating points: `tau = 0.8` (≤1% val cap) and `tau = 0.5` (≤5% val cap). On test, `tau = 0.8` discards `13.13%` of tiles at `1.80%` false-deletion rate (FP32/FP16).
  - Lossless 16-bit `.jp2` (verified bit-exact on `50/50` random test tiles) reaches `49.68%` of raw volume alone (`145.8 / 293.5 MB`); adding `tau = 0.8` filtering reaches `43.18%` (`126.8 MB`, an extra `6.5` percentage points of raw volume).

## Phase 5 Hardware Stress, Bit-Flip Fault Injection, Failure Analysis & Statistics (ResNet-18, Seed 0)

Every Phase 5 script verifies `results/phase4_weights.sha256` via `src/verify_weights.py` before running (`ckpt_seed0.pt` SHA-256 unchanged). Full summary table: `results/day34_phase5_master.csv`.

- **5A GPU Energy per Patch & 5C Swath Feasibility (`RTX 5060`, `TF32=False`, idle `3.27 W`, warm steady-state temps `68–84 C`, 3 repeats, `ddof=1`):**
  - `batch=1`: FP32 `241.3 ± 9.1 p/s` (`53.67 W`, total `222.29 ± 5.64`, net `208.72 ± 6.16 mJ/p`) → FP16 `380.8 ± 38.2 p/s` (`39.88 W`, total `105.22 ± 7.65`, net `96.57 ± 6.73 mJ/p`) — `1.58x` faster, `52.7%` lower total (`53.7%` net) energy *(repeat 1 ramped slower at `b=1` and is kept in full 3-repeat stats)*.
  - `batch=8`: FP32 `940.5 ± 2.6 p/s` (`57.39 W`, total `61.02 ± 0.95`, net `57.54 ± 0.96 mJ/p`) → FP16 `2402.1 ± 7.4 p/s` (`51.50 W`, total `21.44 ± 0.24`, net `20.08 ± 0.25 mJ/p`) — `2.55x` faster, `64.9%` lower total (`65.1%` net) energy.
  - `batch=32`: FP32 `1540.2 ± 0.7 p/s` (`59.35 W`, total `38.53 ± 0.14`, net `36.41 ± 0.14 mJ/p`) → FP16 `3127.7 ± 1.1 p/s` (`59.29 W`, total `18.96 ± 0.09`, net `17.91 ± 0.09 mJ/p`) — `2.03x` faster, `50.8%` lower total & net energy. Combining FP16 and batching (`FP32 b=1 → FP16 b=32`) cuts energy/patch by `11.7x`.
  - Against Sentinel-2's `1,943 km²/s` swath (`290 km × 6.7 km/s`), every tested config clears resized `509×509` tiles (`~75 p/s`), including CPU ORT INT8 4T (`289.2 p/s`), while batched GPU clears native `224×224` tiles (`~387 p/s`) by `2.4x–8.1x` (forward-pass compute only).
- **5B 10-Minute Sustained Thermal Stress (`batch=32`, `TF32=False`, `Balanced`, pre-test idle `50.2 C`):**
  - Both precisions hit `87 C` and lock into `SW_THERMAL_SLOWDOWN` (`mask=32`: FP32 at `77.73 s`, FP16 at `37.55 s`), stabilizing by minute 4–5.
  - First-60 s vs. last-60 s latency drifted `+14.57%` for FP32 (`21.3237 → 24.4309 ms`; `1481.6 → 1301.4 p/s`, SM `1837 → 1350 MHz` min) and `+20.91%` for FP16 (`10.4392 → 12.6221 ms`; `3031.9 → 2516.4 p/s`, SM `2182 → 1590 MHz` min). Throttled FP16 (`2516.4 p/s`) still holds a `1.93x` throughput edge (`1.94x` lower median latency) over throttled FP32.
- **5D Software Bit-Flip Fault Injection (`21` `Conv2d`/`Linear` weights = `11,167,936` weights; `357.4M` FP32 vs. `89.3M` INT8 bits; `1,500` total trials, 30 seeds/point):**
  - Compares native FP32 (`0.9179` clean) against GPU simulated per-channel symmetric weight-only INT8 (`sim_int8`, `0.9169` clean — distinct from Phase 4 CPU ORT W8A8 INT8 `0.9210`) at frozen Phase 4 thresholds (`tau = 0.8` and `0.5`). Non-finite logits count strictly as incorrect and fail safe (`NaN >= tau` is `False`).
  - **Equal BER (`1e-8`–`1e-4`):** FP32 degrades at `1e-8` (`0.8935 ± 0.0197`, `5/30` collapses `<85%`), collapses on `13/30` at `1e-7` (`0.8109 ± 0.0689`, silent finite `fd@0.8` up to `98.32%`), `30/30` at `1e-6` (`0.4755 ± 0.1237`, `27/30` non-finite, max `fd@0.8 = 98.71%`), and `30/30` non-finite at `1e-5` (`0.0346`) and `1e-4` (`0.0000`). `sim_int8` holds `0.9169–0.9186` (`0/150` collapses, `0/150` non-finite, max `fd@0.8 = 2.32%`).
  - **Equal total flips (`K = 1`–`10,000`):** Removing the `4x` bit-count advantage, FP32 still collapses on `7/30` at `K=10` (max `fd@0.8 = 39.95%`), `26/30` at `K=100` (`57.35%`), `30/30` at `K=1,000` (`97.16%`), and `30/30` non-finite at `K=10,000` (`0.0176`). `sim_int8` holds `0.9169–0.9186` through `K=1,000` and `0.9143 ± 0.0042` at `K=10,000` (min `0.8687`, `0/180` collapses, `0/180` non-finite, max `fd@0.8 = 2.32%`).
  - **Bit-position breakdown (`K = 1, 10, 100`):** Fragility lives exclusively in FP32 exponent MSB `b30`. A single `b30` flip (`K=1`) collapses `23/30` trials (`0.7142 ± 0.0768`, median `0.7959`) and causes `8/30` catastrophic false deletions peaking at `92.27%` (`0%` non-finite; only `4/30` trip `NaN`). All tested FP32 mantissa (`b0, b11, b22`), lower exponent (`b23–b29`), sign (`b31`), and all 8 `sim_int8` bits (`b0–b7`) show `0/30` collapses at `K=10`, and FP32 mantissa/sign plus `sim_int8` low/MSB remain at `0/30` collapses at `K=100`.
  - **Safety takeaway:** High BER (`>= 1e-5`) drops FP32 `fd@0.8` to `0%` only because `NaN` logits trip the fail-safe keep rule, whereas moderate faults (`1e-7`–`1e-6`, `K=10`–`1,000`, or a single `b30` flip) produce finite corrupted logits that silently discard up to `98.32%` of clear tiles with `0%` non-finite outputs (`92.27%` from a single `b30` flip; up to `98.71%` at `1e-6` with `99.3%` finite tiles).
- **5E Failure Analysis & 5F Statistics (`N = 975` test):**
  - At `tau = 0.8` (Seed 0 FP32), `0/464` (`0.00%`) tiles with `<30%` cloud were falsely deleted; all `14/776` (`1.80%`) false deletions fall in `30–70%` partial cloud (`41%–68%` cloud), with the #1 (`60%` cloud, `40%` shadow, `0%` clear) and #3 (`63%` cloud, `35%` shadow, `2%` clear) worst cases labeled keep-worthy only because shadow counts as ground.
  - Worst cloud leaks (`71%–95%` cloudy, `0%` shadow, `p_fp32 = 0.0019–0.0193`) are visually uniform thin cirrus over dark surfaces, reflecting the physical limit of RGB without Sentinel-2's `B10` (1375 nm cirrus) or SWIR bands.
  - ROI-cluster bootstrap 95% CIs (`B = 2,000`, Seed 0): FP32/FP16 Acc `0.9179 [0.8995, 0.9354]`, AUROC `0.9596 [0.9445, 0.9725]`; CPU ORT INT8 Acc `0.9210 [0.9015, 0.9385]`, AUROC `0.9618 [0.9477, 0.9737]`; GPU `sim_int8` Acc `0.9169 [0.8974, 0.9344]`, AUROC `0.9587 [0.9431, 0.9720]`.
  - Exact two-sided McNemar tests confirm no statistically significant paired difference between FP32 and INT8 across all 3 seeds at `tau = 0.5` (`p = 0.5078, 0.0768, 0.6636`), FP32 vs. `sim_int8` (`p = 1.0000`), or Seed-0 `tau = 0.8` false deletions (`p = 0.2500` ORT-INT8; `p = 0.5000` `sim_int8`).

## Known issues
- Windows Application Control blocks `pyarrow` here, so `tacoreader` runs only in Colab.
- It also blocked `torch`'s `_C` once; an import-order workaround is in `src/dataset.py`, and the cause is undiagnosed.

## Limitations
- Consumer laptop hardware (`RTX 5060`, `i7-13620H`), not flight radiation-hardened hardware; latency, power, and thermal numbers are relative proxies rather than in-orbit performance.
- Single model architecture (`ResNet-18`; MobileNetV2 skipped per scope); classification accuracy/F1/AUROC/ECE are 3-seed across FP32, FP16, and CPU ORT INT8, whereas policy, calibration, compression, latency, energy, thermal, and bit-flip sweeps are Seed 0 (`seed-0 v2`) only.
- Dataset scope: uses 3 of 13 Sentinel-2 bands (`224×224` RGB thumbnails, missing `B10`/SWIR cirrus bands), 4,000 of 8,490 training patches, and ROI-level split checks.
- CloudSEN12+ cloud prevalence is stratified (`20.4%` cloudy in test), so discard rates do not represent global orbital cloud cover.
- Validation optimism: the 535-tile validation split controlled both early stopping and threshold (`tau`) selection, so validation false-deletion caps do not guarantee identical rates out-of-sample (`0.72%` val vs. `1.80%` test at `tau = 0.8`).
- Compression volumes use `224×224` 3-band RGB thumbnails (not full-resolution 13-band Sentinel-2 scenes); only relative percentages apply, and CCSDS 122.0 is cited rather than implemented.
- Swath throughput feasibility (`5C`) measures GPU forward-pass compute only on synthetic VRAM-resident tensors, excluding sensor readout, resizing, host-to-device transfer, and compression.
- No TensorRT or native GPU INT8 latency/energy results (excluded per scope). CPU INT8 (`onnxruntime` W8A8 on `i7-13620H`) and GPU FP16 (`RTX 5060`) run on different hardware and cannot be compared head-to-head.
- Phase 5 bit-flip fault injection is software-only on `Conv2d`/`Linear` weights (excluding BatchNorm buffers, biases, activations, multi-bit upsets, and FP16 bit flips) and uses GPU simulated per-channel weight-only INT8 (`sim_int8`, `0.9169` clean), which is distinct from Phase 4's CPU ONNX Runtime W8A8 INT8 (`0.9210`), and assumes the per-channel FP32 scale factors (`~19.2 KB`, `0.04%` of FP32 weight storage) are protected from bit flips (e.g., via ECC/TMR).

## Reproduce
```text
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
python -m pip install onnx onnxruntime psutil

# Note: Before running the local pipeline, export dataset shards from Google Colab using notebooks/colab/make_manifest.py
python src/verify_env.py
python src/merge_shards.py
python src/check_loaders.py
python src/day15_overfit.py
python src/train.py --seeds 0 1 2 --epochs 15
python src/eval_val.py
python src/day19_telemetry_compare.py
python src/day20_test_and_bench.py --power-mode "Balanced" --repeats 3 --cooldown 60
python src/day21_gate.py
python src/day22_fp16.py
python src/day22_gpu_paired_bench.py --repeats 3 --cooldown 60 --power-mode Balanced
python src/day23_onnx_cpu.py
python src/day24_int8_ptq.py
python src/day24_int8_multiseed.py
python src/policy.py
python src/day26_calibration.py
python src/day27_compression.py
python src/day28_gate.py
python src/day28_replot_figures.py
python src/verify_weights.py
python src/day29_energy_and_feasibility.py --repeats 3 --cooldown 30 --power-mode Balanced
python src/day30_thermal_test.py --duration 600 --cooldown 180 --batch 32 --power-mode Balanced
python src/fault_injection.py
python src/day31_ber_sweep.py --seeds 30 --batch-size 64
python src/day32_bit_position.py --seeds 30 --batch-size 64
python src/day33_error_and_stats.py
python src/day34_gate.py
```
