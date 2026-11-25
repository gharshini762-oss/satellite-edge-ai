# Onboard Cloud Filtering with Quantized Neural Networks (work in progress)

## Goal (planned)
Compare FP32, FP16, and INT8 cloud-vs-clear classifiers on Sentinel-2
patches under satellite downlink limits.
Planned measurements: accuracy, model size, latency, and memory; the trade-off between bandwidth saved and wrongly deleted clear tiles under a high-confidence deletion threshold; and software bit-flip fault injection.
A laptop GPU stands in for onboard hardware, so results will be relative comparisons, not in-orbit performance.

## Status (last updated 2026-11-24)
- Phase 1 (environment and measurement tools): complete.
- Phase 2 (data): complete.
- Phase 3 (FP32 baseline): complete.
- Phase 4 (FP16/INT8 quantization): complete.
- Phase 5 (benchmarks and bit-flip tests): Not started.

## Environment
- Device: Acer Nitro ANV15-52
- GPU: NVIDIA GeForce RTX 5060 Laptop GPU, 8 GB VRAM, driver 621.01
- Python 3.12.10
- PyTorch 2.11.0+cu128, CUDA 12.8 build
- Test conditions: `plugged_in=True` and NitroSense `Balanced` mode are self-reported CLI/script settings. Training used PyTorch default precision, whereas evaluation and latency benchmarking enforced strict `TF32=False`.
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

All means and standard deviations use sample std (`ddof=1`). Seed 0 uses `seed-0 v2` (`ckpt_seed0.pt`, SHA-256 `B7FDA259...D210`), synced across Phase 3 and Phase 4 (original `baseline-frozen` Seed-0 hash `FC79C352...A504` is preserved in git history and logged in `log.md`).

- **3-seed Test Metrics (`N = 975`, threshold `0.5`, Table II):**
  - FP32 (seeds 0, 1, 2): Accuracy `0.9241 ± 0.0067`, F1 `0.8114 ± 0.0180`, AUROC `0.9637 ± 0.0047`, 10-bin ECE `0.0243 ± 0.0051`.
  - FP16 (seeds 0, 1, 2; `1.0000` prediction agreement with FP32): Accuracy `0.9241 ± 0.0067`, F1 `0.8114 ± 0.0180`, AUROC `0.9637 ± 0.0047`, 10-bin ECE `0.0243 ± 0.0050`.
  - INT8 MinMax (seeds 0, 1, 2; `98.67%–99.28%` test agreement): Accuracy `0.9214 ± 0.0016`, F1 `0.8116 ± 0.0069`, AUROC `0.9636 ± 0.0046`, 10-bin ECE `0.0240 ± 0.0063`.
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

## Known issues
- Windows Application Control blocks pyarrow here, so tacoreader runs only in
  Colab.
- It also blocked torch's _C once; an import-order workaround is in src/dataset.py, and the cause is undiagnosed.



## Limitations
- Laptop GPU (`RTX 5060`), not flight hardware; results are relative comparisons rather than in-orbit performance.
- Dataset scope: uses 3 of 13 Sentinel-2 bands (`224×224` RGB thumbnails), 4,000 of 8,490 training patches, and ROI-level split checks.
- CloudSEN12+ cloud prevalence is stratified (`20.4%` cloudy in test), so discard rates do not represent global orbital cloud cover.
- Validation optimism: the 535-tile validation split controlled both early stopping and threshold ($\tau$) selection, so validation false-deletion caps do not guarantee identical rates out-of-sample (`0.72%` val vs. `1.80%` test at $\tau = 0.8$).
- Policy, calibration, compression, and latency benchmarks are Seed 0 only (classification accuracy/F1/AUROC/ECE are 3-seed across FP32, FP16, and INT8).
- Compression volumes use `224×224` 3-band RGB thumbnails (not full-resolution 13-band Sentinel-2 scenes); only relative percentages apply, and CCSDS 122.0 is cited rather than implemented.
- No TensorRT or GPU INT8 results (excluded per scope).
- INT8 ran on CPU via ONNX Runtime (`i7-13620H`) while FP16 ran on GPU (`RTX 5060`), so GPU FP16 and CPU INT8 latencies are not directly comparable.

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
```
## Planned phases
5. Full benchmarks and bit-flip tests
6. Paper

