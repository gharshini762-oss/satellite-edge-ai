# Onboard Cloud Filtering with Quantized Neural Networks (work in progress)

## Goal (planned)
Compare FP32, FP16, and INT8 cloud-vs-clear classifiers on Sentinel-2
patches under satellite downlink limits.
Planned measurements: accuracy, model size, latency, and memory; the trade-off between bandwidth saved and wrongly deleted clear tiles under a high-confidence deletion threshold; and software bit-flip fault injection.
A laptop GPU stands in for onboard hardware, so results will be relative comparisons, not in-orbit performance.

## Status (last updated 2026-11-15)
- Phase 1 (environment and measurement tools): complete.
- Phase 2 (data): complete.
- Phase 3 (FP32 baseline): complete.
- Phase 4 (FP16/INT8 quantization): Not Started.

## Environment
- Device: Acer Nitro ANV15-52
- GPU: NVIDIA GeForce RTX 5060 Laptop GPU, 8 GB VRAM, driver 621.01
- Python 3.12.10
- PyTorch 2.11.0+cu128, CUDA 12.8 build
- Test conditions: `plugged_in=True` and NitroSense `Balanced` mode are self-reported CLI/script settings. Training used PyTorch default precision, whereas evaluation and latency benchmarking enforced strict `TF32=False`.

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
  - Validation (`N = 535`, selected by early stopping on val NLL): Accuracy `0.9296 ± 0.0140`, Precision `0.8282 ± 0.0689`, Recall `0.8632 ± 0.0256`, F1 `0.8437 ± 0.0239`, AUROC `0.9752 ± 0.0013`, 10-bin ECE `0.0349 ± 0.0067`.
  - Test (`N = 975`, evaluated once): Accuracy `0.9231 ± 0.0082`, Precision `0.7997 ± 0.0506`, Recall `0.8375 ± 0.0353`, F1 `0.8167 ± 0.0098`, AUROC `0.9637 ± 0.0046`, 10-bin ECE `0.0314 ± 0.0148`.
- **Error distribution:** Validation errors concentrate in the `30-70%` broken-cloud bin (`11.86%`), `70-90%` thin/broken cloud bin (`14.17%`), and tiles with `>= 20%` cloud shadow (`16.02%`), vs. `0.67%` on `0-10%` clear sky.
- **Strict FP32 GPU inference benchmark (Seed 0, `TF32=False`, `warmup=50, iters=500`, 3 repeats, 60 s cooldown, self-reported Balanced mode):**
  - `batch = 1`: median `3.8511-4.1518 ms` (p95 `4.3441-4.8872 ms`; rep 1 started warm at `61 C` before cooldown, reps 2-3 at `46-48 C` ran in `3.8511-3.8707 ms`), peak allocated VRAM `144.81 MiB` (`pynvml` `565.1 MiB`).
  - `batch = 8`: median `8.3295-8.3677 ms` (p95 `8.6730-8.7718 ms`, `46-47 C`), peak allocated VRAM `152.85-153.76 MiB` (`pynvml` `617.1-639.1 MiB`).
- `batch = 32`: median `20.2801-20.3666 ms` (p95 `20.9369-21.0784 ms`, `47-49 C`), peak allocated VRAM `312.13 MiB` (`pynvml` `705.1 MiB`).

## Known issues
- Windows Application Control blocks pyarrow here, so tacoreader runs only in
  Colab.
- It also blocked torch's _C once; an import-order workaround is in
  src/dataset.py, and the cause is undiagnosed.


## Limitations
- Laptop GPU, not flight hardware.
- Dataset limitations: covers all cloud amounts so discard rates aren't real-world, uses 3 of 13 bands, 4,000 of 8,490 train patches, and ROI-level split checks.

## Reproduce
```text
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
pip install torch torchvision --index-url [https://download.pytorch.org/whl/cu128](https://download.pytorch.org/whl/cu128)

# Note: Before running the local pipeline, export dataset shards from Google Colab using notebooks/colab/make_patches.py
python src/verify_env.py
python src/merge_shards.py
python src/check_loaders.py
python src/day15_overfit.py
python src/train.py --seeds 0 1 2 --epochs 15
python src/eval_val.py
python src/day19_telemetry_compare.py
python src/day20_test_and_bench.py --power-mode "Balanced" --repeats 3 --cooldown 60
python src/day21_gate.py
```

## Planned phases
4. FP16/INT8 quantization and threshold sweep
5. Full benchmarks and bit-flip tests
6. Paper

