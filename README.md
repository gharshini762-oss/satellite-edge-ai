# Onboard Cloud Filtering with Quantized Neural Networks (work in progress)

## Goal (planned)
Compare FP32, FP16, and INT8 cloud-vs-clear classifiers on Sentinel-2
patches under satellite downlink limits.
Planned measurements: accuracy, model size, latency, and memory; the trade-off between bandwidth saved and wrongly deleted clear tiles under a high-confidence deletion threshold; and software bit-flip fault injection.
A laptop GPU stands in for onboard hardware, so results will be relative comparisons, not in-orbit performance.

## Status (last updated 2026-11-11)
- Phase 1 (environment and measurement tools): complete
- Phase 2 (data): complete
- Phase 3 (FP32 baseline): not started
- No model has been trained or quantized yet, so there are no model results.

## Environment
- Device: Acer Nitro ANV15-52
- GPU: NVIDIA GeForce RTX 5060 Laptop GPU, 8 GB VRAM, driver 621.01
- Python 3.12.10
- PyTorch 2.11.0+cu128, CUDA 12.8 build
- Test conditions: "not recorded"

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
- src/verify_env.py, day4_image_basics.py, day5_gpu_benchmark.py, telemetry.py:
  Phase 1 environment checks, image basics, GPU timing, and GPU telemetry
  (temperature, power, clock, VRAM)
- src/merge_shards.py: merges Colab shards, asserts alignment, writes norm stats
- src/dataset.py, check_loaders.py: dataset class, batch checks, sample grids
- [Colab export script: notebooks/...]
- results/: figures, manifests, and logs.

## Phase 1 measurements
- GPU timing: synchronized(Measures actual GPU compute time) v/s unsynchronized (Measures CPU handoff only). GPU calls return before the work finishes, so timing without synchronize() undercounts.

- Telemetry : clock rose from about 200 MHz to about 1750 MHz; temperature rose from about 54 C to about 78 C. 
The temperature trace oscillates regularly, and the cause has not been investigated yet.

## Phase 2 pipeline & observations          
- Idexing mechanism: Discovered that tacoreader expects 0-based row positions (pos) rather than pandas index labels (idx). Passing labels initially caused a silent image-label mismatch, which was identified through control audits and resolved by enforcing position-based reads with automated semantic mask validation.

- Split integrity: Confirmed strict separation across region-of-interest identifiers (roi_id), ensuring no overlapping locations between training, validation, and test subsets.

- Normalization approach: Channel means and standard deviations were computed exclusively from the training split to avoid introducing evaluation bias into validation or test sets.

- Memory mapping: Utilizing memory-mapped array access (mmap_mode='r') significantly reduced memory overhead and allowed high data throughput during batch iteration.

## Known issues
- Windows Application Control blocks pyarrow here, so tacoreader runs only in
  Colab.
- It also blocked torch's _C once; an import-order workaround is in
  src/dataset.py, and the cause is undiagnosed.


## Limitations
- Laptop GPU, not flight hardware
- Measurements so far use random data, not real imagery

## Reproduce
```
python -m venv .venv
.venv\Scripts\Activate.ps1
# install PyTorch with the command from pytorch.org (CUDA 12.8 or newer)
pip install -r requirements.txt
python src/verify_env.py
```
## Planned phases
3. FP32 baseline model and benchmark
4. FP16/INT8 quantization and threshold sweep
5. Full benchmarks and bit-flip tests
6. Paper

