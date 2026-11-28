# Research Log

## 2026-11-01 (Day 1)
- Laptop model: Acer Nitro ANV15-52
- GPU:NVIDIA GeForce RTX 5060 (plus intel  UHD   Graphics)
- VRAM:8 GB 
- driver:621.01
- RAM:16.0GB(15.7GB usable)
- Free disk space:275 GB
- Python:3.12.10 
- Git versions:2.56.0.windows.4
- CUDA UMD Version: 13.5

## Notes

 PHASE-1

Day1:
- Did today: set up the project folder in VS Code, created the GitHub repo, pushed the first commit
- Broke / confusing: misspelled GitHub username in the remote address (oss vs ocs); push blocked by email privacy (GH007), fixed with the noreply address; data.lnk was a shortcut, not a folder
- Next: Day 2, create the virtual environment and install PyTorch for the RTX 5060

Day2:
- Did today:Activated `.venv`, installed PyTorch 2.11.0 with CUDA 12.8 support, verified GPU tensor math on the RTX 5060 Laptop GPU (8124 MB VRAM, sm_120), exported dependencies to requirements.txt, and pushed commit 6ca1e38 to GitHub.
- Broke / confusing: none
- Next:Day 3, NumPy + OpenCV

Day3:
- Did today: Wrote src/day4_image_basics.py covering NumPy array inspection (.shape, .dtype, min/max), OpenCV BGR-to-RGB conversion, 224x224 resizing, and 12-bit (0-4095) to 8-bit (0-255) dynamic range rescaling.
- Observed: Plotting raw OpenCV BGR arrays in Matplotlib swaps the Red and Blue channels; rescaling 12-bit data down to 8-bit introduces a small quantization rounding error (up to ~8 intensity levels out of 4095).
- Next: Day 4, PyTorch tensors and GPU matrix multiplication timing benchmark with torch.cuda.synchronize().

Day 4:
- Did today: Created 4096x4096 PyTorch tensors, moved them to the RTX 5060 GPU with .to("cuda"), added hardware warm-up loops, and benchmarked 100 matrix multiplication runs comparing CPU, unsynchronized GPU, and synchronized GPU speeds.
- Observed : Without torch.cuda.synchronize(), Python's time.time() reports a sub-millisecond execution time because it only measures how fast the CPU dispatches tasks to the GPU queue rather than when the hardware actually finishes computing. 
- Next: Day 5, build the telemetry logger to monitor GPU temperature, VRAM usage, and power draw during model execution.

Day 5:
- Did today: Built src/telemetry.py using pynvml to query live RTX 5060 hardware sensors (temperature, power draw, core clock, and VRAM usage) at 5 Hz during a continuous PyTorch matrix multiplication workload. Added torch.cuda.max_memory_allocated() reporting and plotted temperature/clock stability curves to check for throttling.
- Observed: Peak VRAM allocation was logged successfully, and core clocks remained stable during the stress test window without thermal throttling.
- Next: Day 6, complete the Phase 1 review.

Day 6:
- Did today: Phase 1 review.
- Observed: plugged in, the 4096x4096 matmul took 19.30, 20.00 and 19.13 ms (synchronized median); on battery it took 28.19 ms. Unsynchronized timing read ~0 ms, below the timer's resolution, so I don't report it. Telemetry samples are unevenly spaced (about 0.12 to 0.38 s) and some sensor values repeat between samples.
- Next: read about Sentinel-2.

Day 7 :
- Did today: read about Sentinel-2 (bands, L1C vs. L2A, tile structure). Source: ESA Sentinel-2 User Handbook
- Next: Day 8, make a Copernicus Data Space account.
 

PHASE-2

Day 8: 
- Did today: Created and verified on Copernicus Data Space Ecosystem.
- observed: What a Band is- Instead of capturing just standard Red, Green, and Blue (RGB), Sentinel-2 captures 13 separate wavelength channels (bands) from visible light (10m resolution) to short-wave infrared (20m/60m resolution) to detect surface features and high-altitude cirrus clouds.
Dataset & Proxy Notes
- Sentinel-2 Level-1C provides calibrated Top-of-Atmosphere (TOA) reflectance, which serves as the closest publicly available proxy to raw on-orbit satellite imagery.

Why use L1C: An edge-AI filter running onboard a satellite before downlink only has access to uncorrected L1C data; L2A is only generated on Earth after the bandwidth has already been spent.
Images observed-
T43REK(Tile ID) captured on 2026/11/02 with cloud % of 0.09% and
T43REJ(Tile ID) captured on 2026/07/25 with cloud % of 80%.
- Next: Day 9, Open real satellite pixels

Day 9:
- Did today: opened Sentinel-2 L1C bands B04/B03/B02 of tile T43REK (sensed 2026-11-02, baseline N05.13) with rasterio, downsampled 4x (10,980 -> 2,745 px), stacked as RGB, converted DN to TOA reflectance with (DN - 1000) / 10000, applied a 2-98 percentile stretch for viewing.
- Observed: 61% of the tile is no-data (swath edge). Valid raw DN range 1 to 24,933 (max reflectance 2.39 before clipping). Fraction of valid pixels <0: 0.025%; >1: 0.0004%, so clipping to [0, 1] loses almost nothing on this tile.
- Next: Day 10, get the labeled dataset (CloudSEN12) and read its documentation.

Day 10:
- Did today: Loaded the CloudSEN12+ (`cloudsen12-l1c`) and generated 5-patch sample preview grid (first 5 rows, test-split ROI_00001) and analyzed image edge padding and value scaling.
- Observed: Total rows in dataset are 50,247. The patches contain a 3-pixel zero-padding strip at the edges (expanding native 509x509 to 512x512). The label values inside this zero-padded strip evaluate entirely to `0` ("clear"), confirming that uncropped padded patches would artificially inflate clear-sky fraction metrics.
- Offset evidence: Colab `stac:time_start` agg(['min', 'max']) shows earliest date 2015-12-16 and latest date 2020-11-20. Day 9 used a raw baseline N05.13 product which requires subtracting 1000, but CloudSEN12 predates the Jan 2022 ESA offset baseline. Local check on train_X.npy: min 178 | p1 351.0 | p50 1385.0 | max 18727.

Zero-padded rows count: 3
Zero-padded cols count: 3
Label values in zero-padded strip: (array([0], dtype=uint8), array([512]))

- `0`: **Clear** – Unobstructed ground visibility without cloud cover or shadows.
- `1`: **Thick Cloud** – Opaque, dense cloud cover that completely blocks the ground underneath.
- `2`: **Thin Cloud** – Semi-transparent cloud cover (such as thin cirrus or haze) where ground features remain partially visible through the vapor.
- `3`: **Cloud Shadow** – The dark projected silhouette cast by a cloud onto the underlying terrain.

- Broke / confusing: Local `tacoreader` import failed because Windows Security blocked `pyarrow`'s `_compute.pyd` file via an Application Control policy. Workaround: Used Google Colab to stream data and download samples.

- Dataset counts (CloudSEN12+ l1c): label_type high 10,847 / scribble 10,000 / nolabel 29,400. High-quality 509-px patches: train 8,490 / val 535 / test 975 (10,000 total). High-quality 2000-px patches: 847, not used for now. Scribble patches also appear in val/test, so I filter label_type == "high".
- Leakage check on the official split: 0 of 10,727 roi_ids appear in more than one split.


- Source: CloudSEN12+ v1.1.2, Hugging Face tacofoundation/cloudsen12, loaded with tacoreader (mode cloudsen12-l1c); data from ESA Sentinel-2, hosted by the Image & Signal Processing group. Papers to cite: Aybar et al. 2022 (Scientific Data) and 2024 (Data in Brief). Accessed 2026-11-04.
- License: CC0 (stated on the dataset page for this version). Attribution not required; I will cite the papers anyway.
- Classes: clear, thick cloud, thin cloud, cloud shadow. Label types: high-quality, scribble, no-annotation. I will use high-quality labels for training and evaluation.
- Patch size: 509x509 px (5.09 km at 10 m) and 2000x2000 px, padded to 512x512 and 2048x2048 with zeros on the bottom and right (indices 509..511.)

- Next: Day 11, define labels and cut patches.

Day 11:
- Did today: Computed per-patch cloud fraction (`thick_percentage + thin_percentage`) on the 10,000 high-quality 509x509 patches, plotted `results/cloud_hist.png`, defined labeling thresholds, and ran `src/make_patches.py` in Colab to  export 900 resized 224x224 RGB patches (v1 pilot, superseded) (`245` clear, `199` cloudy, `456` ambiguous_eval_only).
- Observed: Histogram has a spike near 0% (~25% of patches) and a flat spread up to 100% . Train, val, and test splits have matching distributions (means 36.6%, 37.6%, 37.6%; medians 31%, 32%, 33%). Across all high-quality 509-px patches, 35.4% are >=50% cloudy, 20.8% are >=70% cloudy, and 7.6% are >=90% cloudy.

  - Clear (`0`, keep): `cloud_pct <= 10%` (~30% of patches). Cloudy (`1`, discard): `cloud_pct >= 70%` (~20.8% of patches).

  - Ambiguous mid-range (`10% < cloud_pct < 70%`) and heavy shadow (`>= 50%`) were initially excluded from binary training.[SUPERSEDED: switched to training on all patches (Section 4) because the filter must score every patch in deployment. The old rule is superseded.]

  - Cropped 3-pixel zero-padding to 509x509 and resized to 224x224 (coarsening resolution ~2.3x) so full-patch cloud fraction matches the visible patch.

 - Self verification:
Band descriptions (1-indexed): ('B01', 'B02', 'B03', 'B04', 'B05', 'B06', 'B07', 'B08', 'B8A', 'B09', 'B10', 'B11', 'B12')
Bands [4, 3, 2] are: ['B04', 'B03', 'B02']
Valid row range: 0..508 (len 509), Valid col range: 0..508 (len 509)
Max abs difference (cropped 509x509 mask cloud % vs metadata thick+thin %): 54.0000% [NOTE: caused by the v1 index bug where `ds.read(i)` was passed the pandas index label `idx` instead of the 0-based row position `pos`; fixed in `cloud_export_v2` on Day 13, where max difference across all 5,510 patches is <= 2.61%]. Average streaming/read time per patch: 4.47 s/patch



- Next: Day 12, build leak-proof splits (verify zero scene overlap across train/val/test, check class balance per split, and save `manifest.csv`).

 
 Raw output from Day 10/11:

tortilla:data_split         test  train  validation
label_type real_proj_shape                         
high       2000               85    685          77
           509               975   8490         535
nolabel    509                 0  29400           0
scribble   509               655   8785         560
                      count       mean        std  min  25%   50%   75%    max
tortilla:data_split                                                           
test                  975.0  37.586667  32.062542  0.0  5.0  33.0  64.0  100.0
train                8490.0  36.571967  31.903606  0.0  5.0  31.0  63.0  100.0
validation            535.0  37.646729  31.942195  0.0  7.0  32.0  64.0  100.0
cloudy at >= 50 : 0.354
cloudy at >= 70 : 0.208
cloudy at >= 90 : 0.076

Day 12 :
- Did today: 
  - Built the master manifest (`manifest_all.csv`) from high-quality patches and executed leak-proof validation.
  - Asserted zero scene overlap using `assert (man.groupby('roi_id')['split'].nunique() == 1).all()`,confirming no roi_id appears in two splits.
  - Set `N_TRAIN = 4000` based on pilot streaming speeds (~4.47s per patch). Exporting all 8,490 training patches would require ~10+ hours; 4,000 provides an optimal, stratified training subset while keeping execution times manageable.
  - Generated `manifest_export.csv` and executed the sharded export loop to save test and validation arrays into Google Drive (`/content/drive/MyDrive/cloud_export`).
- Observed & Verified:
  - **Leakage check:** Zero ROI/scene overlaps confirmed successfully.
  - **Export outputs:** Successfully streamed and saved `test_000.npz` (500 patches), `test_001.npz` (475 patches), `validation_000.npz` (500 patches), and `validation_001.npz` (35 patches).
  Measured bulk streaming speed: ~3.6 s/patch for test/val (1,510 patches in ~1.5 h) and ~2.25 s/patch for train (4,000 patches in ~2.5 h).
- Next: Day 13, implement `src/dataset.py` 

- Mistakes / Debugging Notes (Day 12 -> Day 13 transition):
  1. Ternary vs. Binary Label Confusion: Initially coded `label` with `-1` for ambiguous mid-range clouds (`10% < cloud_pct < 70%`) which produced negative label means (`-0.293` on test, `-0.276` on train). Caught the mismatch against the Section 4 pre-registered rule
   (`label = 1` if `cloud_pct >= 70` else `0` across all patches) and updated `manifest_all.csv`, `manifest_export.csv`, and the shard labels to strict binary `0/1` (giving the true cloudy share: `20.73%` train, `21.87%` validation, `20.41%` test).
  2. Shard Resolution & Dtype Mismatch: The initial Day 12 `process()` function exported `test` and `validation` shards as unresized `(N, 3, 509, 509)` `float32` arrays instead of resized `(N, 3, 224, 224)` `int16` arrays (`cv2.INTER_AREA`). Instead of re-streaming 1,510 patches over HTTP (~1.5 hours), wrote an in-place conversion script in Colab to resize the existing Drive shards to `(N, 3, 224, 224)` `int16` before exporting the `train` shards.

Day 13:
- Did today:
  - Converted the existing `test` (`test_000.npz`, `test_001.npz`) and `validation` (`validation_000.npz`, `validation_001.npz`) shards on Google Drive from `(N, 3, 509, 509)` `float32` to `(N, 3, 224, 224)` `int16` (`cv2.INTER_AREA`) in-place, and updated all shard and manifest labels to strict binary (`label = 1` if `cloud_pct >= 70.0` else `0`).
  - Finished streaming and exporting all 8 stratified training shards (`train_000.npz` through `train_007.npz`, `N_TRAIN = 4000`) in Colab (~2h 30m total streaming run).
  - Downloaded all 12 `.npz` shards to `data/processed/shards/` (shards of 150,536,728 bytes).
  - Wrote and ran `src/merge_shards.py` to concatenate the shards into memory-mappable `.npy` arrays.
  - Computed per-channel RGB mean and standard deviation:
    - `TRAIN_MEAN = [0.215731, 0.209375, 0.223319]` **[v1, superseded]**
    - `TRAIN_STD  = [0.197625, 0.178327, 0.183987]` **[v1, superseded]**
  - Implemented `CloudDS` (`torch.utils.data.Dataset`) in `src/dataset.py`.

- Re-export event (`cloud_export_v2`, 2026-11-11):
  - What I saw: The initial v1 grids showed severe image-label mismatches (e.g., clear water labeled 81% cloud, and Day 11's pilot mask check was off by up to 54.00%).
  - Cause: `ds.read(i)` expects a 0-based table row position (`pos`), but v1 passed the filtered pandas DataFrame index label (`idx`), loading the wrong image whenever `idx != pos`.
  - Exporter change: Wrote `notebooks/colab/make_manifest.py` (`cloud_export_v2`) to map each patch to its 0-based position `pos`, compute `mask_cloud_pct` on the cropped 509x509 semantic mask (`classes 1 + 2`), and assert `abs(mask_cloud_pct - cloud_pct) <= 3.0%` before resizing. Across all 5,510 exported v2 patches, mask vs. metadata difference dropped from up to 54.00% (by label) to mean 0.24–0.26% and max 2.61% (by position).
  - Time taken & storage: ~3.5 hours in Colab to export all 12 v2 shards (150,545,484 bytes per full 500-patch shard) to Google Drive `/content/drive/MyDrive/cloud_export_v2` and locally to `data/processed/shards/`. The v2 training mean/std are saved in `results/norm_stats.json`.
- Single-item `CloudDS` verification (`python src/dataset.py`):
  - `train`: `len=4000`, `item[0] shape=(3, 224, 224)`, `dtype=torch.float32`, `label=0`
  - `validation`: `len=535`, `item[0] shape=(3, 224, 224)`, `dtype=torch.float32`, `label=0`
  - `test`: `len=975`, `item[0] shape=(3, 224, 224)`, `dtype=torch.float32`, `label=0`
- Where it lives: Google Drive `/content/drive/MyDrive/cloud_export_v2` (150,545,484 bytes per shard) and locally in `data/processed/shards/`. The v2 mean/std stats are now saved in `results/norm_stats.json`.
- Next: Day 14, run batch verification and loader throughput benchmarks (`src/check_loaders.py`), inspect 5x5 visual sample grids, and write `dataset_card.md` to pass the Phase 2 gate.

Day 14:
- Did today:
  - Implemented `src/check_loaders.py` to build `train_loader`, `val_loader`, and `test_loader` (`batch_size=64`, `shuffle=True` for train only, `num_workers=0`).
  - Checked a full training batch for shape, dtype, finite values, class mix, and post-normalization channel statistics.
  - Plotted and saved stratified random 5x5 visual inspection grids (12 `y=1` and 13 `y=0` patches, brightened x2.5 for display) for training (`results/day14_train_grid.png`) and validation (`results/day14_val_grid.png`).
  - Benchmarked full-epoch DataLoader throughput across all 4,000 training patches.
  - Wrote `dataset_card.md` documenting the CloudSEN12+ v1.1.2 L1C source, CC0 license, Aybar et al. (2022, 2024) citations, selection criteria, split counts, leak check, preprocessing, normalization stats, and known dataset biases.
- Observed & Verified (Phase 2 Gate):
  - **Batch check (`next(iter(train_loader))`):**
    - Shape: `(64, 3, 224, 224)` | Dtype: `torch.float32`
    - All finite: `True`
    - Label counts (`0=keep, 1=cloudy`): `[47, 17]` (`cloudy frac = 0.266`)
    - Normalized batch per-channel mean: `[-0.142, -0.168, -0.179]` (near `0.0`)
  - Normalized batch per-channel std: `[0.827, 0.804, 0.788]` (near `1.0`).
  - **Loader throughput:** `1140.8 images/sec` (`3.51s` for `4,000` images with `mmap_mode='r'` and `num_workers=0`).
  - **Visual grid & label quality reaction:** Initial `v1` grids exposed image-label mismatches caused by passing DataFrame index labels (`idx`) to `ds.read()`, which was resolved by re-exporting `cloud_export_v2` using 0-based row positions (`pos`). In the regenerated `v2` stratified grids (brightened x2.5 for display, zero black padding strips on any border), all 12 `y=1` (`>=70%`) tiles match their labels (showing either thick white cloud decks or widespread thin haze/cirrus such as `row=3223` at 77% in train and `row=52` at 82% in val), while `y=0` tiles show clear land/water at 0%, bright snow/ice at 0–5% (`row=638` in train, `row=107` in val), and broken clouds around 45–66% just below the `T=70%` cutoff.
- Debugging / Windows DLL Note:
  - Running `python src/dataset.py` initially triggered `ImportError: DLL load failed while importing _C: An Application Control policy has blocked this file` when `numpy` was imported right before `torch`, whereas `src/check_loaders.py` ran cleanly likely because importing os and pandas first pre-loaded shared C++ runtime DLLs (unverified hypothesis; kept the import-order workaround in src/dataset.py and src/models.py). Reordered the top imports in `src/dataset.py`, after which `python src/dataset.py` passed without error.
- Next: Day 15 (Phase 3 start), watch 3Blue1Brown Ch. 3 & 4 (gradient descent and backpropagation), implement ResNet-18 in `src/models.py`, and run the 150-step single-batch overfit test.

PHASE-3

Day 15:
- Did today:
  - Watched 3Blue1Brown Neural Networks Ch. 3 & 4 (gradient descent and backpropagation) and a short CNN overview video.
  - Wrote `src/models.py` (`build_resnet18()` using `IMAGENET1K_V1` weights with `m.fc = Linear(512, 2)`) and `src/day15_overfit.py`.
  - Ran a 150-step single-batch overfit test on one training batch (`batch_size=64`, `AdamW`, `lr=1e-3`).
- Concept notes (in my own words):
  - Weight: A learnable number on a connection between neurons that scales how much an input signal pushes the next neuron's output up or down.
  - Loss: A single number (cross-entropy here) measuring how far the model's predicted probabilities are from the true 0/1 labels.
  - Epoch: One complete pass through all 4,000 training patches (63 batches of 64).
  - Transfer learning: Starting from a model already trained on a large dataset and fine-tuning its weights on our dataset instead of starting from random weights.
  - Why ResNet-18 starts from ImageNet weights: The early convolutional layers already detect basic edges, textures, and brightness contrasts, so training only needs to adapt those features to satellite cloud patches rather than learning basic vision from scratch on 4,000 images.
- Observed:
  - Peak GPU memory allocated after step 1 (`torch.cuda.max_memory_allocated()`): `1,571,670,016 bytes` (`1498.86 MiB`).
  - Overfit loss curve across 150 steps on one 64-image batch:
    - `step 0`: loss `1.358067`, batch_acc `0.2656`
    - `step 25`: loss `0.000097`, batch_acc `1.0000`
    - `step 50`: loss `0.000013`, batch_acc `1.0000`
    - `step 75`: loss `0.000009`, batch_acc `1.0000`
    - `step 100`: loss `0.000007`, batch_acc `1.0000`
    - `step 125`: loss `0.000006`, batch_acc `1.0000`
    - `step 149`: loss `0.000005`, batch_acc `1.0000`
  - Why this test matters: Memorizing one batch to near-zero loss confirms that the forward pass, label shapes, cross-entropy loss, and gradient updates have no wiring bugs before running full training.
- Next: Day 16, write `src/train.py` and run a 3-epoch smoke test.

Day 16:
- Did today:
  - Wrote `src/train.py` with fixed seeding (`set_seed`), `AdamW(lr=3e-4, weight_decay=1e-4)`, `batch_size=64`, max 15 epochs, early stopping on validation NLL loss (`patience=3`), checkpoint saving (`results/ckpt_seed{s}.pt`), and per-epoch CSV logging.
  - Ran a 3-epoch smoke test (`python src/train.py --seeds 0 --epochs 3 --smoke`) while plugged in and checked `nvidia-smi`.
  - Precision & reproducibility note: Training uses PyTorch default settings (cuDNN/TF32 enabled), whereas strict `TF32=False` is enforced during evaluation and benchmarking (Day 20). Because `torch.use_deterministic_algorithms(True)` is not set on CUDA convs/atomic adds, runs are seeded (`set_seed`) rather than bit-reproducible across separate invocations (smoke seed 0 had epoch-0 `val_loss=0.251544` vs. `0.205256` in the Day 17 run).
- Observed (`results/train_log_smoke_seed0.csv`):
  - `epoch 0`: train_loss `0.277273`, val_loss `0.251544`, val_acc `0.902804`, `28.01s` [saved best]
  - `epoch 1`: train_loss `0.206290`, val_loss `0.174396`, val_acc `0.942056`, `11.36s` [saved best]
  - `epoch 2`: train_loss `0.169944`, val_loss `0.272569`, val_acc `0.882243`, `10.68s` [patience 1/3]
  - Speed per epoch: `28.01s` on epoch 0 (initial disk read into memory cache), then `11.36s` and `10.68s` once cached.
  - Validation accuracy after 3 epochs: peaked at `0.9421` on epoch 1 and dropped to `0.8822` on epoch 2.
  - GPU sensors (`nvidia-smi` right after completion): temperature cooled from `64C -> 63C -> 58.25C` as the GPU returned to P4 idle (`3.30 W` to `7 W`, `180 MHz`, power cap reported as `50W`).
- Next: Day 17, train seeds 0, 1, and 2 with early stopping and save training curves.

Day 17:
- Did today:
  - Trained three ResNet-18 runs (`seeds 0, 1, 2`, PyTorch default precision) via `python src/train.py --seeds 0 1 2 --epochs 15`.
  - Saved best validation-loss checkpoints (`results/ckpt_seed0.pt`, `ckpt_seed1.pt`, `ckpt_seed2.pt`), epoch logs (`results/train_log_seed{0,1,2}.csv`), and loss/accuracy curves (`results/curve_seed{0,1,2}.png`).
  - Did not touch the test set.
- Observed:
  - Epoch timing stayed around `10.8s–12.7s` per epoch across all three runs.
  - In all three runs, training loss continued falling (`~0.28 -> ~0.14–0.16`) while validation NLL loss bottomed out around epochs 3–6 and then started rising, triggering early stopping (`patience=3`):
    - **Seed 0 (`train_log_seed0.csv`):** Best at **epoch 3** (`train_loss=0.167937`, `val_loss=0.182599`, `val_acc=0.921495`); stopped after **epoch 6** (`val_loss=0.204739`, `val_acc=0.910280`).
    - **Seed 1 (`train_log_seed1.csv`):** Best at **epoch 6** (`train_loss=0.145337`, `val_loss=0.155966`, `val_acc=0.945794`); stopped after **epoch 9** (`val_loss=0.195449`, `val_acc=0.936449`).
    - **Seed 2 (`train_log_seed2.csv`):** Best at **epoch 4** (`train_loss=0.168860`, `val_loss=0.167350`, `val_acc=0.921495`); stopped after **epoch 7** (`val_loss=0.235554`, `val_acc=0.930841`).
  - Run-to-run spread across seeds (at the best val_loss checkpoint): validation accuracy ranged from `0.9215` (seeds 0 and 2) to `0.9458` (seed 1), a spread of `2.43` percentage points (`0.0243`), and best validation NLL loss ranged from `0.1560` to `0.1826`.
- Next: Day 18, evaluate validation metrics, calibration, and error bins across the three seeds.

Day 18:
- Did today:
  - Wrote `src/metrics.py` (`auroc`, `ece`, `basic` in NumPy) and `src/eval_val.py`.
  - Evaluated `ckpt_seed0.pt`, `ckpt_seed1.pt`, and `ckpt_seed2.pt` on the validation split (`N = 535`: `117` cloudy `y=1`, `418` keep-worthy `y=0`), asserting exact label alignment with `manifest_export.csv`.
  - Saved `results/day18_val_metrics.csv`, `results/day18_val_error_bins.csv`, 10-bin reliability diagrams (`results/day18_val_reliability.png`), and the 12 worst errors gallery for seed 0 (`results/day18_val_failure_gallery.png`).
- Observed:
  - **Per-seed validation metrics (`results/day18_val_metrics.csv`, threshold 0.5, recomputed with strict `TF32=False` to match Day 20):**
    - `seed 0`: acc `0.9215` (493/535), precision `0.7820` (`tp=104, fp=29`), recall `0.8889` (`fn=13, tn=389`), f1 `0.8320`, auroc `0.9737`, ece `0.0392` (`0.0389` with default TF32)
    - `seed 1`: acc `0.9458` (506/535), precision `0.9074` (`tp=98, fp=10`), recall `0.8376` (`fn=19, tn=408`), f1 `0.8711`, auroc `0.9759`, ece `0.0272`
    - `seed 2`: acc `0.9215` (493/535), precision `0.7953` (`tp=101, fp=26`), recall `0.8632` (`fn=16, tn=392`), f1 `0.8279`, auroc `0.9760`, ece `0.0381`
  - **Validation mean ± std across 3 seeds (`TF32=False`, note: optimistic because checkpoints were selected by early stopping on validation NLL):**
    - Accuracy: `0.9296 ± 0.0140`
    - Precision: `0.8282 ± 0.0689`
    - Recall: `0.8632 ± 0.0256`
    - F1: `0.8437 ± 0.0239`
    - AUROC: `0.9752 ± 0.0013`
    - ECE (10 bins): `0.0349 ± 0.0067` (`0.0347 ± 0.0066` under default TF32)
  - **Calibration (`results/day18_val_reliability.png`):** ECE is between `0.0272` and `0.0392`, though mid-probability bins (`0.2–0.7`) have few samples and sit below the diagonal, showing overconfidence in uncertain predictions.
  - **Error rate by `cloud_pct` bin and shadow (`results/day18_val_error_bins.csv`):**
    - `0 to 10%` (`n=150`): seed0 `0.0000`, seed1 `0.0067`, seed2 `0.0133` | mean `0.0067`
    - `10 to 30%` (`n=105`): seed0 `0.0095`, seed1 `0.0095`, seed2 `0.0190` | mean `0.0127`
    - `30 to 70%` (`n=163`): seed0 `0.1718`, seed1 `0.0491`, seed2 `0.1350` | mean `0.1186`
    - `70 to 90%` (`n=80`): seed0 `0.1000`, seed1 `0.1750`, seed2 `0.1500` | mean `0.1417`
    - `90 to 100%` (`n=37`): seed0 `0.1351`, seed1 `0.1351`, seed2 `0.1081` | mean `0.1261`
    - `shadow >= 20%` (`n=77`): seed0 `0.2208`, seed1 `0.0909`, seed2 `0.1688` | mean `0.1602`
  - **Failure gallery & error breakdown (`results/day18_val_failure_gallery.png`, Seed 0):**
    - Tabulating metadata across the 12 worst Seed 0 errors shows 6 False Positives (`y=0, pred=1`: rows `213, 435, 451, 78, 120, 456`, with `cloud_pct` between `60%` and `69%` just below the `T=70%` cutoff and `shadow_pct` between `11%` and `31%`) and 6 False Negatives (`y=1, pred=0`: rows `48, 270, 32, 111, 162, 375`, with `cloud_pct` between `70%` and `92%` and `clear_pct` up to `21%`).
  - **Recipe freeze decision:** Tried `0` hyperparameter changes on validation; froze the recipe as-is.
- Next: Day 19, fix Phase 1 timer resolution and compare `pynvml` against `nvidia-smi` telemetry under a 20-second continuous load.

Day 19:
- Did today:
  - Skipped MobileNetV2 per the Day 19 schedule rule to keep Phase 3 on schedule.
  - Updated `src/day5_gpu_benchmark.py` to use `time.perf_counter()` and logged the unsynchronized vs. synchronized 4096x4096 matmul latencies.
  - Wrote and ran `src/day19_telemetry_compare.py` to log a 20-second continuous GPU matrix multiplication load at 5 Hz (`200 ms` target interval) using both `pynvml` (`results/day19_pynvml.csv`) and `nvidia-smi -lms 200` (`results/day19_nvidiasmi.csv`), and plotted the overlay (`results/day19_telemetry_compare.png`).
- Observed:
  - **Phase 1 timer fix (`python src/day5_gpu_benchmark.py`):** With `time.perf_counter()`, unsynchronized CPU kernel-dispatch latency across 100 runs is `0.0124 ms` median (`p95 = 0.0208 ms`; `0.0114 ms` median / `0.0193 ms` p95 in the `day21_gate.py` check), resolving the `0.0 ms` truncation from `time.time()`. Synchronized GPU matmul measured `21.37 ms` median (`p95 = 24.85 ms`).
  - **Telemetry alignment & early oscillation (0 to 6 s):** `pynvml` (integer temperatures `54 C` to `81 C`) and `nvidia-smi` (`53.75 C` to `80.81 C`) overlay closely. As power ramps from idle (`6.71 W`, `180 MHz`) to peak boost (`63.62 W`, `1740 MHz`), the boost controller steps core clocks down (`1740 -> 1125 -> 1687 -> 1237 -> 1672 -> 1470 MHz`), causing temperature to oscillate between `65.72 C` and `73.00 C`. After `t = 8.0 s`, power settles at the active P0 limit (`~59.8–60.0 W` in self-reported `Balanced` mode, vs. the `50 W` cap reported in P4 idle on Day 16), clocks stabilize at `1680–1747 MHz`, and temperature rises smoothly from `78 C` to `81 C`.
  - **Telemetry repeat rate & effective sensor refresh:** In `results/day19_pynvml.csv`, `21 / 99` consecutive intervals repeat identical `(temp, power, clock)` values (`21.2%`, or ~1 repeat every `4.7` rows at 5 Hz), implying an underlying driver register update rate of `~3.94 Hz` (`~254 ms`). In `results/day19_nvidiasmi.csv`, `17 / 97` intervals repeat (`17.5%`, ~1 every `5.7` rows), implying `~4.12 Hz` (`~243 ms`). Both tools query the same NVML driver sensor (~4 Hz / ~250 ms refresh); I use `pynvml` with `time.perf_counter()` in Python benchmarks to avoid subprocess overhead.
- Next: Day 20, run the one-time FP32 test evaluation across all 3 seeds, save frozen validation and test logits, and benchmark FP32 inference latency and VRAM across batch sizes 1, 8, and 32.

Day 20:
- Did today:
  - Wrote and ran `src/day20_test_and_bench.py --power-mode "Balanced" --repeats 3 --cooldown 60` (`TF32=False`; note: `plugged_in=True` and `power_mode="Balanced"` are self-reported CLI/script settings).
  - Evaluated frozen checkpoints (`ckpt_seed0.pt`, `ckpt_seed1.pt`, `ckpt_seed2.pt`) once on the validation (`N = 535`) and test (`N = 975`: `199` cloudy `y=1`, `776` keep-worthy `y=0`) splits with `TF32=False`, saved `results/day20_test_metrics.csv`, and exported frozen logits (`results/val_logits_fp32_seed{0,1,2}.npy` and `results/test_logits_fp32_seed{0,1,2}.npy`).
  - Benchmarked strict FP32 inference latency (`median` and `p95` across `iters=500` timed iterations after `warmup=50` warm-up runs using CUDA events) and VRAM (`torch.cuda.max_memory_allocated()` and `pynvml`) on seed 0 across `batch_size = 1, 8, 32` with 3 repeats and 60-second cooldowns (`results/baseline_results.csv`).
- Observed:
  - **Model size:** `11,177,538` parameters (`42.71 MiB` checkpoint size on disk).
  - **One-time FP32 test set metrics (`results/day20_test_metrics.csv`, threshold 0.5, N = 975):**
    - `seed 0`: acc `0.9149` (`tp=174, fp=58, fn=25, tn=718`), precision `0.7500`, recall `0.8744`, f1 `0.8074`, auroc `0.9597`, ece `0.0476`
    - `seed 1`: acc `0.9313` (`tp=160, fp=28, fn=39, tn=748`), precision `0.8511`, recall `0.8040`, f1 `0.8269`, auroc `0.9627`, ece `0.0281`
    - `seed 2`: acc `0.9231` (`tp=166, fp=42, fn=33, tn=734`), precision `0.7981`, recall `0.8342`, f1 `0.8157`, auroc `0.9688`, ece `0.0185`
  - **Test mean ± std across 3 seeds:**
    - Accuracy: `0.9231 ± 0.0082` (0.65 percentage points below validation accuracy `0.9296 ± 0.0140`, which was selected by early stopping)
    - Precision: `0.7997 ± 0.0506`
    - Recall: `0.8375 ± 0.0353`
    - F1: `0.8167 ± 0.0098`
    - AUROC: `0.9637 ± 0.0046`
    - ECE (10 bins): `0.0314 ± 0.0148`
  - **FP32 GPU inference benchmark (`results/baseline_results.csv`, seed 0, `warmup=50, iters=500`, self-reported Balanced mode):**
    - `batch = 1`: rep 1 `4.1518 ms` (p95 `4.8872 ms`, start `61.0 C`), rep 2 `3.8511 ms` (p95 `4.3441 ms`, start `48.0 C`), rep 3 `3.8707 ms` (p95 `4.3550 ms`, start `46.0 C`) | peak tensor VRAM `144.81 MiB` (`pynvml` total `565.1 MiB`). Note: rep 1 started warm (`61.0 C`) immediately after Step 1 evaluation without a prior cooldown, making it `~7.5%` slower than cool repeats 2 and 3.
    - `batch = 8`: rep 1 `8.3295 ms` (p95 `8.6730 ms`, `46.0 C`), rep 2 `8.3677 ms` (p95 `8.7718 ms`, `47.0 C`), rep 3 `8.3562 ms` (p95 `8.7667 ms`, `47.0 C`) | mean median `8.3511 ms` (`1.0439 ms/image`), peak tensor VRAM `152.85–153.76 MiB` (`pynvml` `617.1–639.1 MiB`).
    - `batch = 32`: rep 1 `20.2801 ms` (p95 `20.9369 ms`, `47.0 C`), rep 2 `20.3326 ms` (p95 `21.0274 ms`, `48.0 C`), rep 3 `20.3666 ms` (p95 `21.0784 ms`, `49.0 C`) | mean median `20.3264 ms` (`0.6352 ms/image`), peak tensor VRAM `312.13 MiB` (`pynvml` `705.1 MiB`).
- Next: Day 21, recompute validation metrics from `TF32=False` saved logits, generate test diagnostics figure, log carry-forward Phase 4 diagnostics and checkpoint SHA-256 hashes, update `README.md`, and tag `baseline-frozen`.

Day 21:
- Did today:
  - Wrote and ran `src/day21_gate.py` on the saved `TF32=False` `.npy` validation and test logits to generate `results/day21_baseline_summary.csv` and `results/day21_test_cm_and_reliability.png`.
  - Computed exact 10-bin test reliability numbers from `test_logits_fp32_seed0.npy`, Phase 4 carry-forward data diagnostics (mask vs. metadata label flips at `T=70%` and reflectance clipping fraction at `1.0`), and SHA-256 checkpoint hashes.
  - Backed up `ckpt_seed{0,1,2}.pt` to Google Drive, updated `dataset_card.md`, `.gitignore`, and `README.md`, and prepared the repository for the `baseline-frozen` commit and tag.
- Observed:
  - **Frozen FP32 Baseline Summary (`results/day21_baseline_summary.csv`, mean ± std across seeds 0, 1, 2 at thr = 0.5, both computed from `TF32=False` saved logits; validation numbers reflect early-stopping selection on val NLL):**
    - `acc`: val `0.9296 ± 0.0140` | test `0.9231 ± 0.0082`
    - `precision`: val `0.8282 ± 0.0689` | test `0.7997 ± 0.0506`
    - `recall`: val `0.8632 ± 0.0256` | test `0.8375 ± 0.0353`
    - `f1`: val `0.8437 ± 0.0239` | test `0.8167 ± 0.0098`
    - `auroc`: val `0.9752 ± 0.0013` | test `0.9637 ± 0.0046`
    - `ece` (10 bins): val `0.0349 ± 0.0067` | test `0.0314 ± 0.0148`
  - **Seed 0 test confusion matrix & exact 10-bin reliability numbers (`results/day21_test_cm_and_reliability.png`, N = 975):**
    - Confusion matrix at `thr = 0.5` (`Acc = 0.9149`, `F1 = 0.8074`): `tn = 718` (`73.6%`), `fp = 58` (`5.9%`), `fn = 25` (`2.6%`), `tp = 174` (`17.8%`). At `thr = 0.5`, `58 / 776 = 7.47%` of keep-worthy test tiles (`y = 0`) are falsely deleted, motivating a high operational threshold (`tau >= 0.95`) in Phase 4.
    - Exact 10-bin test reliability values from `test_logits_fp32_seed0.npy` (`ECE = 0.0476`):
      - `[0.0, 0.1]`: `n=621`, `mean_p=0.0115`, `emp_cloudy=0.0161`
      - `[0.1, 0.2]`: `n=50`, `mean_p=0.1419`, `emp_cloudy=0.0600`
      - `[0.2, 0.3]`: `n=23`, `mean_p=0.2441`, `emp_cloudy=0.1739`
      - `[0.3, 0.4]`: `n=39`, `mean_p=0.3523`, `emp_cloudy=0.1795`
      - `[0.4, 0.5]`: `n=10`, `mean_p=0.4524`, `emp_cloudy=0.1000`
      - `[0.5, 0.6]`: `n=21`, `mean_p=0.5598`, `emp_cloudy=0.3810`
      - `[0.6, 0.7]`: `n=19`, `mean_p=0.6553`, `emp_cloudy=0.4737`
      - `[0.7, 0.8]`: `n=29`, `mean_p=0.7631`, `emp_cloudy=0.4828`
      - `[0.8, 0.9]`: `n=33`, `mean_p=0.8546`, `emp_cloudy=0.6667`
      - `[0.9, 1.0]`: `n=130`, `mean_p=0.9773`, `emp_cloudy=0.9308`
  - **Phase 4 carry-forward diagnostics:**
    - Validation has `418` keep-worthy (`y=0`) tiles, so a `<= 1%` false-deletion constraint corresponds to `<= 4` tiles and has substantial sampling noise; Phase 4 will report ROI-cluster bootstrap 95% confidence intervals and keep `clear_ground_lost` alongside `false_deletion` (since almost all false positives are `30–70%` partly cloudy tiles rather than `0–10%` clear sky).
    - Label flips at `T = 70%` (`mask_cloud_pct >= 70%` vs. integer metadata `cloud_pct >= 70%`): `14 / 5,510` total (`0.25%`): `10 / 4,000` train (`0.25%`), `2 / 535` validation (`0.37%`), `2 / 975` test (`0.21%`).
    - Reflectance clipping at `1.0` (`DN > 10,000`): `0.39%` of pixels in cloudy training tiles (`y = 1`) and `0.50%` across all training tiles exceed `DN = 10,000` (max `18,727`).
  - **Frozen Checkpoint SHA-256 Hashes (`42.71 MiB` each):**
    - `ckpt_seed0.pt`: `FC79C352FF9ACC234B35702E3B7DD19E6ADD388C377E5B5A06288E5B753DA504`
    - `ckpt_seed1.pt`: `8DF2EFCF57D16DDE59EC6E205681DE3E6000B6F5D230B8D1C99F436317584AD2`
    - `ckpt_seed2.pt`: `3A86424FC29248F0BE02F35C7AED8A99CB642ED4D3291AB43A437550B1354DC2`
- Next: Day 22 (Phase 4 start), convert the frozen checkpoints to FP16 (`model.half()`), evaluate validation and test logits once, check prediction agreement and max logit drift against FP32, and benchmark FP16 GPU latency and VRAM across batch sizes 1, 8, and 32.


Phase 4:

Day 22:
- Did today: Hit `WinError 4551` (Windows Application Control) before running any code—`pip.exe` itself was blocked, `pandas` was pulling in blocked `pyarrow` DLLs, and `torch_python.dll` failed to load. Rather than touching Windows security settings, I removed the Colab-only packages (`pyarrow 26.0.1`, `tacoreader 2.4.21`, `duckdb 2.0.0`, and unused `torchaudio 2.11.0+cu128`) using `python -m pip uninstall`. After that, `torch 2.11.0+cu128` loaded cleanly with `CUDA: True`. Switched all Phase 4 scripts to Python's built-in `csv` module instead of `pandas` to keep `pyarrow` out of the pipeline.
- Seed-0 provenance mismatch caught and resolved (`seed-0 v2`):
  - First run of `src/day22_fp16.py` showed a red flag: Seeds 1 and 2 had `100%` FP16/FP32 agreement with max logit diffs around `0.015–0.021`, while Seed 0 showed `94.58%` validation agreement (`93.95%` test) and a `3.493` max logit diff.
  - Checking the live weights confirmed that `results/ckpt_seed0.pt` (SHA-256 `B7FDA259D6593EBFE57F6244053217AE48645407391428E591B9F465F2D1D210`, modified `2026-11-15 14:14`) no longer matched the original Phase 3 logged hash (`FC79C352FF9ACC234B35702E3B7DD19E6ADD388C377E5B5A06288E5B753DA504`):
    - Live `results/ckpt_seed0.pt`: validation accuracy `0.9421` (`504/535`), validation NLL `0.160261`.
    - Original Phase 3 `val_logits_fp32_seed0.npy`: validation accuracy `0.9215` (`493/535`); max absolute logit difference `3.4958`.
  - No backup copy of `FC79C352...A504` exists locally or on Google Drive. Because live validation NLL (`0.160261`) did not match the Day 16 smoke-run epoch-1 loss (`0.174396` at the same `504/535` accuracy), the live file is not the smoke-test checkpoint.
  - Resolution: Declared live `ckpt_seed0.pt` as `seed-0 v2`, synced `val_logits_fp32_seed0.npy`, `test_logits_fp32_seed0.npy`, `day20_test_metrics.csv`, and `day21_baseline_summary.csv` to match `seed-0 v2` so Phase 3 and Phase 4 evaluate the exact same weights, and preserved the original `baseline-frozen` artifacts untouched in git history (note: the `baseline-frozen` tag points at `b35ec49`, moved from `e883155` during Phase 3 formatting fixes). Explicitly noting that `seed-0 v2` scores higher on validation (`94.21%` vs. `92.15%`) and slightly higher on test accuracy (`91.79%` vs. `91.49%`, lower test F1 `0.7917` vs. `0.8074`, lower test ECE `0.0263` vs. `0.0476`) than the lost original to avoid any appearance of cherry-picking.
- Prediction parity (all 3 seeds, val and test): `1.0000` (100%) agreement with FP32 across every split and seed, zero accuracy change and negligible AUROC/ECE drift (`|Δacc| = 0`, `|ΔAUROC| ≤ 6e-5`, `|ΔECE| ≤ 4.2e-5`), and max logit diffs between `0.0152` and `0.0215`.
  - Seed 0 test: `acc=0.9179`, `f1=0.7917`, `auroc=0.9596`, `ece=0.0263`
  - Seed 1 test: `acc=0.9313`, `f1=0.8269`, `auroc=0.9627`, `ece=0.0281`
  - Seed 2 test: `acc=0.9231`, `f1=0.8157`, `auroc=0.9688`, `ece=0.0186`
- Checkpoint size: Dropped `2.00x` from `42.71 MiB` (FP32) to `21.38 MiB` (FP16).
- GPU latency & VRAM (Seed 0, same-session interleaved paired benchmark via `src/day22_gpu_paired_bench.py`, RTX 5060, `TF32=False`, plugged in, NitroSense `Balanced`, 50 warm-up, 500 CUDA-event iters, 3 repeats, 60 s cooldown before every run including repeat 1, isolated single-model loading, start temps `48–54°C`, sample std `ddof=1`, `max` peak VRAM across repeats; replaces earlier provisional numbers that compared FP16 against the older Day 20 FP32 run):
  - Batch 1: `4.1455 ± 0.1013 ms` (p95 `4.8998 ms`) → `2.6147 ± 0.0581 ms` (p95 `4.1809 ms`) — **1.59x faster**; peak VRAM `98.87 → 36.86 MiB` (`62.7%` drop).
  - Batch 8: `8.6842 ± 0.0571 ms` (p95 `9.0605 ms`) → `3.1590 ± 0.0267 ms` (p95 `3.5895 ms`) — **2.75x faster**; peak VRAM `106.91 → 61.66 MiB` (`42.3%` drop).
  - Batch 32: `20.4855 ± 0.0255 ms` (p95 `21.2737 ms`) → `10.1905 ± 0.0900 ms` (p95 `11.3769 ms`) — **2.01x faster**; peak VRAM `266.19 → 151.55 MiB` (`43.1%` drop).

Even at batch 1, FP16 was noticeably faster rather than completely launch-bound, and batch 8 hit the biggest relative speedup.

Day 23:
- Did today: Installed `onnx` and `onnxruntime` via `python -m pip`; Windows Application Control did not block either package, so the ONNX Runtime CPU path worked without needing a fallback.
- Environment recorded: CPU `13th Gen Intel(R) Core(TM) i7-13620H`; `torch 2.11.0+cu128`, `onnx 1.23.2`, `onnxruntime 1.32.0`, `psutil 7.2.2`, `cv2 5.0.0`, `numpy 2.5.4`.
- Export: Exported `results/ckpt_seed0.pt` to `results/resnet18_fp32.onnx` (`42.63 MiB`, SHA-256 `7A3CCA07CC09CD538354ABD24F5F0E69BA77526047D1BF052C0E18B29E45C815`) at opset 17 using `dynamo=False`. PyTorch 2.11 printed a deprecation warning for the legacy TorchScript exporter, which is expected here.
- Parity check (256 validation images vs. `seed-0 v2` FP32 logits): `max_abs_diff = 1.452e-04`, `argmax_agree = 1.0000`. (Before syncing the Seed-0 reference file, this printed `3.4958`, which helped confirm the Day 22 mismatch.)
- CPU FP32 latency (`CPUExecutionProvider`, `i7-13620H`, batch 1, 50 warm-up, 300 iters, 3 repeats, plugged in, sample std `ddof=1`):
  - 1 thread: `26.5741 ± 0.0066 ms` (p95 `28.1545 ms`)
  - 2 threads: `13.8985 ± 0.0229 ms` (p95 `15.2255 ms`)
  - 4 threads: `8.3058 ± 0.1141 ms` (p95 `10.6894 ms`)

CPU scaling across 1→2→4 threads is clean and repeatable. These numbers are strictly the CPU baseline for Day 24 INT8, not a direct comparison against RTX 5060 GPU times.

Day 24:
- Did today: Built a 256-image calibration subset from the `train` split (`train=False`, `seed=0`, never validation or test) and ran ONNX Runtime static QDQ quantization (`QUInt8` activations, `QInt8` per-channel weights) across MinMax, Percentile, and Entropy calibration. ONNX Runtime printed its standard warning recommending graph pre-processing before quantization, which I left as-is for the baseline comparison.
- Bug caught: My first run showed `acc_drop = +0.0187` because it was still comparing INT8 (`0.9402`) against the old `0.9215` FP32 file. Updated the reference to `seed-0 v2` (`0.9421`) and reran.
- MinMax vs. Entropy ONNX equivalence: Hashing the exported files showed that `resnet18_int8_minmax.onnx` and `resnet18_int8_entropy.onnx` are byte-for-byte identical (both SHA-256 `4E6DB7B61A44525D59FDA541219B3D1E0F87A037AF7360F955F5BA460A86A3C9`), while `resnet18_int8_percentile.onnx` is distinct (`908F83FFDD1C735C669EA26ED4B18DEFF64FE90D6198CA08DB011AA28D640DBD`). ONNX Runtime therefore produced two distinct quantized graphs (`minmax`/`entropy` vs. `percentile`), not three, and CPU latency differences between `minmax` and `entropy` are run-to-run noise.
- Validation comparison (Seed 0, file size `42.63 → 10.77 MiB`, `3.96x` smaller):
  - `minmax` (= `entropy`): `acc=0.9402` (`-0.0019`, i.e. 1 of 535 validation tiles), `f1=0.8632`, `auroc=0.9754`, `ece=0.0210`, `agree=0.9963`
  - `percentile`: `acc=0.9346` (`-0.0075`), `f1=0.8511`, `auroc=0.9763`, `ece=0.0221`, `agree=0.9925`
- Selection & layer exclusions: `minmax` stayed well within the 1-point validation cutoff, so I kept full-model quantization with no layer exclusions and no QAT, selecting `minmax` on validation.
- CPU INT8 latency (`minmax`, `i7-13620H`, batch 1, 3 repeats, plugged in, `ddof=1`, vs. Day 23 ORT FP32):
  - 1 thread: `8.7145 ± 0.1355 ms` (p95 `10.1963 ms`) — **3.05x faster**
  - 2 threads: `4.7895 ± 0.0545 ms` (p95 `5.6378 ms`) — **2.90x faster**
  - 4 threads: `3.4573 ± 0.3160 ms` (p95 `4.8635 ms`) — **2.40x faster**
  - *(Entropy and Percentile latencies sat within run-to-run variance: `8.7042–8.8264 ms` at 1T, `4.7595–4.8393 ms` at 2T, `3.1541–3.2980 ms` at 4T; because `entropy` and `minmax` are identical graphs, their latency gap is pure noise.)*
- Test once (`minmax`, Seed 0): `acc=0.9210` (`+0.31` percentage points over Seed-0 FP32 `0.9179`), `f1=0.8041`, `auroc=0.9618`, `ece=0.0183`, `agree=0.9928`. Because validation showed `-0.19%` (1 tile of 535) while Seed-0 test showed `+0.31%`, Seed-0 FP32 and INT8 accuracy are not distinguishable.
- **3-seed INT8 check (`src/day24_int8_multiseed.py`):** Applied the frozen `minmax` recipe (same 256 train calibration images) to Seeds 1 and 2 (`10.77 MiB` each; ONNX FP32 parity `1.10e-05` Seed 1 and `6.68e-06` Seed 2; SHA-256 `B07036E35FA329FD7C3C6D6E0FD1566E4066B061DF3421120236088137C16BD9` for Seed 1 and `896A04A345973F9C8D1D6BC1E19B9B87A58CC37F5D1673B85B3CE416CCDA9A6C` for Seed 2).
  - Seed 1 test: `acc=0.9231 (-0.0082)`, `auroc=0.9601`, `ece=0.0229`, `agree=0.9867`.
  - Seed 2 test: `acc=0.9200 (-0.0031)`, `auroc=0.9687`, `ece=0.0307`, `agree=0.9867`.
  - Across all 3 seeds (`ddof=1`), INT8 test metrics are `acc 0.9214 ± 0.0016`, `f1 0.8116 ± 0.0069`, `auroc 0.9636 ± 0.0046`, `ece 0.0240 ± 0.0063`—a `-0.27` percentage-point mean accuracy difference vs. FP32 (`0.9241 ± 0.0067`), within seed-to-seed variance across three seeds (no formal significance test at n=3).

Day 25:
- Did today: Wrote `src/policy.py` and swept $\tau \in \{0.5, 0.8, 0.9, 0.95, 0.99, 0.999\}$ on validation, then evaluated the chosen thresholds on test with 1,000-replicate `roi_id` cluster bootstrap 95% confidence intervals.
- Bug caught: After fixing Days 22–24, I missed updating `src/policy.py` (and Day 26), so FP32 still loaded the old Seed-0 logits and disagreed with FP16 even at $\tau=0.5$ (`bw=0.2379` vs `0.1897`). Synced to `seed-0 v2` and reran; FP32 and FP16 then matched to 6 decimal places across all 6 thresholds.
- Validation-selected operating points: All three precisions chose $\tau = 0.8$ for the $\le 1\%$ false-deletion cap (allowing `3/418` = `0.72%` false deletions on validation) and $\tau = 0.5$ for the $\le 5\%$ cap.
- **Test results (975 tiles: 776 keep-worthy, 199 cloudy; 95% cluster CIs):**
  - **$\tau = 0.8$ (chosen for $\le 1\%$ val cap):**
    - FP32 & FP16: bandwidth saved `13.13%` (`[11.38%, 14.97%]`), false deletion `1.80%` (`14/776`, `[0.91%, 2.82%]`), cloud leakage `42.71%` (`85/199` cloudy tiles still downlinked), clear ground lost `1.32%`.
    - INT8: bandwidth saved `13.64%` (`[11.79%, 15.59%]`), false deletion `2.19%` (`[1.18%, 3.31%]`), cloud leakage `41.71%`, clear ground lost `1.46%`.
  - **$\tau = 0.5$ (chosen for $\le 5\%$ val cap):**
    - FP32 & FP16: bandwidth saved `18.97%` (`[17.13%, 20.82%]`), false deletion `4.25%` (`[2.93%, 5.70%]`), cloud leakage `23.62%`, clear ground lost `3.14%`.
    - INT8: bandwidth saved `19.90%` (`[18.05%, 21.85%]`), false deletion `4.64%` (`[3.22%, 6.13%]`), cloud leakage `20.60%`, clear ground lost `3.38%`.
- Comparison & caveats: FP32 and FP16 coincide (`FP32 ≡ FP16`), while INT8 differs from FP32 by 3 false-deleted tiles at both $\tau = 0.8$ and $\tau = 0.5$, well inside overlapping 95% bootstrap CIs (no causal difference is claimed). Critically, $\tau=0.8$ allowed `3/418` (`0.72%`) false deletions on validation but overshot to `14/776` (`1.80%`) on test (`2.19%` for INT8); because validation also served as the early-stopping split during training, validation-selected $\tau$ is optimistic, and I left it untouched rather than retuning on test. Bandwidth numbers apply to CloudSEN12+'s stratified cloud mix (`20.4%` cloudy in test), not real-world orbital frequency. Regenerated `results/day25_policy_tradeoff.png` (`src/day28_replot_figures.py`) so dashed FP16 sits visibly over solid FP32 (`FP32 ≡ FP16`), with $\tau$ labels, circled validation-selected operating points, and 95% CI bars.

Day 26:
- Did today: Computed 10-bin ECE and plotted reliability diagrams (`results/day26_calibration.png`, regenerated via `src/day28_replot_figures.py` with dashed FP16 over solid FP32 and per-bin counts `n=...`) on validation (535 tiles) and test (975 tiles). Fitted single-parameter temperature scaling $T$ on validation NLL only and applied it to test.
- ECE before vs. after temperature scaling:
  - FP32: $T_{\text{val}} = 0.95$ | Val ECE `0.0195 → 0.0237` | Test ECE `0.0263 → 0.0269`
  - FP16: $T_{\text{val}} = 0.95$ | Val ECE `0.0194 → 0.0237` | Test ECE `0.0263 → 0.0269`
  - INT8: $T_{\text{val}} = 0.90$ | Val ECE `0.0210 → 0.0256` | Test ECE `0.0183 → 0.0206`
- Scaling effect: $T < 1.0$ reflects a global NLL fit across validation (whereas per-bin reliability is mixed across bins rather than uniformly under-confident). Applying $T < 1$ slightly increased 10-bin ECE on both splits because NLL and 10-bin ECE optimize different objectives. At $\tau = 0.95$, post-hoc scaling left FP32/FP16 false deletion unchanged (`0.13%`, `1/776`) and shifted INT8 from `0.13%` (`1/776`) to `0.26%` (`2/776`)—a single-tile difference—while ranking/AUROC stayed identical.
- **High-confidence check at $\tau = 0.95$ on test (consistent with $\ge 95\%$ cloudy purity; note that $\tau = 0.95$ is not a pre-registered operating point):**
  - FP32 & FP16 discarded `72` tiles; `98.61%` (`71/72`, 95% Wilson CI `[92.5%, 99.8%]`) truly had `cloud_pct >= 70`, with `1.67%` mean clear ground inside discarded tiles.
  - INT8 discarded `75` tiles; `98.67%` (`74/75`, 95% Wilson CI `[92.8%, 99.8%]`) truly had `cloud_pct >= 70`, with `1.87%` mean clear ground inside discarded tiles.
- FP32 vs. INT8: Static INT8 quantization did not degrade calibration on this test set (raw INT8 test ECE `0.0183` vs. FP32 `0.0263`, within small-sample bin variance). At $\tau = 0.95$, INT8 and FP32 differ by 3 discarded tiles (`75` vs. `72`) and by 1 vs. 2 false-deleted tiles after scaling, well within overlapping CIs; no causal claim is made about sharper confidence driving extra false deletions.

Day 27:
- Did today: Ran `src/day27_compression.py` across all 975 test tiles (`224x224` 3-band `int16`, `301,056` bytes/tile raw).
- Encoder check: OpenCV's JPEG 2000 encoder (`.jp2`, `IMWRITE_JPEG2000_COMPRESSION_X1000 = 1000`) passed a bit-exact 16-bit decode round-trip check (`np.array_equal`), additionally verified bit-exact on `50/50` seeded random test tiles, so no PNG stand-in was needed. Real spacecraft downlinks use CCSDS 122.0, which is cited rather than implemented.
- Compression ratio (all 975 tiles): Raw `293.53 MB` (`100.0%`) → Compress-only `145.83 MB` (`49.68%`), a **2.01x** lossless ratio.
- Four scenarios at $\tau = 0.8$ (val $\le 1\%$ cap):
  - FP32 & FP16 (847 tiles kept): Raw `293.53 MB` (`100.0%`) | Compress-only `145.83 MB` (`49.68%`) | Filter-only `254.99 MB` (`86.87%`) | Filter + Compress `126.76 MB` (`43.18%`).
  - INT8 (842 tiles kept): Filter-only `253.49 MB` (`86.36%`) | Filter + Compress `126.15 MB` (`42.98%`).
  - *(Note: lossless `.jp2` compression alone already reaches `49.68%` of raw volume, so $\tau = 0.8$ filtering adds `6.5` percentage points of raw reduction while still downlinking `85` of `199` cloudy test tiles.)*
- Four scenarios at $\tau = 0.5$ (val $\le 5\%$ cap):
  - FP32 & FP16 (790 tiles kept): Filter-only `237.83 MB` (`81.03%`) | Filter + Compress `118.11 MB` (`40.24%`).
  - INT8 (781 tiles kept): Filter-only `235.12 MB` (`80.10%`) | Filter + Compress `116.93 MB` (`39.84%`).
- Limitation: These tiles are `224x224` 3-band RGB thumbnails, not full-resolution 13-band Sentinel-2 scenes. Absolute megabytes mean nothing for an actual satellite downlink; only the relative percentages matter. Saved `results/day27_compression_bars.png`.

Day 28:
- Did today: Ran `src/day28_gate.py` and `src/day28_replot_figures.py` to assemble Tables II–IV and finalize all Phase 4 deliverables.
- Bugs & audit fixes resolved:
  - Fixed empty/float-formatted `batch` row parsing in the initial Table III script, switched GPU rows to the same-session paired benchmark (`results/day22_gpu_paired_bench.csv`) using `max(peak_vram_mb)` across repeats, and updated all standard deviations in Tables II and III from `ddof=0` to sample std (`ddof=1`) to match Phase 3.
  - Verified all 6 gate artifacts (`day28_table2_accuracy.csv`, `day28_table3_systems.csv`, `day28_table4_policy.csv`, and the three figures) and logged all gitignored ONNX SHA-256 hashes (Days 23–24).
- Table II summary (Test set, 3 seeds, `ddof=1`):
  - FP32 (seeds 0, 1, 2): `acc 0.9241 ± 0.0067`, `f1 0.8114 ± 0.0180`, `auroc 0.9637 ± 0.0047`, `ece 0.0243 ± 0.0051`.
  - FP16 (seeds 0, 1, 2): `acc 0.9241 ± 0.0067`, `f1 0.8114 ± 0.0180`, `auroc 0.9637 ± 0.0047`, `ece 0.0243 ± 0.0050` (negligible drift vs. FP32).
  - INT8 (seeds 0, 1, 2): `acc 0.9214 ± 0.0016`, `f1 0.8116 ± 0.0069`, `auroc 0.9636 ± 0.0046`, `ece 0.0240 ± 0.0063`.
- Pre-registration & cut-list audit:
  - Operating $\tau$ values and temperature $T$ were fitted strictly using validation data only (FP16 and INT8 test logits had already been saved earlier in the pipeline, but were never read during threshold or temperature selection). Validation also served as the early-stopping split, making validation-selected $\tau$ optimistic.
  - MobileNetV2 remained skipped (logged in Phase 3); TensorRT / GPU INT8 was skipped per the Section 5 cut list and documented in `README.md` Limitations alongside CPU-vs-GPU latency non-comparability, Seed-0-only policy/calibration/compression/latency runs, thumbnail sizing, and stratified cloud prevalence.
  - Phase 3 and Phase 4 FP32 Seed-0 summaries are anchored to `seed-0 v2` (`ckpt_seed0.pt`, SHA-256 `B7FDA259...D210`, logged on Day 22).
  - RQ1: Moving ResNet-18 from FP32 to FP16 on the RTX 5060 halves model size (`42.71 → 21.38 MiB`), reduces peak VRAM by `42.3%–62.7%`, and runs `1.59x–2.75x` faster in same-session paired benchmarks with no measurable change in accuracy, AUROC, or calibration across three seeds (`|Δacc| = 0`, `|ΔAUROC| ≤ 6e-5`, `|ΔECE| ≤ 4.2e-5`). Moving to static INT8 on the CPU (`i7-13620H`, `onnxruntime 1.32.0`) shrinks the ONNX model `3.96x` (`42.63 → 10.77 MiB`) and speeds up CPU inference by `2.40x–3.05x` with a `-0.27` percentage-point mean test accuracy delta across three seeds (`0.9241 ± 0.0067 → 0.9214 ± 0.0016`; Seed-0 val `-0.19` pp, Seed-0 test `+0.31` pp), within seed-to-seed variance across three seeds (no formal significance test at n=3), and virtually unchanged F1, AUROC, and ECE.
  - RQ2: Lossless `.jp2` compression alone reduces test downlink volume to `49.68%` of raw (`145.83 / 293.53 MB`); adding $\tau=0.8$ filtering brings volume to `43.18%` (`42.98%` for INT8)—an extra `6.5` percentage points of raw reduction while still downlinking `~85/199` cloudy tiles. High-confidence discards at $\tau=0.95$ (not a pre-registered operating point) are consistent with $\ge 95\%$ cloudy purity (`71/72 = 98.61%`, 95% Wilson CI `[92.5%, 99.8%]`, under `2%` average clear ground). However, choosing a $\le 1\%$ false-deletion cap on a 535-tile validation split that also controlled early stopping does not guarantee $\le 1\%$ out-of-sample—test false deletion rose from `3/418` (`0.72%`) on validation to `14/776` (`1.80%`) for FP32/FP16 and `2.19%` for INT8, showing deployments need a safety margin above validation estimates.
- What surprised me: Two things—first, that NLL-fitted temperature scaling ($T < 1$) slightly *worsened* 10-bin ECE because NLL and ECE optimize different objectives; second, how easily a strict 1% validation safety cap overshoots to ~2% on test even when AUROC is `0.96+`.
- Next: Update `README.md`, commit and push with `git tag phase4-frozen`, then start Phase 5 prep (verify exact titles and read abstracts for *Terminal Brain Damage* and *Bit-Flip Attack*, review IEEE-754 / INT8 binary notes, and sketch `src/fault_injection.py`).


## Phase 5

*(Honest logging note: I fixed the Phase 5 experimental plan before running anything, then ran all experiments straight through and compiled this written log at the gate directly from the raw CSVs and plots. I used coding and audit assistance to speed up script boilerplate and cross-check numbers, while executing every run, inspecting every figure, and verifying all outputs locally on my machine.)*

### Phase 5 Plan (Fixed Before Running)
| Item | Decision |
| :--- | :--- |
| Model & weight lock | ResNet-18 `seed-0 v2` (`ckpt_seed0.pt`, SHA-256 `B7FDA259...F2D1D210`). Every Phase 5 script runs `src/verify_weights.py` against `results/phase4_weights.sha256` at startup so silent weight drift cannot happen again. MobileNetV2 and TensorRT stay cut per Phases 3–4. |
| 5A Energy & throughput | Strict FP32 (`TF32=False`) vs. FP16 on the RTX 5060 (`batch` 1, 8, 32; 15 s idle baseline, 12 s timed windows, 30 s cooldown, 3 repeats, `ddof=1`, `pynvml` polled at 10 Hz). Report both total and net dynamic `(mean - idle)` mJ/patch. |
| 5B Thermal stress | 10-minute continuous `batch=32` inference (FP32, then 180 s cooldown, then FP16), logging 5 s windows for latency, throughput, temp, SM clock, power, and NVML throttle mask. |
| 5C Swath feasibility | Back-of-envelope Sentinel-2 coverage rate (`290 km × 6.7 km/s = 1,943 km²/s`) compared against measured throughput for both native `224×224` (`~387 p/s`) and resized `509×509` (`~75 p/s`) tiles. |
| 5D Fault injection | Target the 21 `Conv2d` and `Linear` weight tensors (`11,167,936` weights; `357,373,952` FP32 bits vs. `89,343,488` INT8 bits, exact `4.00×` ratio), leaving BatchNorm buffers and biases untouched so both precisions target identical layers. Compare native FP32 against GPU simulated per-channel symmetric weight-only INT8 (`sim_int8` — kept strictly separate from Phase 4's CPU ONNX Runtime W8A8 INT8). 30 independent `SeedSequence` trials per point across equal BER (`1e-8` to `1e-4`), equal flip count (`K` in 1, 10, 30, 100, 1000, 10000), and bit-position sweeps (`K` = 1, 10, 100). Evaluate accuracy and false deletion at the frozen Phase 4 thresholds (`τ = 0.8` and `τ = 0.5`) with zero retuning; `NaN`/`Inf` logits count strictly as incorrect and fail safe (`NaN >= τ` evaluates to `False`). |
| 5E & 5F | Test-set failure gallery (6 worst false deletions and 6 worst cloud leaks), test error rates by cloud/shadow bin, ROI-cluster bootstrap 95% CIs (`B = 2,000`), and exact two-sided McNemar tests (FP32 vs. INT8). |
| Conditions | Plugged in, NitroSense `Balanced`, `TF32=False`. |

---

### Day 29: Weight Guard, Energy-per-Patch (5A) & Swath Feasibility (5C)
- **Weight guard (`src/verify_weights.py`):** Verified all `12/12` gitignored `.pt` and `.onnx` files against `results/phase4_weights.sha256` before starting.
- **5A Energy & throughput (`src/day29_energy_and_feasibility.py`, `results/day29_energy_and_throughput.csv`, 18 runs):**
  - Idle GPU baseline measured `3.27 W` hands-off. With 30 s cooldowns after 12 s of continuous load, the GPU stayed in a warm steady state (`68–84°C` across runs) rather than cooling back to `~50°C`.
  - **Batch 1:** FP32 ran `241.3 ± 9.1 p/s` at `53.67 W` (total `222.29 ± 5.64 mJ/p`, net `208.72 ± 6.16 mJ/p`). FP16 ran `380.8 ± 38.2 p/s` at `39.88 W` (total `105.22 ± 7.65 mJ/p`, net `96.57 ± 6.73 mJ/p`) — `1.58×` faster and `52.7%` lower total energy (`53.7%` net, `2.11×` / `2.16×`).
  - **Batch 8:** FP32 ran `940.5 ± 2.6 p/s` at `57.39 W` (total `61.02 ± 0.95 mJ/p`, net `57.54 ± 0.96 mJ/p`). FP16 ran `2402.1 ± 7.4 p/s` at `51.50 W` (total `21.44 ± 0.24 mJ/p`, net `20.08 ± 0.25 mJ/p`) — `2.55×` faster and `64.9%` lower total energy (`65.1%` net, `2.85×` / `2.87×`).
  - **Batch 32:** FP32 ran `1540.2 ± 0.7 p/s` at `59.35 W` (total `38.53 ± 0.14 mJ/p`, net `36.41 ± 0.14 mJ/p`). FP16 ran `3127.7 ± 1.1 p/s` at `59.29 W` (total `18.96 ± 0.09 mJ/p`, net `17.91 ± 0.09 mJ/p`) — `2.03×` faster and `50.8%` lower total and net energy (`2.03×`).
  - **What stood out:** At `b=1` and `b=8`, FP16 saved more energy than its raw speedup because it also pulled noticeably less wattage. At `b=32`, both precisions slammed into the `~60 W` power ceiling, so energy savings matched the `2.03×` speedup exactly. Combining FP16 with batching (`FP32 b=1 → FP16 b=32`) cut total energy per patch by `11.7×` (`222.29 → 18.96 mJ/p`).
  - **Messy detail kept in the stats:** On repeat 1 at `b=1`, both precisions ramped slower (`230.8 p/s` at `49.81 W` for FP32; `336.73 p/s` at `38.41 W` for FP16) than on repeats 2–3 (`246.4–246.7` and `402.2–403.6 p/s`), which is why `b=1` has a much wider spread than `b=8` and `b=32`. I kept all 3 repeats instead of dropping repeat 1.
- **5C Swath feasibility (`290 km × 6.7 km/s = 1,943 km²/s`):**
  - If tiles cover `509×509` px at 10 m (`25.91 km²/patch`, matching CloudSEN12+ resized to `224×224`), the sensor generates `~75 patches/s`. Every config clears this comfortably, including Phase 4 CPU ORT INT8 `b=1, 4T` (`289.2 p/s`, `3.9×`) and GPU FP32 `b=1` (`241.3 p/s`, `3.2×`).
  - If tiles are native `224×224` px at 10 m (`5.02 km²/patch`), the swath needs `~387 patches/s`. Single-tile GPU FP32 (`241.3 p/s`) and CPU INT8 4T (`289.2 p/s`) fall short, GPU FP16 `b=1` (`380.8 ± 38.2 p/s`) sits right on the edge, and batched GPU clears it by `2.4×–8.1×`.
  - **Caveats I cannot gloss over:** This measures pure GPU forward-pass compute on synthetic tensors already in VRAM—no sensor readout, resizing, host-to-device transfer, or `.jp2`/CCSDS compression—and a `~50–60 W` laptop RTX 5060 is nowhere near a `5–15 W` flight payload.
- **Bit-flip literature & RQ3 hypothesis:** Read the abstracts of Hong et al. (*Terminal Brain Damage*, USENIX Security 2019) and Rakin et al. (*Bit-Flip Attack*, ICCV 2019). The mechanism we are testing is simple: in IEEE-754 FP32 (1 sign, 8 exponent, 23 mantissa bits), flipping the top exponent bit on a small trained weight can blow it up by `2^128 ≈ 3.4 × 10^38` (or turn it into `NaN`/`Inf`) and wreck the forward pass, whereas signed INT8 stays trapped in `[-128, 127]`, bounding any single bit flip to at most 128 quantization steps.

### Day 30: 10-Minute Sustained Thermal Test (5B) & Fault Harness (5D Prep)
- **5B Sustained thermal stress (`src/day30_thermal_test.py`, `results/day30_thermal_10min.csv`, `results/day30_thermal_drift.png`):**
  - Waited until the GPU cooled back to `50.2°C` (`4.68 W`, `180 MHz`) before starting two 10-minute continuous `batch=32` runs separated by a 180 s cooldown.
  - **FP32:** First-60 s mean latency was `21.3237 ms` (`1481.6 p/s`, peak SM `1837 MHz`, `59.67 W`). By the last 60 s it slowed to `24.4309 ms` (`1301.4 p/s`, `~1395 MHz` late, min `1350 MHz`, `~42.5–44.5 W`) — a `+14.57%` drift (`+19.3%` slower than cold Phase 4 paired `20.49 ms`). The GPU hit `87°C` at `57.7 s` and switched from power-cap (`throttle_mask = 4`) to thermal slowdown (`throttle_mask = 32`) at `77.7 s`.
  - **FP16:** First-60 s mean latency was `10.4392 ms` (`3031.9 p/s`, peak SM `2182 MHz`, `59.88 W`), drifting to `12.6221 ms` (`2516.4 p/s`, `~1650 MHz` late, min `1590 MHz`, `~41.4–42.8 W`) in the last 60 s — a `+20.91%` drift (`+23.9%` slower than cold Phase 4 paired `10.19 ms`). It started 4°C warmer in window 0 (`75°C` vs. `71°C` after the 3-minute cooldown), hit `87°C` at `32.5 s`, and entered `mask = 32` at `37.55 s` (FP32 at `77.73 s`).
  - **Takeaway:** Both curves flattened out completely by minute 4–5. Short burst benchmarks overstate sustained orbital-style throughput by `~15–24%`, yet even fully throttled, FP16 (`2516.4 p/s`) kept a `1.93×` throughput edge (`1.94×` lower median latency) over throttled FP32 (`1301.4 p/s`) and stayed `6.5×` above the `~387 p/s` native swath target.
- **5D Fault harness & unit tests (`src/fault_injection.py`, `4/4 PASS`):**
  - Confirmed target layer scope: `21` `Conv2d`/`Linear` tensors, `11,167,936` weights (`357,373,952` FP32 bits vs. `89,343,488` INT8 bits).
  - **BER = 0 parity:** FP32 matched saved Phase 4 Seed-0 test logits to `max_diff = 1.00e-05` (`acc = 0.9179`, `fd@0.8 = 0.0180`, `fd@0.5 = 0.0425`).
  - **Clean `sim_int8` baseline:** Simulated per-channel symmetric weight-only INT8 on GPU scored `acc = 0.9169` (`99.69%` agreement with FP32, differing on just 3 of 975 tiles), `fd@0.8 = 0.0206`, `fd@0.5 = 0.0425`. Keeping this explicitly distinct from Phase 4 CPU ORT W8A8 INT8 (`0.9210`).
  - **Bit integrity & smoke test:** Verified exact Hamming distance of `100` flipped bits with `clean_state` untouched. On a quick 10-flip MSB smoke test (`seed=7`), FP32 `b30` immediately collapsed to `0.7959` (`776/975`, the all-Keep majority floor) while INT8 `b7` stayed flat at `0.9169`.

### Day 31: Monte Carlo Bit-Flip Sweeps — Equal BER & Equal Flip Count (5D Part 1)
- **Run (`src/day31_ber_sweep.py`, `660` faulted evaluations = `30` seeds × `11` levels × `2` precisions; `results/day31_ber_sweep_raw.csv`, `day31_ber_sweep_summary.csv`, `day31_ber_and_flips.png`):**
- **Equal per-bit BER (physical radiation flux per memory cell, where FP32 takes `4.00×` more flips):**
  - **FP32:** Already wobbles at `1e-8` (~4.2 flips: `0.8935 ± 0.0197`, median `0.9179`, min `0.7026`, `5/30` collapses `< 85%`, `1/30` non-finite). At `1e-7` (~32.8 flips), `13/30` collapse (`0.8109 ± 0.0689`, median `0.8754`, `3/30` non-finite) and `4/30` suffer catastrophic false deletion (`> 10%`), peaking at **`98.32%`** with zero `NaN`s. At `1e-6` (~357.4 flips), `30/30` collapse (`0.4755 ± 0.1237`, median `0.6067`, `27/30` non-finite, `3/30` catastrophic FD up to **`98.71%`**). By `1e-5` (~3,586 flips) and `1e-4` (~35,697 flips), `30/30` trials go non-finite (`acc = 0.0346` and `0.0000`).
  - **Sim-INT8:** Barely flinched across the entire sweep — `0.9169 ± 0.0000` at `1e-8` (~1.1 flips), `0.9170` at `1e-7` and `1e-6`, `0.9186 ± 0.0006` at `1e-5` (~880 flips), and `0.9184 ± 0.0013` at `1e-4` (~8,928 flips, worst single trial `0.9067`). Across all `150` equal-BER trials: **`0` collapses, `0` non-finite trials, `0` catastrophic false deletions** (max `fd@0.8 = 2.32%`).
- **Equal total flip count (removing the `4×` bit-count advantage to isolate numerical representation):**
  - **FP32:** Clean at `K=1` (`0.9179`), then falls apart fast: `K=10` (`0.8796 ± 0.0286`, median `0.9179`, `7/30` collapses, `2/30` nf, `1/30` catFD max `39.95%`), `K=30` (`0.8581 ± 0.0420`, median `0.9179`, `11/30` collapses, `2/30` nf, max `fd@0.8 = 77.96%`), `K=100` (`0.7546 ± 0.0639`, median `0.7959`, `26/30` collapses, `8/30` nf, `3/30` catFD max `57.35%`), `K=1000` (`0.3021 ± 0.1229`, median `0.1487`, `30/30` collapses, `29/30` nf, `5/30` catFD max `97.16%`), and `K=10000` (`0.0176 ± 0.0355`, `30/30` nf).
  - **Sim-INT8:** Held `0.9169–0.9186` all the way through `K=1000` and only nudged down to `0.9143 ± 0.0042` at `K=10000` (median `0.9185`, min `0.8687`), with **`0/180` collapses, `0/180` non-finite outputs, and max `fd@0.8 = 2.32%`**. Interestingly, mild symmetric weight noise slightly dampened logit confidence at high flip counts, trimming `fd@0.8` from `2.06%` to `1.54%` (and `bw@0.8` from `13.33%` to `11.94%`).
- **Biggest surprise & safety lesson:** Seeing FP32's false-deletion curve drop back to `0.0%` at `1e-5`, `1e-4`, and `K=10000` looked weird at first glance until I checked the raw rows: it is **not** recovery. At extreme fault rates, `30/30` trials blow up to `NaN`, and our fail-safe discard rule (`NaN >= 0.8` is `False`) keeps every tile. The far nastier mission hazard is moderate radiation (`1e-7`–`1e-6` or `K=10`–`1000`), where FP32 stays *finite*—slipping right past a `NaN` watchdog—deleting up to `98.32%` of keep-worthy clear ground with `0%` non-finite tiles at `1e-7` (and `98.71%` at `1e-6`, where `99.3%` of tiles remained finite and `0.7%` hit `NaN`). Because FP32 trials are heavily skewed/bimodal, I am reporting median/IQR and collapse counts alongside mean ± 95% Student-t CI everywhere.

### Day 32: Bit-Position Sensitivity Breakdown (5D Part 2)
- **Run (`src/day32_bit_position.py`, `840` faulted evaluations = `28` blocks × `30` seeds; `results/day32_bit_position_raw.csv`, `day32_bit_position_summary.csv`, `day32_bit_position.png`):**
- **Per-bit sweep (`K=10`):** Every tested FP32 mantissa bit (`b0, b11, b22`), lower exponent bit (`b23–b29`), and sign bit (`b31`) stayed pinned at `0.9179–0.9180` (`0/30` collapses, `0/30` non-finite). Every single one of the 8 Sim-INT8 bits (`b0–b7`) stayed flat at `0.9169–0.9171` (`0/30` collapses). The **only** bit that broke anything was **`FP32 b30` (exponent MSB)**, dropping accuracy to `0.5276 ± 0.1160` (median `0.7103`, `30/30` collapses, `26/30` non-finite, `2/30` catastrophic FD up to `91.62%`). That makes complete sense from IEEE-754 math: trained weights are mostly small (`|w| < 2`), so `b30` is `0` and flipping it multiplies by `2^128`, while `b29` is usually `1` and flipping it merely shrinks the weight toward `0`.
- **Single-flip test (`K=1`):** Just **one single bit flip** in `FP32 b30` collapsed `23/30` trials (`0.7142 ± 0.0768`, median `0.7959`, min `0.2544`) and caused **`8/30` catastrophic false deletions (`> 10%`)**, peaking at **`92.27%`** with zero `NaN`s (only `4/30` single-flip trials triggered `NaN`). Picking one random bit across all 8 FP32 exponent bits (`b23–b30`) collapsed `3/30` (`0.9044 ± 0.0112`), right near the expected `1/8` hit rate on `b30`. A single flip in `INT8 b7` stayed dead flat at `0.9169 ± 0.0000` (`0/30` collapses).
- **Heavy-flip field groups (`K=100`):** FP32 Mantissa (`0.9179`), FP32 Sign (`0.9177`), INT8 Low `b0–6` (`0.9169`), and INT8 MSB `b7` (`0.9180`) all had `0/30` collapses and `0/30` non-finite outputs, while FP32 Exponent (`b23–30`) collapsed `30/30` (`0.5079 ± 0.1232`, `25/30` non-finite, `3/30` catFD up to `68.17%`).

### Day 33: Failure Analysis (5E) & Statistical Verification (5F)
- **Run (`src/day33_error_and_stats.py`; `results/day33_failure_gallery.png`, `day33_failure_cases.csv`, `day33_test_error_bins.csv`, `day33_bootstrap_ci.csv`, `day33_mcnemar_tests.csv`):**
- **5E Test error bins (`N = 975`, 3-seed mean at `τ = 0.5`):**
  - Errors cluster around the decision boundary and thin cloud: `70–90%` (`n=124`, FP32 `25.54%` / ORT-INT8 `22.04%`), `90–100%` (`n=75`, `10.67%` / `8.44%`), `30–70%` (`n=312`, `10.15%` / `11.43%`), and heavy shadow `≥ 20%` (`n=165`, `9.90%` / `11.11%`), versus `0.71%` (`0–10%`, `n=281`) and `0.36%` (`10–30%`, `n=183`).
  - The reassuring check for mission safety: at `τ = 0.8` (Seed 0 FP32), **`0/464` (`0.00%`)** tiles with `< 30%` cloud were falsely deleted. All `14/776` (`1.80%`) false deletions came strictly from `30–70%` partial-cloud tiles (`41%–68%` cloud).
- **5E What the failure gallery actually showed:**
  - *Worst false deletions:* My #1 and #3 worst false deletions (`idx=224`: `60%` cloud, `40%` shadow, **`0%` clear**, `p_fp32=0.9742`; `idx=747`: `63%` cloud, `35%` shadow, **`2%` clear**, `p_fp32=0.9156`) are completely overcast/shadowed tiles that only count as "keep-worthy" because our pre-registered label rule grouped cloud shadow with ground.
  - *Worst cloud leaks:* Every one of the top-6 leaks (`71%–95%` cloudy, `0%` shadow, `p_fp32 = 0.0019–0.0193`) looks like dark, uniform, bluish-gray thin cirrus or haze over dark water/land. That visually confirms the physical limit of 3-band RGB without Sentinel-2's `B10` (1375 nm cirrus) or SWIR bands.
  - *Visual fix caught before freezing:* On the first render of `day33_failure_gallery.png`, the bottom-row subplot titles (`Leak #1–#6`) slightly overlapped the bottom edge of the top-row images. Increased figure height to `6.8` and added `hspace=0.38`, then re-ran and verified clean spacing.
- **5F ROI-cluster bootstrap 95% CIs (`B = 2,000`, Seed 0):**
  - FP32 & FP16: Acc `0.9179 [0.8995, 0.9354]`, F1 `0.7917 [0.7430, 0.8374]`, AUROC `0.9596 [0.9445, 0.9725]`, ECE `0.0263 [0.0196, 0.0461]`.
  - CPU ORT INT8: Acc `0.9210 [0.9015, 0.9385]`, F1 `0.8041 [0.7538, 0.8458]`, AUROC `0.9618 [0.9477, 0.9737]`, ECE `0.0183 [0.0168, 0.0380]`.
  - GPU `sim_int8`: Acc `0.9169 [0.8974, 0.9344]`, F1 `0.7885 [0.7384, 0.8321]`, AUROC `0.9587 [0.9431, 0.9720]`, ECE `0.0309 [0.0231, 0.0488]`.
- **5F Exact two-sided McNemar tests (paired test predictions):**
  - Every comparison came back non-significant (`p > 0.05`): FP32 vs. CPU ORT INT8 accuracy at `τ = 0.5` across Seeds 0, 1, 2 (`b=3, c=6, p=0.5078`; `b=12, c=4, p=0.0768`; `b=12, c=9, p=0.6636`), FP32 vs. `sim_int8` (`b=2, c=1, p=1.0000`), FP32 vs. FP16 (`b=0, c=0, p=1.0000`), and Seed-0 `τ = 0.8` false deletions on the `776` keep-worthy tiles (`b=0, c=3, p=0.2500` for ORT-INT8; `b=0, c=2, p=0.5000` for `sim_int8`) as well as total discards (`b=3, c=8, p=0.2266`).

### Day 34: Phase 5 Gate & Honest Wrap-Up
- **Master summary & deliverables (`src/day34_gate.py`):** Compiled `results/day34_phase5_master.csv` directly from the raw Phase 5 CSVs (`22/22` scripts, CSVs, and figures verified; `ckpt_seed0.pt` SHA-256 unchanged).
- **RQ1 (complete):** Across Phases 4 and 5, FP16 cuts GPU energy/patch by `50.8%–65.1%` (`11.7×` combined with `batch=32`) and sustains `2516.4 p/s` (`1.93×` throughput / `1.94×` lower latency over FP32) even under `87°C` thermal throttling, while static INT8 cuts size `3.96×` and CPU latency `2.40×–3.05×` with no statistically significant accuracy or `τ = 0.8` false-deletion degradation (`p > 0.05` on every McNemar test).
- **RQ3 (answered):** Across `1,500` Monte Carlo bit-flip trials (`660` Day 31 + `840` Day 32), the bounded-damage hypothesis held even more sharply than expected. Native FP32 fragility is localized 100% to exponent MSB `b30` (`23/30` collapses from a single `b30` flip; strictly finite (`0%` `NaN`) false deletions up to `98.32%` at `BER = 1e-7` and `92.27%` from a single `b30` flip; up to `98.71%` at `1e-6` with `99.3%` finite tiles), while simulated per-channel INT8 recorded `0` collapses, `0` non-finite outputs, and `≤ 2.32%` false deletion across every BER level (`1e-8`–`1e-4`), equal flip count (`K = 1`–`10,000`), and all 8 bit positions.
- **Boundaries going into Phase 6:** Keep all five limits front and center in the paper — (1) consumer laptop hardware (`RTX 5060`, `i7-13620H`) as a proxy rather than flight hardware, (2) compute-only swath throughput without ingest/resize/compression overhead, (3) RGB `224×224` thumbnails missing `B10`/SWIR cirrus bands, (4) single architecture (`ResNet-18`) with downstream sweeps on `seed-0 v2`, and (5) software-only weight flips on `Conv2d`/`Linear` layers (excluding BatchNorm, activations, multi-bit upsets, and FP16 flips) with Phase 5 GPU weight-only `sim_int8` kept separate from Phase 4 CPU ORT W8A8 INT8 and assuming the tiny per-channel FP32 scale factors (`~19.2 KB`) are protected by ECC/TMR.
- **Next:** `README.md` updated and audited; tag `phase5-frozen` and move to Phase 6 (Overleaf IEEE paper draft).