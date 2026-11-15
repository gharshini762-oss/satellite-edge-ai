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