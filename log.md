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
- Did today: read about Sentinel-2 (bands, L1C vs. L2A, tile structure). Source: [ESA docs / other].
- Next: Day 8, make a Copernicus Data Space account.
 

PHASE-2

Day 8: 
- Did today: Created and verified on Copernicus Data Space Ecosystem.
- observed: What a Band is- Instead of capturing just standard Red, Green, and Blue (RGB), Sentinel-2 captures 13 separate wavelength channels (bands) from visible light (10m resolution) to short-wave infrared (20m/60m resolution) to detect surface features and high-altitude cirrus clouds.
L1C vs. L2A- L1C is Top-of-Atmosphere (TOA) reflectance containing raw sensor measurements including clouds and atmospheric haze, whereas L2A is Bottom-of-Atmosphere (BOA) surface reflectance after heavy ground-station atmospheric correction.
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
- Did today: Loaded the CloudSEN12+ (`cloudsen12-l1c`) and generated a 5-sample validation grid of RGB images and masks, and analyzed image edge padding.
- Observed: Total rows in dataset are 50,247. The patches contain a 3-pixel zero-padding strip at the edges (expanding native 509x509 to 512x512). The label values inside this zero-padded strip evaluate entirely to `0` ("clear"), confirming that uncropped padded patches would artificially inflate clear-sky fraction metrics. Verified scaling factor 0.0001 for reflectance conversion.

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
- Classes: clear, thick cloud, thin cloud, cloud shadow. Label types: high-quality, scribble, no-annotation. I will use high-quality labels only for evaluation.
- Patch size: 509x509 px (5.09 km at 10 m) and 2000x2000 px, padded to 512x512 and 2048x2048 with zeros on the left and bottom.

- Next: Day 11, define labels and cut patches.

Day 11:
- Did today: Computed per-patch cloud fraction (`thick_percentage + thin_percentage`) on the 10,000 high-quality 509x509 patches, plotted `results/cloud_hist.png`, defined labeling thresholds, and ran `src/make_patches.py` in Colab to export 900 resized 224x224 RGB patches (`245` clear, `199` cloudy, `456` ambiguous_eval_only).
- Observed: Histogram has a spike near 0% (~25% of patches) and a flat spread up to 100% . Train, val, and test splits have matching distributions (means 36.6%, 37.6%, 37.6%; medians 31%, 32%, 33%). Across all high-quality 509-px patches, 35.4% are >=50% cloudy, 20.8% are >=70% cloudy, and 7.6% are >=90% cloudy.

  - Clear (`0`, keep): `cloud_pct <= 10%` (~30% of patches). Cloudy (`1`, discard): `cloud_pct >= 70%` (~20.8% of patches).

  - Ambiguous mid-range (`10% < cloud_pct < 70%`) and heavy shadow (`>= 50%`) are excluded from binary training only, but kept in evaluation/test sets.

  - Cropped 3-pixel zero-padding to 509x509 and resized to 224x224 (coarsening resolution ~2.3x) so full-patch cloud fraction matches the visible patch.

 - Self verification:
Band descriptions (1-indexed): ('B01', 'B02', 'B03', 'B04', 'B05', 'B06', 'B07', 'B08', 'B8A', 'B09', 'B10', 'B11', 'B12')
Bands [4, 3, 2] are: ['B04', 'B03', 'B02']
Valid row range: 0..508 (len 509), Valid col range: 0..508 (len 509)
Max abs difference (cropped 509x509 mask cloud % vs metadata thick+thin %): 54.0000%
Average streaming/read time per patch: 4.47 s/patch



- Next: Day 12, build leak-proof splits (verify zero scene overlap across train/val/test, check class balance per split, and save `manifest.csv`).

 
 # Raw data of day 10/11:

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
  - Asserted zero scene overlap using `assert (man.groupby('roi_id')['split'].nunique() == 1).all()`, guaranteeing zero data leakage across splits.
  - Set `N_TRAIN = 4000` based on pilot streaming speeds (~4.47s per patch). Exporting all 8,490 training patches would require ~10+ hours; 4,000 provides an optimal, stratified training subset while keeping execution times manageable.
  - Generated `manifest_export.csv` and executed the sharded export loop to save test and validation arrays into Google Drive (`/content/drive/MyDrive/cloud_export`).
- Observed & Verified:
  - **Leakage check:** Zero ROI/scene overlaps confirmed successfully.
  - **Export outputs:** Successfully streamed and saved `test_000.npz` (500 patches), `test_001.npz` (475 patches), `validation_000.npz` (500 patches), and `validation_001.npz` (35 patches).

  results/manifest_all.csv 
             count      mean
split                      
test          975  0.204103
train        8490  0.207303
validation    535  0.218692

results/manifest_export.csv 
             count      mean
split                      
test          975  0.204103
train        4000  0.207250
validation    535  0.218692
- Next: Day 13, implement `src/dataset.py` 

- Mistakes / Debugging Notes (Day 12 -> Day 13 transition):
  1. Ternary vs. Binary Label Confusion: Initially coded `label` with `-1` for ambiguous mid-range clouds (`10% < cloud_pct < 70%`) which produced negative label means (`-0.293` on test, `-0.276` on train). Caught the mismatch against the Section 4 pre-registered rule
   (`label = 1` if `cloud_pct >= 70` else `0` across all patches) and updated `manifest_all.csv`, `manifest_export.csv`, and the shard labels to strict binary `0/1` (giving the true cloudy share: `20.73%` train, `21.87%` validation, `20.41%` test).
  2. Shard Resolution & Dtype Mismatch: The initial Day 12 `process()` function exported `test` and `validation` shards as unresized `(N, 3, 509, 509)` `float32` arrays instead of resized `(N, 3, 224, 224)` `int16` arrays (`cv2.INTER_AREA`). Instead of re-streaming 1,510 patches over HTTP (~1.5 hours), wrote an in-place conversion script in Colab to resize the existing Drive shards to `(N, 3, 224, 224)` `int16` before exporting the `train` shards.