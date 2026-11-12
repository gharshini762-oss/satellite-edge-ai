# Dataset Card: CloudSEN12+ L1C Binary Cloud-Screening Subset (`cloud_export_v2`)

## 1. Source, Version, and License
- **Dataset:** CloudSEN12+ (v1.1.2, Level-1C Top-of-Atmosphere reflectance), accessed via `tacoreader` (`tacofoundation:cloudsen12-l1c` / Hugging Face `tacofoundation/cloudsen12`).
- **License:** CC0 1.0 Universal (Public Domain), per the dataset documentation.
- **Citations (with DOIs):**
  - Aybar, C. et al. (2022). *CloudSEN12, a global dataset for semantic understanding of cloud and cloud shadow in Sentinel-2*. Scientific Data, 9(1), 782. DOI: `https://doi.org/10.1038/s41597-022-01878-2`
  - Aybar, C. et al. (2024). *CloudSEN12+: The largest dataset of expert-labeled pixels for cloud and cloud shadow detection in Sentinel-2*. Data in Brief, 56, 110852. DOI: `https://doi.org/10.1016/j.dib.2024.110852`

## 2. Selection Rule, Split Method & ROI-Level Leakage Check
- **Selection Rule:** Filtered strictly to high-quality expert-labeled patches (`label_type == 'high'`) with native projection size $509 \times 509$ (`real_proj_shape == 509`), yielding 10,000 total patches (`results/manifest_all.csv`, excluding 847 high-quality $2000 \times 2000$ patches and all scribble/unlabeled patches).
- **Split Method & ROI-Level Check:** Official `tortilla:data_split` assignment grouped by region of interest (`roi_id`). Verified at the ROI level that no `roi_id` appears in more than one split via `assert (man.groupby('roi_id')['split'].nunique() == 1).all()`.
- **Exported Counts & Class Balance (`results/manifest_export.csv`):**
  - **Train:** 4,000 patches (stratified subsample across 10% cloud-cover deciles with `random_state=42` from the 8,490 official train patches; binary cloudy fraction = `0.207250`, vs. `0.207303` across all 8,490 train patches).
  - **Validation:** 535 patches (100% of official validation split; binary cloudy fraction = `0.218692`).
  - **Test:** 975 patches (100% of official test split; binary cloudy fraction = `0.204103`).

## 3. Labeling Rule & Mask Alignment Verification
- **Cloud Fraction:** `cloud_pct = thick_percentage + thin_percentage`.
- **Position-Indexed Mask Verification (`cloud_export_v2`):** Each patch is read by its 0-based `tacoreader` table position (`pos`) and checked against its cropped $509 \times 509$ pixel-level semantic mask (`classes 1 + 2`). Across all 5,510 exported patches, the absolute difference between mask-derived cloud percentage and metadata `cloud_pct` is verified to be $\le 3.0\%$ (`train`: mean diff `0.24%`, max `1.92%`; `validation`: mean diff `0.24%`, max `1.33%`; `test`: mean diff `0.26%`, max `2.61%`).
- **Binary Target (`label`):**
  - `1` (**Cloudy / Discard candidate**): `cloud_pct >= 70.0%` ($T = 70\%$ fixed before any model was trained, after viewing the cloud-percentage histogram).
  - `0` (**Keep-worthy tile**): `cloud_pct < 70.0%` (trained on all patches without dropping intermediate cloud bands).
- **Cloud Shadow Handling:** `cloud_shadow_percentage` counts as ground (`0`), not cloud, and is retained in `results/manifest_export.csv` for downstream error analysis.
- **Imbalance Policy:** No resampling or class reweighting is used during training; probability calibration (reliability diagram and Brier score) is measured downstream during Phase 4 evaluation.

## 4. Preprocessing & Normalization Statistics
1. **Bands:** Uses only Sentinel-2 L1C RGB bands `B04` (Red), `B03` (Green), and `B02` (Blue), corresponding to 1-indexed `rasterio` channels `[4, 3, 2]`; the other ten Sentinel-2 spectral bands are unused.
2. **Padding Crop:** Cropped the 3-pixel zero-padding strip along the bottom and right edges (`img[:, :509, :509]`).
3. **Whole-Patch Resize:** Resized full $509 \times 509$ patches to $224 \times 224$ via area interpolation (`cv2.INTER_AREA`) and stored as `int16` digital numbers (`1,659,081,360` bytes / ~1.66 GB across 12 `.npz` shards in `cloud_export_v2`). Whole-patch resizing preserves alignment with the whole-patch `cloud_pct` label.
4. **Reflectance Scaling & Offset Verification:** ESA introduced the `+1000` L1C radiometric offset in Processing Baseline `04.00` (January 25, 2022), whereas CloudSEN12 L1C imagery was acquired prior to that baseline (2015-12-16 to 2020-11-20) and stores direct TOA reflectance digital numbers (`DN / 10,000`, verified via `stac:time_start` timestamps and local dark/clear-sky pixel percentiles in `train_X.npy`). Values are divided by `10,000.0`, clamped to `[0.0, 1.0]`, and standardized using training-split-only statistics (`results/norm_stats.json`, computed on `train_X.npy[::10]`):
   - **Per-channel Mean (R, G, B):** `[0.219371, 0.213987, 0.229659]`
   - **Per-channel Std (R, G, B):** `[0.208468, 0.186713, 0.197026]`
5. **Training Augmentations:** Random horizontal flip ($p = 0.5$), random vertical flip ($p = 0.5$), and random $90^\circ$ rotations ($k \in \{0, 1, 2, 3\}$). No color jitter or brightness changes are applied.

## 5. Known Biases & Limitations
- **Curated Cloud-Cover Distribution:** Per Aybar et al. (2022, Section "Methods — Cloud cover stratification"), each CloudSEN12 ROI was intentionally sampled across five cloud-cover strata (clear, low-cloudy, almost-clear, mid-cloudy, cloudy) to balance cloud percentages across scenes. Because of this stratified curation (producing a ~25% clear-sky peak and a near-uniform spread from 10% to 100% in our histogram), raw discard rates on this dataset do not reflect natural orbital cloud encounter frequencies without reweighting.
- **Spectral & Spatial Scope:** Uses only 3-band RGB (`B04`, `B03`, `B02`) resized to $224 \times 224$ while leaving the other 10 Sentinel-2 bands unused, serving as a proxy for a low-cost RGB cubesat sensor rather than a full multispectral instrument.