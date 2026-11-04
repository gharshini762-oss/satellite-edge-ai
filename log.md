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
