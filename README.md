# Onboard Cloud Filtering with Quantized Neural Networks (work in progress)

## Goal (planned)
Compare FP32, FP16, and INT8 cloud-vs-clear classifiers on Sentinel-2
patches under satellite downlink limits.
Planned measurements: accuracy, model size, latency, and memory; the trade-off between bandwidth saved and wrongly deleted clear tiles under a high-confidence deletion threshold; and software bit-flip fault injection.
A laptop GPU stands in for onboard hardware, so results will be relative comparisons, not in-orbit performance.

## Status (last updated 2026-11-01)
- Phase 1 (environment and measurement tools): complete
- Phase 2 (data): not started
- No model has been trained or quantized yet.

## Environment
- Device: Acer Nitro ANV15-52
- GPU: NVIDIA GeForce RTX 5060 Laptop GPU, 8 GB VRAM, driver 621.01
- Python 3.12.10
- PyTorch 2.11.0+cu128, CUDA 12.8 build
- Test conditions: "not recorded"

## What's here so far
- src/verify_env.py: checks that PyTorch can run math on the GPU
- src/day4_image_basics.py: images as NumPy arrays, BGR-to-RGB
  correction, 224x224 resize, 12-bit to 8-bit rescaling error
- src/day5_gpu_benchmark.py: synchronized v/s unsynchronized GPU 
  timing over 100 trials
- src/telemetry.py: logs GPU temperature, clock, VRAM, and power at 5 Hz
- results/: figures and logs produced by the scripts above

## Phase 1 measurements
- GPU timing: synchronized(Measures actual GPU compute time) 22.4552 ms (median) v/s unsynchronized (Measures CPU handoff only) 0.0000 ms (median). GPU calls return before the work finishes, so timing without synchronize() undercounts.
- Telemetry (10-second stress test): clock rose from about 300 MHz to about 1750 MHz; temperature rose from about 56 C to about 78 C. 
The temperature trace oscillates regularly, and the cause has not been investigated yet.

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
2. Data: Sentinel-2 patches with cloud labels
3. FP32 baseline model and benchmark
4. FP16/INT8 quantization and threshold sweep
5. Full benchmarks and bit-flip tests
6. Paper