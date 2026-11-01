
## Satellite Edge AI: Hardware-Efficient Image Compression & Telemetry

## Overview
This repository contains the Phase-1 of research's foundational codebase and hardware benchmarking suite, for evaluating neural network efficiency on resource-constrained edge devices. Designed for Earth observation applications (such as Sentinel-2 satellite downlink optimization), this project benchmarks quantization trade-offs, GPU tensor throughput, and hardware telemetry under continuous workloads.

## Hardware Environment
- Device: Acer Nitro ANV15-52
- GPU: NVIDIA GeForce RTX 5060 Laptop GPU (8GB VRAM, sm_120 architecture)
- Environment: Python 3.12, PyTorch (CUDA 12.8), OpenCV, NVML Telemetry

## Project Structure
- src/verify_env.py: CUDA capability and hardware verification script.
- src/day4_image_basics.py : NumPy array inspection, OpenCV BGR-to-RGB correction, 224x224 resizing, and 12-bit (0-4095) to 8-bit (0-255) quantization error analysis.
- src/day5_gpu_benchmark.py: Synchronized vs. unsynchronized matrix multiplication timing benchmarks across 100 trials.
- src/telemetry.py: Live NVML hardware sensor logger (temperature, power draw, core clock, and VRAM) operating at 5 Hz with peak allocation tracking.
- results: Contains generated telemetry logs (`.csv`), performance plots (`.png`), and pipeline verification figures.
- log.md: Chronological development and research log.