---
name: System Hardware Specs
description: Full hardware specs for the development/trading PC — determines what models can run locally
type: user
---

## Hardware

| Component | Spec |
|-----------|------|
| **CPU** | AMD Ryzen 5 5600X — 6 cores / 12 threads |
| **RAM** | 16 GB (17.1 GB raw) |
| **GPU** | NVIDIA GeForce RTX 3060 — **12 GB VRAM** |
| **CUDA** | 12.1 (compute capability 8.6) |
| **Disk C:** | 500 GB, 203 GB free |
| **Disk D:** | 480 GB, 448 GB free |
| **OS** | Windows 11 Home 64-bit |

## Software

| Component | Version |
|-----------|---------|
| Python | 3.11.9 |
| PyTorch | 2.5.1+cu121 |
| CUDA | 12.1 |
| NVIDIA Driver | 591.86 |

## Local LLM Capacity

With 12 GB VRAM:
- **7B models** (Llama 3 7B, Mistral 7B, Qwen 7B): YES — fits in ~4-6 GB, room for KV cache
- **13B models** (Llama 13B, Qwen 14B): YES — fits in ~8-10 GB at Q4 quantization
- **34B models** (CodeLlama 34B, Yi 34B): TIGHT — needs Q4 quant, ~12 GB, no room for other GPU tasks
- **70B+ models**: NO — won't fit even quantized

Sweet spot: **7B-13B models quantized to Q4/Q5** — fast inference, fits with room for trading system GPU tasks.

Can run simultaneously with CNN training/inference since models are small.

## Limitations
- 16 GB RAM limits large dataframe operations (1s data for full year would need chunking)
- 12 GB VRAM shared between trading CNN + local LLM if running both
- 6 cores — parallel data processing limited vs higher-end CPUs
