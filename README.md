# Deep-Water Image Segmentation: Classical vs Hybrid Quantum Pipeline

## Project Overview
This repository contains the implementation of a comparative study for underwater image segmentation, aimed at identifying subsea critical infrastructure (e.g., pipelines, valves). The project evaluates classical deep learning baselines (e.g., SegFormer, U-Net) against Hybrid Quantum-Classical Neural Networks (HQ-UNet).

## Repository Structure
* `notebooks/`: Interactive Jupyter/Colab notebooks for exploratory data analysis, optical preprocessing (CLAHE, MSRCR, UDCP), and metric tracking.
* `src/`: Modular Python scripts containing preprocessing filters, model architectures, and utility functions.
* `requirements.txt`: Python dependencies required to reproduce the environment.
* `requirements-inference.txt`: Lightweight dependencies for offline qualitative inference notebooks.
* `docs/`: Methodological notes and fine-tuning proposals.

## Phase 1: Optical Preprocessing & Data Pipeline
Currently addressing the heterogenous nature of deep-water datasets (light absorption, scattering) to establish a clean, standardized dataset for supervised training.

## Benchmark and Inference Workflow

Public supervised benchmarks such as SUIM or DUT should be structured for MMSegmentation with:

```text
images/training
images/validation
annotations/training
annotations/validation
```

For local development, the prepared SUIM benchmark is expected at:

```text
data/SUIM
```

This local dataset folder is ignored by Git because it contains many image and mask files. For Colab or shared execution, the same structure can be reproduced in the project Drive.

Use `src/data/prepare_mmseg_dataset.py` to copy matched image/mask pairs into the MMSegmentation structure, and `src/data/validate_mmseg_dataset.py` to verify the result.

Internal processed imagery from `Under_Water_Dataset/Dataset_Processed` is used for qualitative offline inference comparisons, not as a substitute for public benchmark masks unless matching annotations are available.
