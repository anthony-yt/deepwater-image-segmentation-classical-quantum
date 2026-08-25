# Deep-Water Image Segmentation: Classical vs Hybrid Quantum Pipeline

## Project Overview
This repository contains the implementation of a comparative study for underwater image segmentation, aimed at identifying subsea critical infrastructure (e.g., pipelines, valves). The project evaluates classical deep learning baselines (e.g., SegFormer, U-Net) against Hybrid Quantum-Classical Neural Networks (HQ-UNet).

## Repository Structure
* `notebooks/`: Interactive Jupyter/Colab notebooks for exploratory data analysis, optical preprocessing (CLAHE, MSRCR, UDCP), and metric tracking.
* `src/`: Modular Python scripts containing preprocessing filters, model architectures, and utility functions.
* `requirements.txt`: Python dependencies required to reproduce the environment.

## Phase 1: Optical Preprocessing & Data Pipeline
Currently addressing the heterogenous nature of deep-water datasets (light absorption, scattering) to establish a clean, standardized dataset for supervised training.