# DeepONet for Reaction-Diffusion Problems

[![PyTorch Lightning](https://img.shields.io/badge/PyTorch%20Lightning-1.9+-blue.svg)](https://pytorch-lightning.readthedocs.io/)
[![Weights & Biases](https://img.shields.io/badge/Weights_&_Biases-0.15+-yellowgreen)](https://wandb.ai)

PyTorch Lightning implementation of DeepONet for solving reaction-diffusion equations, with Weights & Biases integration for experiment tracking.

## Purpose

This repository modernizes the original DeepONet architecture by:
- Migrating to PyTorch Lightning for cleaner, more maintainable code
- Adding Weights & Biases integration for experiment tracking
- Implementing a structured training pipeline with checkpointing
- Providing both training and testing modes

## Original Paper

The DeepONet architecture was introduced in:
1. Lu, L., Jin, P., Pang, G., Zhang, Z., & Karniadakis, G. E. (2021). **Learning nonlinear operators via DeepONet based on the universal approximation theorem of operators**. *Nature Machine Intelligence*, 3(3), 218-229. [DOI](https://doi.org/10.1038/s42256-021-00302-5)


## Features

- 🚀 **PyTorch Lightning** implementation for cleaner training loops
- 📊 **Weights & Biases** integration for experiment tracking
- 💾 Automatic checkpointing and model saving
- ⚙️ Configurable via command line arguments
- 🔍 Both training and inference modes
- 🛠️ Early stopping and model checkpointing callbacks

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/deeponet-reaction-diffusion.git
cd deeponet-reaction-diffusion

# Create conda environment (recommended)
conda create -n deeponet python=3.11
conda activate deeponet

# Install requirements
pip install -r requirements.txt

# Install Weights & Biases
wandb login