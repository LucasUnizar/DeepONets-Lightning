# DeepONet for 1D Burgers' and Reaction-Diffusion Equations

[![PyTorch Lightning](https://img.shields.io/badge/PyTorch%20Lightning-2.0+-blue.svg)](https://pytorch-lightning.readthedocs.io/)
[![Weights & Biases](https://img.shields.io/badge/Weights_&_Biases-0.16+-yellowgreen)](https://wandb.ai)

A clean and adaptable PyTorch Lightning implementation of the Deep Operator Network (DeepONet) for solving one-dimensional Burgers' and reaction-diffusion partial differential equations (PDEs), with integrated Weights & Biases for robust experiment tracking and visualization.

**Repository Link:** [https://github.com/LucasUnizar/DeepONets-Lightning](https://github.com/LucasUnizar/DeepONets-Lightning)

## Purpose and Scope

This repository provides a modernized and streamlined adaptation of the DeepONet architecture, specifically tailored for learning the solution operators of 1D Burgers' and reaction-diffusion problems. The core objectives of this project are:

-   **PyTorch Lightning Integration**: To leverage the capabilities of PyTorch Lightning for cleaner, more organized, and maintainable code, simplifying the training and evaluation workflows.
-   **Weights & Biases Integration**: To enable comprehensive experiment tracking, logging, and visualization through Weights & Biases (W&B), facilitating hyperparameter tuning, model comparison, and reproducibility.
-   **Structured Training Pipeline**: To implement a robust training pipeline complete with automatic checkpointing, early stopping, and configurable parameters.
-   **Adaptability**: To offer a clear and modular codebase that can be easily extended or adapted for other operator learning problems or variations of the DeepONet architecture.

**Important Note**: This implementation is a clean code adaptation and not a direct copy of any specific existing codebase. The underlying DeepONet architecture, its theoretical foundations, and standard licensing belong to the original papers cited below. This repository aims to make the implementation accessible and easy to use for research and development purposes.

## Original DeepONet Architecture and Known Work

The DeepONet architecture, a pioneering approach in operator learning, was introduced and extensively explored in the following foundational works by Lu, Karniadakis, and collaborators. These papers lay the theoretical groundwork and demonstrate the broad applicability of DeepONets:

**Learning nonlinear operators via DeepONet based on the universal approximation theorem of operators.**
    Lu, L., Jin, P., Pang, G., Zhang, Z., & Karniadakis, G. E. (2021).
    *Nature Machine Intelligence*, 3(3), 218-229.
    [DOI: 10.1038/s42256-021-00302-5](https://doi.org/10.1038/s42256-021-00302-5)
    *This seminal paper introduces the DeepONet concept, grounding it in the universal approximation theorem for operators.*

**Original Github code**
    [https://github.com/lululxvi/deeponet]

**Physic Informed DeepONets**
    [https://github.com/PredictiveIntelligenceLab/Physics-informed-DeepONets]

The development of DeepONets builds upon and significantly advances the field of scientific machine learning, particularly in solving PDEs and learning complex input-output relationships for various physical systems. Therefore, we strongly recommend reading these original contributions.

## Equations Solved

This repository focuses on solving the following 1D partial differential equations. The exact formulations used for data generation can be found within the `data-creator` folder, typically in files like `Burgers_data.m` and `RD_data.m`.

### 1D Burgers' Equation

The 1D Burgers' equation is a fundamental PDE describing the propagation of a shock wave. It is a simplified model of fluid flow and has both linear and non-linear terms:

$$ \frac{\partial u}{\partial t} + u \frac{\partial u}{\partial x} = \nu \frac{\partial^2 u}{\partial x^2} $$

where $u(x, t)$ is the dependent variable (e.g., velocity), $x$ is the spatial coordinate, $t$ is time, and $\nu$ is the kinematic viscosity, while the IC, $u(x,0) = f(x)$.

### 1D Reaction-Diffusion Equation

The 1D Reaction-Diffusion equation models phenomena where two processes occur: diffusion (spreading out) and reaction (generation or consumption). A common form is:

$$ \frac{\partial u}{\partial t} = D \frac{\partial^2 u}{\partial x^2} + R(u) $$

where $u(x, t)$ is the concentration of a substance, $D$ is the diffusion coefficient, and $R(u)$ is the reaction term, which can be linear or nonlinear. For example, a common reaction term is $R(u) = \alpha u (1-u)$, representing logistic growth.

Please refer to the MATLAB scripts in `data-creator/` for the specific initial and boundary conditions, and any exact source terms or reaction functions used to generate the training and testing data for each problem.

## Features

-   🚀 **PyTorch Lightning**: Clean and efficient training loops for DeepONet models.
-   📊 **Weights & Biases**: Seamless integration for comprehensive experiment tracking, logging metrics, visualizing models, and managing runs.
-   💾 **Automatic Checkpointing**: Models are saved automatically based on performance metrics (e.g., validation loss).
-   ⚙️ **Configurable**: Training and model parameters are easily configurable via command-line arguments.
-   🔍 **Training and Inference Modes**: Supports both training new models and evaluating pre-trained models.
-   🛠️ **Callbacks**: Utilizes PyTorch Lightning's built-in callbacks for early stopping and model checkpointing to enhance training stability and efficiency.

## Installation

To get started with this project, follow these steps:

```bash
# Clone the repository
git clone https://github.com/LucasUnizar/DeepONets-Lightning.git
cd DeepONets-Lightning

# Create a dedicated conda environment (highly recommended for dependency management)
conda create -n deeponet python=3.11
conda activate deeponet

# Install the required Python packages
pip install -r requirements.txt

# Log in to Weights & Biases (if you plan to use W&B for experiment tracking)
wandb login