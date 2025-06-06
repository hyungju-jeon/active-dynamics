# Active Dynamics

Active Learning for Latent Dynamical System Identification

## Overview

This project implements active learning algorithms for identifying latent dynamical systems. It provides tools for learning and controlling complex dynamical systems through active data collection and model-based control.

## Features

- Vector field environments with different dynamics (limit cycle, double limit cycle, multi-attractor)
- Variational Autoencoder (VAE) models for latent space learning
- Model Predictive Control (MPC) policies
- Active learning algorithms for efficient data collection
- Fisher Information metrics for uncertainty quantification

## Installation

### Prerequisites

- Python 3.12
- Conda (recommended) or pip

### Using Conda (Recommended)

1. Clone the repository:
```bash
git clone https://github.com/hyungju-jeon/active-dynamics.git
cd active-dynamics
```

2. Create and activate the conda environment:
```bash
conda env create -f environment.yml
conda activate active-dynamics
```

3. Install the package in development mode:
```bash
pip install -e .
```

### Optional: CUDA Support

If you have a CUDA-capable GPU and want to use it, install CUDA packages after creating the environment:
```bash
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
```

## Project Structure

```
actdyn/
├── core/                   # Core functionality
│   ├── base.py            # Base model classes
│   ├── dynamics.py        # Dynamics models
│   ├── encoder.py         # Encoder implementations
│   └── decoder.py         # Decoder implementations
├── environment/           # Environment implementations
│   ├── base.py           # Base environment classes
│   ├── vectorfield.py    # Vector field environments
│   └── env_wrapper.py    # Environment wrappers
├── policy/               # Policy implementations
│   ├── base.py          # Base policy classes
│   ├── icem.py          # ICEM policy
│   └── mpc.py           # MPC policy
├── utils/               # Utility functions
│   ├── rollout.py      # Rollout utilities
│   └── logger.py       # Logging utilities
└── metrics/            # Evaluation metrics
    └── costs.py        # Cost functions
```

## Usage

### Basic Example
