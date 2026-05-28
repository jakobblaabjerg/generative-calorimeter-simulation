# Generative Models for Calorimeter Simulation

This repository contains code for dataset creation, training, hyperparameter sweeps, sampling, and evaluation of generative deep learning models for calorimeter simulation.

The project supports multiple model architectures and configurable training pipelines through YAML configuration files.

---

# Repository Structure

```text
.
├── configs/                 # YAML configuration files
├── src/                     # Core source code
├── create_dataset.py        # Dataset creation script
├── train.py                 # Model training script
├── sweep.py                 # Hyperparameter sweep script
├── sample.py                # Sampling/generation script
├── evaluate.py              # Evaluation script
└── README.md
```

---

# Installation

Clone the repository:

```bash
git clone https://github.com/jakobblaabjerg/generative-calorimeter-simulation
cd generative-calorimeter-simulation
```

Create a virtual environment and install dependencies:

```bash
python -m venv venv
source venv/bin/activate
```

Install required packages:

```bash
pip install -r requirements.txt
```

---

# Dataset

The raw dataset can be downloaded from Zenodo:

- https://zenodo.org/records/17199427

After downloading the dataset, use the preprocessing script to build the training dataset.

---

# Creating the Dataset

The dataset creation pipeline processes the raw files and stores the processed dataset in the specified output directory.

## Usage

```bash
python create_dataset.py \
    --input_dir <path-to-raw-data> \
    --output_dir <path-to-save-processed-data>
```

## Optional Arguments

```bash
--config_file configs/filters.yaml
```

## Example

```bash
python create_dataset.py \
    --input_dir data/raw \
    --output_dir data/processed
```

---

# Training

Models are trained using configuration files stored in `configs/`.

You can either:

- specify a predefined model name
- provide a custom configuration file

## Usage

### Train using predefined model configuration

```bash
python train.py --model <model_name>
```

### Train using custom config file

```bash
python train.py --cfg_file <path-to-config>
```

---

## Optional Arguments

```bash
--encoder <encoder_name>
--patience <int>
--epochs <int>
--batch_size <int>
--lr <float>
--log_dir <path>
--data_dir <path>
--num_files <int>
--debug
```

---

## Example

```bash
python train.py \
    --model cfm \
    --encoder deepsets \
    --epochs 100 \
    --batch_size 64 \
    --lr 1e-4 \
    --data_dir data/processed
```

---

# Hyperparameter Sweeps

Hyperparameter optimization can be performed using the sweep script.

## Usage

```bash
python sweep.py \
    --model <model_name> \
    --trials <num_trials>
```

---

## Optional Arguments

```bash
--encoder <encoder_name>
--space <search_space_name>
--samples <int>
--debug
```

---

## Example

```bash
python sweep.py \
    --model cfm \
    --encoder deepsets \
    --space encoder \
    --trials 50 \
    --samples 2
```

---

# Sampling

Generate samples from a trained model.

## Usage

```bash
python sample.py \
    --model_dir <trained-model-dir> \
    --data_dir <processed-data-dir> \
    --save_dir <output-dir>
```

---

## Optional Arguments

```bash
--config_filters configs/filters.yaml
--config_sampling configs/sampling.yaml

--phi <float>
--theta <float>
--e_inc <float>
--num_samples <int>
```

---

## Example

```bash
python sample.py \
    --model_dir logs/cfm_run \
    --data_dir data/processed \
    --save_dir samples \
    --num_samples 100
```

---

# Evaluation

Evaluate trained models and compute metrics.

## Usage

```bash
python evaluate.py \
    --model_dir <trained-model-dir> \
    --data_dir <processed-data-dir>
```

---

## Optional Arguments

```bash
--cfg_sampling configs/sampling.yaml
--num_mc_samples <int>
--save_dir <path>
```

---

## Example

```bash
python evaluate.py \
    --model_dir logs/cfm_run \
    --data_dir data/processed \
    --num_mc_samples 10
```

Evaluation metrics are stored in:

```text
metrics.json
```

---

# Configuration

Configuration files are located in:

```text
configs/
```

These YAML files define:

- model architecture
- optimizer settings
- training hyperparameters
- dataset settings
- sampling configuration
- search spaces for sweeps

---

# Logging

Training logs and checkpoints are stored in:

```text
logs/<experiment_name>/
```

---

# Features

- Dataset preprocessing pipeline
- Multiple generative model architectures
- Configurable training system
- Hyperparameter sweeps
- Sampling and generation
- Evaluation pipeline
- YAML-based configuration system

---

# Citation

If you use this repository in your research, please cite the associated work.

