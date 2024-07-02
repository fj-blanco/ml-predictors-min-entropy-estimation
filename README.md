# Machine Learning Predictors for Min-Entropy Estimation

This is the code repository for our work:

> Javier Blanco-Romero, Vicente Lorenzo, Florina Almenares Mendoza, Daniel Díaz-Sánchez. "Machine Learning Predictors for Min-Entropy Estimation." arXiv:2406.19983 [cs.LG], 28 Jun 2024.

[Read the full paper on arXiv](https://arxiv.org/abs/2406.19983)

It contains autoregressive data generation, training and evaluation of two machine learning models (**RCNN** and **GPT-2**), a pipeline for running the experiments, and data analysis scripts.

## Table of Contents

- [Installation](#installation)
  - [With conda](#with-conda)
- [RCNN model](#rcnn-model)
- [Models Usage](#models-usage)
- [Pipeline](#pipeline)
  - [Usage](#usage)
  - [Variable ```target_bits```](#variable-target_bits)
  - [Variable $|\alpha|$](#variable-alpha)
- [Autoregressive prediction](#autoregressive-prediction) 
- [Notation for some special gbAR(p) models](#notation-for-some-special-gbarp-models)
  - [Constant gbAR(p) with mixed signs](#constant-gbarp-with-mixed-signs)
- [Entropy calculations](#entropy-calculations)
  - [Running theoretical entropies calculations (deprecated)](#running-theoretical-entropies-calculations-deprecated)
  - [Monte Carlo simulations](#monte-carlo-simulations)

## Installation

We have tested this in a system with the following specifications:

- Debian GNU/Linux 11 (bullseye)
- RTX 3090Ti GPU

### With conda

Install Miniconda

```bash
wget https://repo.anaconda.com/miniconda/Miniconda3-py38_23.5.2-0-Linux-x86_64.sh
bash Miniconda3-py38_23.5.2-0-Linux-x86_64.sh 
```

Create virtual environment and install dependencies

```bash
conda create -n .rng_ml_pipeline-venv python=3.9
conda activate .rng_ml_pipeline-venv
conda config --add channels conda-forge
conda config --set solver libmamba
conda install --file requirements.txt  -c pytorch -c nvidia
```

## RCNN model

This is the original version of the program from here: [Machine Learning Cryptanalysis of a Quantum Random Number Generator](https://github.com/NeuroSyd/Machine-Learning-Cryptanalysis-of-a-Quantum-Random-Number-Generator)

See also: [Machine Learning Cryptanalysis of a Quantum Random Number Generator | GitHub](https://github.com/NeuroSyd/Machine-Learning-Cryptanalysis-of-a-Quantum-Random-Number-Generator)

This is a modified version of `rng_rcnn` with optimizations that allow training and evaluating the model on batches of data instead of the entire data at once. It also allows training and evaluating the model on bit sequences instead of bytes.

## Models Usage

Both models share most of the input parameters:

```bash
python -m <model_module> \
    --filename ../../data/1mb/cesga/cesga_random_bytes_1000000_batch_run_1.bin \
    --generator cesga  \
    --seqlen 100 \
    --step 3 \
    --num_bytes 10000000000 \
    --target_bits 1 2 3 4 5 6 7 8 \
    --train_ratio 0.8 \
    --test_ratio 0.2 \
    --learning_rate 0.005 \
    --batch_size 512 \
    --epochs 10 \
    --evaluation_checkpoints 1 2 3 4 5 6 7 8 9 10
```

where model_module is one of `models.rcnn.rng_rcnn` or `models.gpt2.rng_gpt2`.

- `--filename`: Name of the file containing the data you want to use for training the model. This is the only required parameter.
- `--generator`: Type of the generator used to generate the data. This parameter is used for naming the output files.
- `--seqlen`: The length of the sequence considered by the model to make a prediction. Default value is 100.
- `--step`: The number of steps to skip between successive sequences. For example, if step is 3, the model will use sequences 1-100, 4-103, 7-106, etc. Default value is 3.
- `--num_bytes`: Total length of the data used for training the model, in bytes. Default value is 10,000,000,000 bytes (approximately 9.31 GB).
- `--target_bits``: Sets the target bit lengths for the model. This parameter defines the size of the output sequences produced by the model. Provide as space-separated values (e.g., '1 2 3'). Defaults to 1 if not provided. Each value must be between 1 and seqlen - 1.
- `--train_ratio`: The portion of the data used for training. Default value is 0.8, which means 80% of the data is used for training, and the rest is used for testing.
- `--test_ratio`: The portion of the data used for testing. This must be less than or equal to (1 - train_ratio). Default value is 0.2.
- `--learning_rate`: Specifies the rate at which the model learns during training. A higher learning rate can lead to faster learning, but may also cause instability. Default value is 0.005 for RCNN and 0.0005 for GPT-2.
- `--batch_size`: Number of samples per gradient update. Default value is 512.
- `--epochs`: Number of epochs to train the model. An epoch is an iteration over the entire data provided. Default value is 10.
- `--is_autoregressive`: (just GPT-2) Activates the autoregressive mode when set. In this mode, the model generates each bit in the sequence based on the previously generated bits. By default, this mode is disabled (False).
- `--evaluate_all_bits`: (just GPT-2) When enabled, the evaluation of the model will consider all bits in the sequence. This is useful for a detailed analysis of the model's performance across the entire bit sequence. By default, this is disabled (False).
- `--evaluation_checkpoints`: Lists the data size goals for evaluations during training. The model will be evaluated at these specific points, allowing for a periodic assessment of its performance. Provide as space-separated values (e.g., '1 2 3 4 5').

## Pipeline

### Usage

```bash
nohup python rng_ml_pipeline.py --num_bytes 10000000 --target_bits 1 2 3 4 5 6 7 8 --corr_intensities 0.5 --model_name gpt2 --distance_scale_p 10 --batch_size 128 --autocorrelation_function constant --learning_rate 0.0005 --hardware RTX3060Ti &
```

- `--model_name`: This sets the model name. Default is 'gpt2'. Possible values include 'rcnn', 'gpt2', etc.
- `--hardware`: This specifies the hardware being used. Must be provided explicitly (e.g., 'RTX3060Ti', 'g5.xlarge').

- `--corr_intensities`: This sets the correlation intensities. Provide as space-separated values (e.g., '0.1 0.2 0.3'). Defaults to a linspace-generated array if not provided.
- `--num_bytes`: This sets the number of bytes. Default is 100000. The value must be at least 100,000.
- `--target_bits`: This sets the target bit lengths for the model. Provide as space-separated values (e.g., '1 2 3'). Defaults to [1] if not provided. Each value must be between 1 and seqlen - 1.
- `--seqlen`: This sets the maximum sequence length. Default is 100.
- `--step`: This sets the step size. Default is the same as seqlen if not provided.
- `--train_ratio`: This sets the ratio of data used for training. The value must be greater than 0 and less than 1. Default is 0.8.
- `--learning_rate`: This sets the learning rate. Default is 0.0001.
- `--batch_size`: This sets the batch size. Must be specified as it has no default.
- `--epochs`: This sets the number of epochs. Default is 1.
- `--distance_scale_p`: This sets the distance scale 'p' parameter. Default is 1.
- `--autocorrelation_function`: This sets the autocorrelation function. Default is 'point-to-point'. Possible values: 'exponential', 'gaussian', 'point-to-point', 'constant'.
- `--is_autoregressive`: Activates the autoregressive mode when set. In this mode, the model generates each bit in the sequence based on the previously generated bits. By default, this mode is disabled (False).
- `--evaluate_all_bits`: When enabled, the evaluation of the model will consider all bits in the sequence. This is useful for a detailed analysis of the model's performance across the entire bit sequence. By default, this is disabled (False).

### Variable ```target_bits```

```bash
nohup python rng_ml_pipeline.py --num_bytes 36000000 --model_name rcnn --distance_scale_p 10 --corr_intensities 0.5 --autocorrelation_function constant --learning_rate 0.005 --hardware RTX3060Ti --target_bits 12 16 14 15 16 --seqlen 100 &
```

will run the pipeline for the hardcoded target bits values {12, 13, 14, 15, 16} with the **RCNN** model.

Analogously,

```bash
nohup python rng_ml_pipeline.py --num_bytes 10000000 --model_name gpt2 --distance_scale_p 10 --corr_intensities 0.5 --autocorrelation_function constant --learning_rate 0.005 --hardware RTX3060Ti --target_bits 2 4 --seqlen 100 &
```

will run the pipeline with **GPT-2** for the hardcoded target bits values {2, 4}.

### Variable $|\alpha|$

For example

```bash
nohup python rng_ml_pipeline.py --num_bytes 10000000 --model_name rcnn --distance_scale_p 2 --autocorrelation_function constant --learning_rate 0.005 --hardware RTX3060Ti --target_bits 8 --seqlen 100 &
```

will run the pipeline for the hardcoded alpha values 

```bash
corr_intensities = np.logspace(-2, -0.001, num=10)
```

## Autoregressive prediction

EVALUATE_ALL_BITS = False
Set `--is_autoregressive` to `True` in `rng_gpt2.py`. Use `--evaluate_all_bits` as desired.

## Notation for some special gbAR(p) models

### Constant gbAR(p) with mixed signs

We will denote these processes as `constant_{signs}`. For example, the process with $\alpha$ vector

$\alpha = normalization \times [1, -1]$ will be denoted as `constant_+_-`.

Example pipeline call:

```bash
python rng_ml_pipeline.py --num_bytes 10000000 --target_bits 1 2 3 4 5 6 7 8 --corr_intensities 0.5 --model_name gpt2 --distance_scale_p 4 --batch_size 128 --autocorrelation_function constant --learning_rate 0.0005 --hardware RTX3060Ti --signs +1 -1 +1 -1
```

## Entropy calculations

### Monte Carlo simulations

The parameters are harcoded in the script (go anc check them before running it)

```bash
python ./entropy_calculation/montecarlo/empirical_entropies_calculation.py
```