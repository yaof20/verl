# Flash RL - VeRL Flash Training

> **⚠️ Early Alpha Release**: This release is in an early-alpha stage. Please expect some rough edges and report any issues you encounter.

Flash RL is an experimental implementation of reinforcement learning training for large language models using flash (quantized) rollout to boost throughput while preserving the performance by nearly 100%.

## Table of Contents

- [Flash RL - VeRL Flash Training](#flash-rl---verl-flash-training)
  - [Table of Contents](#table-of-contents)
  - [Environment Setup](#environment-setup)
    - [Docker Environment](#docker-environment)
  - [Installation](#installation)
  - [Experiments](#experiments)
    - [GSM8K Experiments](#gsm8k-experiments)
      - [Data Preparation](#data-preparation)
      - [Training Commands](#training-commands)
    - [DAPO Experiments](#dapo-experiments)
  - [Model Evaluation](#model-evaluation)
  - [Key Features](#key-features)
  - [Troubleshooting](#troubleshooting)
  - [Contributing](#contributing)

## Environment Setup

### Docker Environment

Run the following Docker command to set up the environment:

```bash
docker run \
    --gpus all \
    --name verl \
    --shm-size=10g \
    --ipc=host \
    --ulimit memlock=-1 \
    --ulimit stack=67108864 \
    -e NCCL_P2P_LEVEL=NVL \
    -it hiyouga/verl:ngc-th2.6.0-cu126-vllm0.8.3-flashinfer0.2.2-cxx11abi0
```

## Installation

1. Install the flash-llm-rl package:
```bash
pip install flash-llm-rl
```

2. Clone the VeRL repository with the flash-rl branch:
```bash
git clone -b flash-rl https://github.com/yaof20/verl
cd verl
pip install --no-deps -e .
```

3. (Optional) Set up Weights & Biases for logging:
```bash
wandb login
```

## Experiments

### GSM8K Experiments

#### Data Preparation

First, prepare the GSM8K dataset:

```bash
python3 examples/data_preprocess/gsm8k.py --local_dir ~/data/gsm8k
```

#### Training Commands

**Basic PPO Training (Qwen2.5-0.5B, bf16):**
```bash
bash recipe/flash_rl/gsm8k_qwen0_5b_bf16.sh
```

**PPO with Truncated Importance Sampling (TIS):**
```bash
bash recipe/flash_rl/gsm8k_qwen0_5b_bf16.sh gsm8k-PPO-Qwen2.5-0.5B-bf16-TIS-2 2
```

**PPO with INT8 Quantized Rollout:**
> **Note**: Quantization rollout is primarily beneficial for models larger than 14B. For smaller models like Qwen2.5-0.5B, this is used to demonstrate TIS effects rather than rollout speedup.

```bash
bash recipe/flash_rl/gsm8k_qwen0_5b_int8.sh
```

**PPO with INT8 Quantized Rollout + Truncated Importance Sampling:**
> **Note**: The downstream performance reported in VeRL is also using INT8 rollout, and a fair comparison to original RL training requires re-evaluation of checkpoints (as below).

```bash
bash recipe/flash_rl/gsm8k_qwen0_5b_int8.sh gsm8k-PPO-Qwen2.5-0.5B-w8a8-TIS-2 2
```

### DAPO Experiments

**Basic DAPO Training (Qwen2.5-32B, bf16):**
```bash
bash recipe/flash_rl/dapo_qwen32b_bf16.sh
```

**DAPO with Truncated Importance Sampling:**
```bash
bash recipe/flash_rl/dapo_qwen32b_bf16.sh flash-bf16-TIS-8 8
```

## Model Evaluation

To evaluate trained model checkpoints:

```bash
# General format:
# bash recipe/flash_rl/eval_gsm8k.sh "checkpoints/GSM8K-PPO/$RUN_NAME/*"

# Example for a specific checkpoint:
bash recipe/flash_rl/eval_gsm8k.sh "checkpoints/GSM8K-PPO/gsm8k-PPO-Qwen2.5-0.5B-w8a8-TIS-2/*"
```

## Key Features

- **Flash Attention**: Optimized attention computation for faster training
- **Truncated Importance Sampling (TIS)**: Improved training stability and efficiency
- **INT8 Quantization**: Memory-efficient rollout for large models
- **Multi-GPU Support**: Distributed training across multiple GPUs

## Troubleshooting

If you encounter issues:

1. Ensure you have sufficient GPU memory for your model size
2. Check that all dependencies are properly installed
3. Verify your Docker environment is correctly configured
4. For INT8 quantization issues, ensure your hardware supports the required operations

## Contributing

This is an experimental implementation. Please report bugs and contribute improvements through the main VeRL repository.
