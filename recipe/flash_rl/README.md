The release is in an early-alpha stage, please expect some rough edges.

## Environment 
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

```bash 
pip install flash-llm-rl
git clone -b flash-rl https://github.com/yaof20/verl 
cd verl & pip install --no-deps -e . 
# optional, if using wandb for logging, run the following as well
wandb login
```

## GSM8K Experiments

To prepare data:
```bash
python3 examples/data_preprocess/gsm8k.py --local_dir ~/data/gsm8k
```

To run Qwen0_5b ppo:
```bash
bash recipe/flash_rl/gsm8k_qwen0_5b_bf16.sh
```
To run Qwen0_5b ppo with Truncated Importance Sampling:
```bash
bash recipe/flash_rl/gsm8k_qwen0_5b_bf16.sh gsm8k-PPO-Qwen2.5-0.5B-bf16-TIS-2 2
```

To run Qwen0_5b ppo with int8 quantized rollout (quantization rollout can only speedup models larger than 14B, this is to showcase the effect of TIS, not rollout speedup):
```bash
# Note that the downstream performance reported in VeRL is also using int8 rollout, and a fair comparison to original RL training requires re-evaluation of ckpts (as below)
bash recipe/flash_rl/gsm8k_qwen0_5b_int8.sh
```

To run Qwen0_5b ppo with int8 quantized rollout and Truncated Importance Sampling (quantization rollout can only speedup models larger than 14B, this is to showcase the effect of TIS, not rollout speedup):
```bash
# Note that the downstream performance reported in VeRL is also using int8 rollout, and a fair comparison to original RL training requires re-evaluation of ckpts (as below)
bash recipe/flash_rl/gsm8k_qwen0_5b_ing8.sh gsm8k-PPO-Qwen2.5-0.5B-w8a8-TIS-2 2
```

To eval model checkpoints: 
```bash
# bash recipe/flash_rl/eval_gsm8k.sh "checkpoints/GSM8K-PPO/$RUN_NAME/*"
bash recipe/flash_rl/eval_gsm8k.sh "checkpoints/GSM8K-PPO/gsm8k-PPO-Qwen2.5-0.5B-w8a8-TIS-2/*"
```

## DAPO Experiments

To run Qwen32b dapo:
```bash
bash recipe/flash_rl/dapo_qwen32b_bf16.sh
```

To run Qwen32b dapo with Truncated Importance Sampling:
```bash
bash recipe/flash_rl/dapo_qwen32b_bf16.sh flash-bf16-TIS-8 8
```
