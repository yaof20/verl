odel_path=${1:-"checkpoints/GSM8K-PPO/gsm8k-PPO-Qwen2.5-0.5B-w8a8/*"}
input_name=${2:-"actor"}
output_name=${3:-"actor_huggingface"}
hf_model_path=${4:-"Qwen/Qwen2.5-0.5B-Instruct"}

log_output=${model_path%\*}

for step in ${model_path}; do
    if [ -d $step/$input_name ]; then
        echo "Converting $step/$input_name to $step/$output_name"
        python scripts/model_merger.py merge \
            --backend fsdp \
            --hf_model_path $hf_model_path \
            --local_dir $step/$input_name \
            --target_dir  $step/$output_name

        echo "{'model': '$step/$output_name'}" | tee -a ${log_output}gsm8k.log

        python3 -m verl.trainer.main_generation \
            trainer.nnodes=1 \
            trainer.n_gpus_per_node=2 \
            data.path=$HOME/data/gsm8k/test.parquet \
            data.prompt_key=prompt \
            data.batch_size=1024 \
            data.n_samples=1 \
            data.output_path=$step/$output_name/test-gsm8k.parquet \
            model.path=$step/$output_name \
            rollout.temperature=0 \
            rollout.top_p=0.95 \
            rollout.prompt_length=1024 \
            rollout.response_length=512 \
            rollout.tensor_model_parallel_size=1 \
            rollout.gpu_memory_utilization=0.9 \
            rollout.max_num_batched_tokens=65536

        python3 -m recipe.flash_rl.gsm8k_eval \
            data.path=$step/$output_name/test-gsm8k.parquet \
            data.prompt_key=prompt \
            data.response_key=responses \
            custom_reward_function.path=verl/utils/reward_score/gsm8k.py \
            custom_reward_function.name=compute_score \
            +custom_reward_function.reward_kwargs={"method":"strict"} | tee -a ${log_output}gsm8k.log
    fi
done

cat ${log_output}gsm8k.log | grep "^{'[mt]" | tee ${log_output}gsm8k.simplified
python recipe/flash_rl/process_simplified_log.py -i ${log_output}gsm8k.simplified -o ${log_output}gsm8k.csv
cat ${log_output}gsm8k.csv
