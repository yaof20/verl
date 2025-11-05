conda create -n test_sgl python=3.10 -y && conda activate test_sgl

# install verl
git clone https://github.com/yaof20/verl.git && cd verl
git checkout -b test_sgl_patch origin/test_sgl_patch

pip install -e .[sglang,test,gpu]
pip install --no-cache-dir "flash-attn==2.6.3"
pip install uvloop==0.21.0
pip install ray==2.50.0

pip install vllm==0.8.5
pip install -U "ray[default]"  # after install vllm, there will be some conflicts and we need to fix
# if the above fix doesn't work, you can try the following:
# pip install -U "opentelemetry-api==1.38.0" "opentelemetry-sdk==1.38.0" \
#   "opentelemetry-proto==1.38.0" "opentelemetry-semantic-conventions==0.59b0" \
#   "opentelemetry-exporter-prometheus==0.59b0"


# prepare data and model
bash test_setup/prepare.sh


# install flash-rl (pull request 38)
git clone https://github.com/yaof20/Flash-RL.git && cd Flash-RL
git fetch origin pull/38/head:pr-38
git checkout pr-38
pip install -e . --no-deps


# prepare quantization
cd ../verl/test_setup
hf download RedHatAI/Qwen2.5-7B-quantized.w8a8 --local-dir ./models/Qwen2.5-7B-quantized
flashrl profile -m Qwen/Qwen2.5-7B-Instruct -q RedHatAI/Qwen2.5-7B-quantized.w8a8 -o ${PROFILE_PATH:-"$PWD/profile.7b.pt"} --fn int8
flashrl setup -m ./models/Qwen2.5-7B-quantized -p $PWD/profile.7b.pt --fn int8 -o ${CONFIG_PATH:-"$PWD/flashrl_config.yaml"}



# run test
cd ..
bash recipe/flash_rl/dapo_qwen7b_sglang.sh  # this runs well without error
bash recipe/flash_rl/dapo_qwen7b_sglang_int8.sh # this runs well without error





