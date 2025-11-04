conda create -n test_sgl python=3.10 -y && conda activate test_sgl

# install verl
git clone https://github.com/yaof20/verl.git && cd verl
git checkout -b test_sgl_patch origin/test_sgl_patch

pip install -e .[sglang]
pip install --no-cache-dir "flash-attn==2.6.3"
pip install uvloop==0.21.0
pip install ray==2.50.0


# prepare data and model
bash test_setup/prepare.sh


# install flash-rl (pull request 38)
git clone https://github.com/yaof20/Flash-RL.git && cd Flash-RL
git fetch origin pull/38/head:pr-38
git checkout pr-38
pip install -e . --no-deps


# prepare quantization
cd ../verl/test_setup
flashrl profile -m Qwen/Qwen2.5-7B-Instruct -q RedHatAI/Qwen2.5-7B-quantized.w8a8 -o ${PROFILE_PATH:-"$PWD/profile.7b.pt"} --fn int8
flashrl setup -m RedHatAI/Qwen2.5-7B-quantized.w8a8 -p $PWD/profile.7b.pt --fn int8 -o ${CONFIG_PATH:-"$PWD/flashrl_config.yaml"}


# run test
cd ..
bash recipe/flash_rl/dapo_qwen7b_sglang.sh  # this runs well without error
bash recipe/flash_rl/dapo_qwen7b_sglang_int8.sh # this runs with the following error





