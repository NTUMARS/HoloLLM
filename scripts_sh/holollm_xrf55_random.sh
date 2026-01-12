#!/bin/bash

NNODES=${NNODES:-1}
NODE_RANK=${NODE_RANK:-0}
PORT=${PORT:-28405}
MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}

export CUDA_PATH=/usr/local/cuda-12.1
export PATH=/usr/local/cuda-12.1/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda-12.1/lib64:$LD_LIBRARY_PATH

LLAMA_7B_PATH="./LLM_ckpt/llama2-7B"
OUTPUT_DIR="./save_models/holollm_xrf55_random"


# holollm_fusion_encoder_clip_ce_v2_xrf55
# we remove --checkpointing
torchrun --nproc_per_node=2 --master_port=$PORT holollm_xrf55_random.py \
--epochs 5 --warmup_epochs 0.05 \
--datasets xrf55_video xrf55_depth xrf55_infra xrf55_rfid xrf55_wifi \
--max_words 1024 --batch_size 4 --accum_iter 4 \
--model_parallel_size 1 \
--data_parallel sdp \
--checkpointing --save_consolidated \
--llama_type holollm_random_xrf55_all64 \
--llama_config config/llama2/7B.json \
--llama_ckpt_dir ${LLAMA_7B_PATH} \
--tokenizer_path config/llama2/tokenizer.model \
--auto_resume \
--weight_decay 0.0 --output_dir ${OUTPUT_DIR} \
--lr 2e-5 --min_lr 0.0 --clip_grad 2 \
--save_interval 1 \
2>&1 | tee -a ${OUTPUT_DIR}/output.log