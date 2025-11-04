#!/bin/bash

set -e

# ========== 기본 설정 ==========
NUM_GPUS=8
NNODES=1
SCRIPT="inference_gpt_for_wbl.py"

# ========== 실행 ==========
python3 -m torch.distributed.run \
  --standalone \
  --nnodes=${NNODES} \
  --nproc_per_node=${NUM_GPUS} \
  ${SCRIPT} \
  \
  --disable-bias-linear \
  --seq-length 4096 \
  --max-position-embeddings 32768 \
  --num-layers 24 \
  --position-embedding-type none \
  --hidden-size 2048 \
  --ffn-hidden-size 9216 \
  --num-attention-heads 16 \
  --attention-dropout 0.0 \
  --hidden-dropout 0.0 \
  --swiglu \
  --untie-embeddings-and-output-weights \
  \
  --num-experts 128 \
  --moe-layer-freq "([0]*1+[1]*23)" \
  --moe-ffn-hidden-size 1024 \
  --moe-shared-expert-intermediate-size 1024 \
  --moe-router-topk 8 \
  --moe-router-score-function sigmoid \
  --moe-router-dtype fp32 \
  --moe-token-dispatcher-type alltoall \
  --moe-router-topk-scaling-factor 1.0 \
  --rotary-scaling-factor 1.0 \
  --sliding-window-size 512 \
  --sliding-window-interleave-k 6 \
  \
  --multi-latent-attention \
  --q-lora-rank 1024 \
  --kv-lora-rank 512 \
  --qk-head-dim 128 \
  --qk-pos-emb-head-dim 64 \
  --v-head-dim 128 \
  --normalization RMSNorm \
  --rope-type rope \
  --apply-layernorm-1p \
  --rotary-base 10000 \
  --rotary-base-global 1000000 \
  --qk-layernorm \
  \
  --tokenizer-type HuggingFaceTokenizer \
  --tokenizer-model bridge/WBL-20B-A2B-HF-Dummy \
  \
  --bf16 \
  \
  --no-load-optim \
  --no-load-rng \
  --attention-backend flash \
  \
  --expert-model-parallel-size 1 \
  --pipeline-model-parallel-size 8 \
  \
  --inference-max-requests 1 \
  --num-tokens-to-generate 100 \
  --ckpt-format torch_dist \
  \
  --load /workspace/data/FlashBlade/klab_data/checkpoints/pretrain_stage1_20b \
  --prompts \
  "Charlotte Perriand (24 October 1903 - 27 October 1999) was"
