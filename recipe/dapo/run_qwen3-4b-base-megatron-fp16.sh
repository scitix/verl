#!/usr/bin/env bash
set -xeuo pipefail


rollout_name="vllm" # sglang or vllm
dtype="float16" # ["bfloat16", "float16"]

project_name='FP16'
experiment_name='DAPO-Qwen3-4B-Base-megatron-fp16'

adv_estimator=grpo

use_kl_in_reward=False
kl_coef=0.0
use_kl_loss=False
kl_loss_coef=0.0

clip_ratio_low=0.2
clip_ratio_high=0.28

max_prompt_length=$((1024 * 2))
max_response_length=$((1024 * 12))
enable_overlong_buffer=True
overlong_buffer_len=$((1024 * 8))
overlong_penalty_factor=1.0

loss_agg_mode="token-mean"

train_prompt_bsz=128
n_resp_per_prompt=16
train_prompt_mini_bsz=64

# data
gsm8k_train_path=/volume/data/tldu/ai4s-job-system/data/gsm8k/train.parquet
gsm8k_test_path=/volume/data/tldu/ai4s-job-system/data/gsm8k/test.parquet
math_train_path=/volume/data/tldu/ai4s-job-system/data/math/train.parquet
curr_train_path=/volume/data/tldu/ai4s-job-system/data/curriculum/deepscaler_skew_difficult_train.parquet
math_test_path=/volume/data/tldu/ai4s-job-system/data/math/test.parquet
dapo_train_path=/volume/data/tldu/ai4s-job-system/data/dapo_17k/dapo_math_train.parquet
dapo_test_path=/volume/data/tldu/ai4s-job-system/data/dapo_17k/dapo_math_train_head5000.parquet
aime2024_test_path=/volume/data/tldu/ai4s-job-system/data/AIME_2024/test.parquet
aime2025_test_path=/volume/data/tldu/ai4s-job-system/data/AIME_2025/test.parquet
TRAIN_FILE="['$dapo_train_path']"
TEST_FILE="['$aime2024_test_path', '$aime2025_test_path']"

# Ray
RAY_ADDRESS=${RAY_ADDRESS:-"http://localhost:8265"}
WORKING_DIR=${WORKING_DIR:-"${PWD}"}
RUNTIME_ENV=${RUNTIME_ENV:-"recipe/dapo/runtime_env.yaml"}
NNODES=${NNODES:-8}
# Paths

RAY_DATA_HOME=/volume/data/tldu/ai4s-job-system/checkpoints/RL
MODEL_PATH=/volume/data/models/qwen3/Qwen3-4B-Base
CKPTS_DIR=${CKPTS_DIR:-"${RAY_DATA_HOME}/ckpts/${project_name}/${experiment_name}"}


# TIS
rollout_is=sequence
rollout_is_threshold=2.0
rollout_is_threshold_lower=null  # No lower bound
rollout_is_level=sequence  # sequence-level
rollout_is_mode=mask  # truncate mode or mask
rollout_is_veto_threshold=null  # No veto


# dapo
enable_filter_groups=True
filter_groups_metric=acc
max_num_gen_batches=10

# Algorithm
temperature=1.0
top_p=1.0
top_k=-1 # 0 for HF rollout, -1 for vLLM rollout
val_top_p=0.7

# Performance Related Parameter
use_dynamic_bsz=True
actor_ppo_max_token_len=$(((max_prompt_length + max_response_length) * 1))
infer_ppo_max_token_len=$(((max_prompt_length + max_response_length) * 1))
offload=True
gen_tp=1
train_tp=2
train_pp=1

# TODO: support dynamic_bsz for megatron

ray job submit --runtime-env="${RUNTIME_ENV}" \
    -- python3 -m recipe.dapo.main_dapo \
    --config-path=config \
    --config-name='dapo_megatron_trainer.yaml' \
    data.train_files="${TRAIN_FILE}" \
    data.val_files="${TEST_FILE}" \
    data.prompt_key=prompt \
    data.truncation='left' \
    data.seed=42 \
    actor_rollout_ref.rollout.name=${rollout_name} \
    actor_rollout_ref.rollout.dtype=${dtype} \
    actor_rollout_ref.actor.megatron.dtype=${dtype} \
    data.max_prompt_length=${max_prompt_length} \
    data.max_response_length=${max_response_length} \
    data.train_batch_size=${train_prompt_bsz} \
    actor_rollout_ref.rollout.n=${n_resp_per_prompt} \
    algorithm.filter_groups.enable=${enable_filter_groups} \
    algorithm.filter_groups.max_num_gen_batches=${max_num_gen_batches} \
    algorithm.filter_groups.metric=${filter_groups_metric} \
    algorithm.adv_estimator=${adv_estimator} \
    algorithm.use_kl_in_reward=${use_kl_in_reward} \
    algorithm.kl_ctrl.kl_coef=${kl_coef} \
    algorithm.rollout_correction.rollout_is=${rollout_is} \
    algorithm.rollout_correction.rollout_is_threshold=${rollout_is_threshold} \
    algorithm.filter_groups.max_num_gen_batches=10 \
    actor_rollout_ref.model.use_fused_kernels=False \
    actor_rollout_ref.actor.use_kl_loss=${use_kl_loss} \
    actor_rollout_ref.actor.kl_loss_coef=${kl_loss_coef} \
    actor_rollout_ref.actor.clip_ratio_low=${clip_ratio_low} \
    actor_rollout_ref.actor.clip_ratio_high=${clip_ratio_high} \
    actor_rollout_ref.actor.clip_ratio_c=10.0 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=2 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=4 \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=4 \
    actor_rollout_ref.actor.use_dynamic_bsz=${use_dynamic_bsz} \
    actor_rollout_ref.ref.log_prob_use_dynamic_bsz=${use_dynamic_bsz} \
    actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=${use_dynamic_bsz} \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=${actor_ppo_max_token_len} \
    actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=${infer_ppo_max_token_len} \
    actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=${infer_ppo_max_token_len} \
    actor_rollout_ref.model.path="${MODEL_PATH}" \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.optim.lr_warmup_steps=10 \
    actor_rollout_ref.actor.optim.weight_decay=0.1 \
    actor_rollout_ref.actor.ppo_mini_batch_size=${train_prompt_mini_bsz} \
    actor_rollout_ref.actor.megatron.param_offload=${offload} \
    actor_rollout_ref.actor.megatron.optimizer_offload=${offload} \
    actor_rollout_ref.actor.megatron.grad_offload=${offload} \
    actor_rollout_ref.actor.megatron.pipeline_model_parallel_size=${train_pp} \
    actor_rollout_ref.actor.megatron.tensor_model_parallel_size=${train_tp} \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.actor.optim.clip_grad=1.0 \
    actor_rollout_ref.actor.loss_agg_mode=${loss_agg_mode} \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.80 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=${gen_tp} \
    actor_rollout_ref.rollout.enable_chunked_prefill=True \
    actor_rollout_ref.rollout.max_num_batched_tokens=$((max_prompt_length + max_response_length)) \
    actor_rollout_ref.rollout.temperature=${temperature} \
    actor_rollout_ref.rollout.top_p=${top_p} \
    actor_rollout_ref.rollout.top_k=${top_k} \
    actor_rollout_ref.nccl_timeout=14400 \
    actor_rollout_ref.rollout.val_kwargs.temperature=${temperature} \
    actor_rollout_ref.rollout.val_kwargs.top_p=${val_top_p} \
    actor_rollout_ref.rollout.val_kwargs.top_k=${top_k} \
    actor_rollout_ref.actor.megatron.use_mbridge=True \
    actor_rollout_ref.rollout.val_kwargs.do_sample=True \
    actor_rollout_ref.rollout.val_kwargs.n=32 \
    actor_rollout_ref.rollout.calculate_log_probs=True \
    +actor_rollout_ref.actor.megatron.override_transformer_config.apply_rope_fusion=True \
    reward_model.reward_manager=dapo \
    trainer.logger=['console','swanlab'] \
    trainer.project_name="${project_name}" \
    trainer.experiment_name="${experiment_name}" \
    trainer.n_gpus_per_node=8 \
    trainer.nnodes="${NNODES}" \
    trainer.val_before_train=False \
    trainer.test_freq=10 \
    trainer.save_freq=40 \
    trainer.total_epochs=10 \
    trainer.default_local_dir="${CKPTS_DIR}" \
    trainer.rollout_data_dir=/volume/data/tldu/ai4s-job-system/checkpoints/RL/data/$experiment_name \
    trainer.resume_mode=auto \
    trainer.log_val_generations=10 \
    $@

# ray job stop raysubmit_ecd4fhpaGHGRGrEm