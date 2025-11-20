set -x

export CUDA_DEVICE_MAX_CONNECTIONS=1 # For megatron communication/computation overlapping

# data
HOME=/volume/data/tldu/ai4s-job-system
WORKING_DIR=${WORKING_DIR:-"${PWD}"}
gsm8k_train_path=$HOME/data/gsm8k/train.parquet
gsm8k_test_path=$HOME/data/gsm8k/test.parquet
math_train_path=$HOME/data/math/train.parquet
curr_train_path=$HOME/data/curriculum/deepscaler_skew_difficult_train.parquet
math_test_path=$HOME/data/math/test.parquet
dapo_train_path=$HOME/data/dapo_17k/dapo_math_train.parquet
aime2024_test_path=$HOME/data/AIME_2024/test.parquet
aime2025_test_path=$HOME/data/AIME_2025/test.parquet
dapo_test_path=/volume/data/tldu/ai4s-job-system/data/dapo_17k/dapo_math_train_head5000.parquet
TRAIN_FILE="['$dapo_train_path']"
TEST_FILE="['$aime2024_test_path', '$aime2025_test_path']"




# paths
project_name='fsdp-megatron-RL'
experiment_name='DAPO-Qwen3-4B-megatron-bf16'
RAY_DATA_HOME=${RAY_DATA_HOME:-"${HOME}/checkpoints/RL"}
CKPTS_DIR=${CKPTS_DIR:-"${RAY_DATA_HOME}/ckpts/${project_name}/${experiment_name}"}
HF_DIR=${CKPTS_DIR:-"${RAY_DATA_HOME}/ckpts/${project_name}/hf/${experiment_name}"}
model_path=/volume/data/models/qwen3/Qwen3-4B-Base
dist_checkpointing_path=/volume/data/tldu/ai4s-job-system/checkpoints/RL/mcore_ckpt/Qwen3-4B-Base
# other
clip_ratio_low=0.2
clip_ratio_high=0.28
offload=True
max_prompt_length=$((1024 * 2))
max_response_length=$((1024 * 12))
enable_overlong_buffer=True
overlong_buffer_len=$((1024 * 8))
overlong_penalty_factor=1.0
RUNTIME_ENV=${RUNTIME_ENV:-"recipe/dapo/runtime_env.yaml"}
temperature=1.0
top_p=1.0
top_k=-1 # 0 for HF rollout, -1 for vLLM rollout
val_top_p=0.7


ray job submit --address="http://127.0.0.1:8265" \
    --runtime-env="${RUNTIME_ENV}" \
    -- python3 -m recipe.dapo.main_dapo --config-path=config \
    --config-name='dapo_megatron_trainer.yaml' \
    algorithm.adv_estimator=grpo \
    data.train_files="${TRAIN_FILE}" \
    data.val_files="${TEST_FILE}" \
    data.train_batch_size=32 \
    data.seed=42 \
    data.max_prompt_length=${max_prompt_length} \
    data.max_response_length=${max_response_length} \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    actor_rollout_ref.actor.optim.optimizer=adam \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.optim.lr_warmup_steps=20 \
    actor_rollout_ref.actor.optim.weight_decay=0.1 \
    actor_rollout_ref.actor.strategy=megatron \
    actor_rollout_ref.actor.clip_ratio_low=${clip_ratio_low} \
    actor_rollout_ref.actor.clip_ratio_high=${clip_ratio_high} \
    actor_rollout_ref.model.path=$model_path \
    actor_rollout_ref.actor.ppo_mini_batch_size=16 \
    actor_rollout_ref.actor.use_dynamic_bsz=True \
    actor_rollout_ref.actor.megatron.use_mbridge=False \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=20000 \
    actor_rollout_ref.actor.megatron.pipeline_model_parallel_size=2 \
    actor_rollout_ref.rollout.max_num_batched_tokens=$((max_prompt_length + max_response_length)) \
    actor_rollout_ref.actor.megatron.tensor_model_parallel_size=4 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.actor.megatron.param_offload=${offload} \
    actor_rollout_ref.actor.megatron.optimizer_offload=${offload} \
    actor_rollout_ref.actor.megatron.grad_offload=${offload} \
    actor_rollout_ref.actor.megatron.use_dist_checkpointing=False \
    actor_rollout_ref.actor.megatron.dist_checkpointing_path=${dist_checkpointing_path} \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=8 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=4 \
    actor_rollout_ref.ref.megatron.param_offload=${offload} \
    actor_rollout_ref.ref.megatron.use_dist_checkpointing=False \
    actor_rollout_ref.ref.megatron.dist_checkpointing_path=${dist_checkpointing_path} \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.dtype=bfloat16 \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.8 \
    actor_rollout_ref.rollout.n=8 \
    actor_rollout_ref.nccl_timeout=14400 \
    actor_rollout_ref.rollout.temperature=${temperature} \
    actor_rollout_ref.rollout.top_p=${top_p} \
    actor_rollout_ref.rollout.top_k="${top_k}" \
    actor_rollout_ref.rollout.val_kwargs.temperature=${temperature} \
    actor_rollout_ref.rollout.val_kwargs.top_p=${val_top_p} \
    actor_rollout_ref.rollout.val_kwargs.top_k=${top_k} \
    actor_rollout_ref.rollout.val_kwargs.n=32 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=8 \
    actor_rollout_ref.ref.megatron.pipeline_model_parallel_size=2 \
    actor_rollout_ref.ref.megatron.tensor_model_parallel_size=4 \
    algorithm.use_kl_in_reward=False \
    algorithm.filter_groups.enable=True \
    algorithm.filter_groups.metric=acc \
    algorithm.filter_groups.max_num_gen_batches=10 \
    reward_model.reward_manager=dapo \
    +reward_model.reward_kwargs.overlong_buffer_cfg.enable=${enable_overlong_buffer} \
    +reward_model.reward_kwargs.overlong_buffer_cfg.len=${overlong_buffer_len} \
    +reward_model.reward_kwargs.overlong_buffer_cfg.penalty_factor=${overlong_penalty_factor} \
    +reward_model.reward_kwargs.overlong_buffer_cfg.log=False \
    +reward_model.reward_kwargs.format_reward_cfg.enable=True \
    +reward_model.reward_kwargs.max_resp_len=${max_response_length} \
    trainer.critic_warmup=0 \
    trainer.logger='["console","tensorboard","swanlab"]' \
    trainer.project_name=${project_name} \
    trainer.experiment_name=${experiment_name} \
    trainer.default_hdfs_dir=${HF_DIR} \
    trainer.default_local_dir="${CKPTS_DIR}" \
    trainer.n_gpus_per_node=8 \
    trainer.nnodes=8 \
    trainer.max_actor_ckpt_to_keep=3 \
    trainer.save_freq=20 \
    trainer.log_val_generations=5 \
    trainer.test_freq=1 \
    trainer.val_before_train=False \
    trainer.total_epochs=10 \
    trainer.rollout_data_dir=/volume/data/tldu/ai4s-job-system/checkpoints/RL/data/$experiment_name \
    trainer.resume_mode=resume_path \
    trainer.resume_from_path=/volume/data/tldu/ai4s-job-system/checkpoints/RL/ckpts/fsdp-megatron-RL/DAPO-Qwen3-4B-megatron-bf16/global_step_20 \
    $@
# ray job logs raysubmit_NeyVC6U9RyANuEi8