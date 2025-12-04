python -m verl.model_merger merge \
    --backend megatron \
    --local_dir /volume/data/tldu/ai4s-job-system/checkpoints/RL/ckpts/FP16/DAPO-Qwen3-4B-megatron-bf16-gatereward/global_step_320/actor \
    --target_dir /volume/data/tldu/ai4s-job-system/checkpoints/RL/ckpts/FP16/DAPO-Qwen3-4B-megatron-bf16-gatereward/hf_320 \