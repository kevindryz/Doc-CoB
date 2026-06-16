NPROC_PER_NODE=8 \
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
swift sft \
    --model ./weight/v1-20260218-142510/checkpoint-14326-merged \
    --train_type lora \
    --dataset './s1s2.json' \
    --torch_dtype bfloat16 \
    --num_train_epochs 1 \
    --per_device_train_batch_size 4 \
    --padding_free true \
    --attn_impl flash_attn \
    --learning_rate 1e-6 \
    --aligner_lr 1e-5 \
    --vit_lr 2e-6 \
    --freeze_llm false \
    --freeze_aligner false \
    --freeze_vit true \
    --lora_rank 16 \
    --lora_alpha 32 \
    --gradient_accumulation_steps 1 \
    --eval_strategy no \
    --save_steps 1000 \
    --save_total_limit 1 \
    --logging_steps 5 \
    --max_length 4096 \
    --warmup_ratio 0.03 \
    --dataloader_num_workers 4 \
    --output_dir ./weight2 \
    --deepspeed zero3