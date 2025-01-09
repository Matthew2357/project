#!/bin/bash

dataset=$1

lrs=(0.0003)
#lrs=(0.2 0.5 1)



for ((h=0; h<${#lrs[@]}; h++)); do
    lr=${lrs[h]}
    echo "---------------- lr: $lr, dataset: $dataset ----------------"
    python -W ignore ./src/main.py --seed 1 \
    --wandb_project "ffa_lr_$dataset"\
    --wandb_name $lr\
    --lr $lr \
    --lora_rank 8 \
    --lora_alpha 16 \
    --trust_freq 100 \
    --pretraining_rounds 0 \
    --iterations 500 \
    --num_clients 12 \
    --dataset $dataset \
    --eval_freq 25\
    --dirichlet_alpha 0 \
    --num_tokens_per_client 500000 \
    --method ffa \
    --wandb \
    --config_format lora \
    --use_pretrained gpt2 \
    --lora_mlp \
    --lora_causal_self_attention \
    --lora_freeze_all_non_lora 
done
