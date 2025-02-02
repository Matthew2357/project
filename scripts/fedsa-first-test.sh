#!/bin/bash

wandb_project=$6
lr=$5
method=$4
seed=$1
A_init=$2
B_init=$3
dataset=$7
dirichlet_alphas=(0 0.1 0.3 1 3 10)
num_clients=12
trust_freq=25


#for dataset in ${datasets[@]}; do
    for dirichlet_alpha in ${dirichlet_alphas[@]}; do
        
        wandb_name="${dirichlet_alpha}_${dataset}_${method}"
        echo "---------------- trust_freq: $trust_freq, method: $method, dirichlet_alpha: $dirichlet_alpha, seed: $seed, lr: $lr, dataset: $dataset ----------------"
        python -W ignore ./src/main.py --seed $seed\
        --wandb_project $wandb_project\
        --wandb_name $wandb_name\
        --lr $lr \
        --lora_rank 8 \
        --lora_alpha 16 \
        --trust_freq $trust_freq \
        --pretraining_rounds 0 \
        --iterations 500 \
        --num_clients $num_clients \
        --dataset $dataset\
        --eval_freq 25\
        --dirichlet_alpha $dirichlet_alpha \
        --num_tokens_per_client 500000 \
        --lora_rank 8\
        --method "$method" \
        --A_init $A_init \
        --B_init $B_init \
        --wandb \
        --config_format lora \
        --use_pretrained gpt2 \
        --lora_mlp \
        --lora_causal_self_attention \
        --lora_freeze_all_non_lora 
    done
#done