#!/bin/bash

lr=$1
method=$2
seed=3

dirichlet_alphas=(0 0.05 0.1 0.3 0.7 1 3 10)
reg_coeff=0.001
dataset=$3
num_clients=12
trust_freq=25


for ((h=0; h<${#dirichlet_alphas[@]}; h++)); do
        
            
    #reg_coeff=${reg_coeffs[0]}
    #dataset=${datasets[0]}
    dirichlet_alpha=${dirichlet_alphas[h]}
    wandb_name=$method
    #lr=${lrs[i]}
    #method=${methods[i]}
    echo "---------------- trust_freq: $trust_freq, method: $method, dirichlet_alpha: $dirichlet_alpha, seed: $seed, lr: $lr, dataset: $dataset, reg_coeff: $reg_coeff ----------------"
    python -W ignore ./src/main.py --seed $seed\
    --wandb_project "${dataset}_${dirichlet_alpha}_${method}"\
    --wandb_name $dirichlet_alpha\
    --lr $lr \
    --lora_rank 8 \
    --lora_alpha 16 \
    --reg_coeff $reg_coeff \
    --trust_freq $trust_freq \
    --pretraining_rounds 0 \
    --iterations 250 \
    --num_clients $num_clients \
    --dataset $dataset\
    --eval_freq 25\
    --dirichlet_alpha $dirichlet_alpha \
    --num_tokens_per_client 500000 \
    --hetlora_ranks 20 15 10 8 8 8 8 8 6 6 6 4 \
    --method "$method" \
    --wandb \
    --config_format lora \
    --use_pretrained gpt2 \
    --lora_mlp \
    --lora_causal_self_attention \
    --lora_freeze_all_non_lora 
done
