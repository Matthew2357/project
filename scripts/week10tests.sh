#!/bin/bash

seeds=(1 2 3)

lrs=(0.0001)
methods=(ffa)
trust_freqs=(25)
dirichlet_alpha=$1
num_clients=$2
dataset=$3
wandb_names=(flexlora fedavg no_collab)



for ((j=0; j<3; j++)); do
    for ((i=0; i<3; i++)); do    
        seed=${seeds[i]}
        lr=${lrs[j]}
        method="${methods[j]}"
        trust_freq=${trust_freqs[j]}
        wandb_name="${wandb_names[j]}"
        
        echo "---------------- trust_freq: $trust_freq, method: $method, dirichlet_alpha: $dirichlet_alpha, seed: $seed ----------------"
        python -W ignore ./src/main.py --seed $seed\
        --wandb_project "week9tests_${dirichlet_alpha}_seed_${seed}_2"\
        --wandb_name "$wandb_name"\
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
        --method "$method" --wandb \
        --config_format lora \
        --use_pretrained gpt2 \
        --lora_mlp \
        --lora_causal_self_attention \
        --lora_freeze_all_non_lora 
    done
done






