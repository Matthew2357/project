#!/bin/bash

lrs=(0.0001 0.0003 0.001 0.003 0.01)
method='fedsa'
seed=1
datasets=(wikimulti slim_pajama)
dirichlet_alpha=0
num_clients=12
trust_freq=25


for lr in ${lrs[@]}; do
    for dataset in ${datasets[@]}; do
        
        wandb_name="fedsa_dirichletalpha0_lr_$lr"
        echo "---------------- trust_freq: $trust_freq, method: $method, dirichlet_alpha: $dirichlet_alpha, seed: $seed, lr: $lr, dataset: $dataset ----------------"
        python -W ignore ./src/main.py --seed $seed\
        --wandb_project "${dataset}_${dirichlet_alpha}_${method}"\
        --wandb_name $dirichlet_alpha\
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
        --wandb \
        --config_format lora \
        --use_pretrained gpt2 \
        --lora_mlp \
        --lora_causal_self_attention \
        --lora_freeze_all_non_lora 
    done
done