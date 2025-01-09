#!/bin/bash

seed=1
wandb_projects=(num_tokens_300000_het num_tokens_400000_het num_tokens_500000_het num_tokens_600000_het num_tokens_1m_het)
wandb_names=(fedavg no_collab)
lrs=(0.0003 0.0003)
methods=(fedavg fedavg)
trust_freqs=(25 1000000000000)
dirichlet_alpha=0.0
num_tokens_per_clients=(300000 400000 500000 600000 1000000)

for ((i=4; i<5; i++)); do
    for ((j=1; j<2; j++)); do
        wandb_project="${wandb_projects[i]}"
        wandb_name="${wandb_names[j]}"
        lr="${lrs[j]}"
        method="${methods[j]}"
        trust_freq="${trust_freqs[j]}"
        num_tokens_per_client="${num_tokens_per_clients[i]}"
        echo "---------------- trust_freq: $trust_freq, num_tokens: $num_tokens_per_client ----------------"
        python -W ignore ./src/main.py --seed $seed --wandb_project "$wandb_project" --wandb_name "$wandb_name" --lr $lr --lora_rank 8 --lora_alpha 16 --trust_freq $trust_freq --pretraining_rounds 100 --iterations 3000 --num_clients 10 --dataset slim_pajama\
        --eval_freq 25 --dirichlet_alpha $dirichlet_alpha --num_tokens_per_client $num_tokens_per_client --method "$method" --wandb --config_format lora --use_pretrained gpt2 --lora_mlp --lora_causal_self_attention --lora_freeze_all_non_lora 
    done
done
