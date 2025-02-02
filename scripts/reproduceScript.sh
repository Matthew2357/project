#!/bin/bash

seeds=(1 2 3)
methods=(fedavg hetlora flexlora ffa fedsa fedavg)
lrs=(0.0003 0.0003 0.001 0.0001 0.001 0.0003)
trust_freqs=(25 25 25 25 25 1000)
dirichlet_alphas=(0 0.1 0.3 1 3 10)
datasets=(slim_pajama wikimulti)


for ((h=0; h<${#lrs[@]}; h++)); do
for ((i=0; i<${#dirichlet_alphas[@]; i++})); do
for ((j=0; j<${#seeds[@]}; j++)); do

lr=${lrs[h]}
method=${methods[h]}
trust_freq=${trust_freqs[h]}
dirichlet_alpha=${dirichlet_alphas[i]}
seed=${seeds[j]}

wandb_name="${method}_${dataset}_${dirichlet_alpha}"

echo "---------------- trust_freq: $trust_freq, method: $method, dirichlet_alpha: $dirichlet_alpha, seed: $seed, lr: $lr, dataset: $dataset ----------------"
python -W ignore ./src/main.py --seed $seed\
--wandb_project "Reproduce_LoRA_in_FL"\
--wandb_name $wandb_name\
--lr $lr \
--lora_rank 8 \
--lora_alpha 16 \
--trust_freq $trust_freq \
--pretraining_rounds 0 \
--iterations 500 \
--num_clients 12 \
--dataset $dataset\
--eval_freq 25\
--dirichlet_alpha $dirichlet_alpha \
--num_tokens_per_client 500000 \
--lora_rank 8\
--method "$method" \
--A_init kaiming \
--B_init zero \
--wandb \
--config_format lora \
--use_pretrained gpt2 \
--lora_mlp \
--lora_causal_self_attention \
--lora_freeze_all_non_lora 

done
done
done
