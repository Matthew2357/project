#!/bin/bash

'''seeds=(3 2 1)
num_clients=12
methods=(ffa_inversed ffa fedsa_inv ffa_inversed fedsa)
lrs=(0.001 0.0001 0.001 0.001 0.001)
trust_freqs=(25 25 25 25 25)
A_inits=(zero kaiming zero zero zero)
B_inits=(kaiming zero kaiming orth kaiming)
dataset=fineweb'''
seeds=(1)
num_clients=12
methods=fedsa
lrs=(0.001)
trust_freqs=(25)
A_inits=(zero)
B_inits=(kaiming)
dataset=fineweb

if [ ${#methods[@]} -ne ${#lrs[@]} ]; then
  echo "Error: methods and lrs arrays must have the same length."
  exit 1
fi

# Iterate through both arrays using an index
for seed in ${seeds[@]}; do
for i in "${!methods[@]}"; do
  method=${methods[$i]}
  lr=${lrs[$i]}
  trust_freq=${trust_freqs[$i]}
  A_init=${A_inits[$i]}
  B_init=${B_inits[$i]}

  echo "Method: $method, Learning Rate: $lr"
  wandb_name="fedsa_inv_${dataset}_${dirichlet_alpha}"
    echo "---------------- trust_freq: $trust_freq, method: $method, seed: $seed, lr: $lr, dataset: $dataset ----------------"
    python -W ignore ./src/main.py --seed $seed\
    --wandb_project fineweb \
    --wandb_name $method\
    --lr $lr \
    --lora_rank 8 \
    --lora_alpha 16 \
    --trust_freq $trust_freq \
    --pretraining_rounds 0 \
    --iterations 500 \
    --num_clients $num_clients \
    --dataset $dataset\
    --eval_freq 25\
    --num_tokens_per_client 500000 \
    --lora_rank 8\
    --method "$method" \
    --A_init $A_init \
    --B_init $B_init \
    --config_format lora \
    --use_pretrained gpt2 \
    --lora_mlp \
    --lora_causal_self_attention \
    --lora_freeze_all_non_lora \
    --wandb
done
done


