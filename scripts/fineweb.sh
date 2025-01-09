#!/bin/bash

seed=$1
num_clients=12
methods=(fedavg fedavg flexlora ffa fedsa)
lrs=(0.0003 0.0003 0.001 0.0001 0.001)
trust_freqs=(1000000000000 25 25 25 25)
dataset=fineweb

if [ ${#methods[@]} -ne ${#lrs[@]} ]; then
  echo "Error: methods and lrs arrays must have the same length."
  exit 1
fi

# Iterate through both arrays using an index
for i in "${!methods[@]}"; do
  method=${methods[$i]}
  lr=${lrs[$i]}
  trust_freq=${trust_freqs[i]}

  echo "Method: $method, Learning Rate: $lr"
  wandb_name="fedsa_${dataset}_${dirichlet_alpha}"
    echo "---------------- trust_freq: $trust_freq, method: $method, seed: $seed, lr: $lr, dataset: $dataset ----------------"
    python -W ignore ./src/main.py --seed $seed\
    --wandb_project "${dataset}"\
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
    --A_orth False \
    --B_orth False \
    --config_format lora \
    --use_pretrained gpt2 \
    --lora_mlp \
    --lora_causal_self_attention \
    --lora_freeze_all_non_lora \
    --wandb
done


