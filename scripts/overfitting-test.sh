#!/bin/bash

python3 ./src/main.py --wandb_project random --wandb_group random --wandb_name fedavg_5_10_6_100 --trust_freq 25 --pretraining_rounds 100 --iterations 500 --lora_rank 100 --lora_alpha 200 --lr 0.0003\
    --num_clients 10 --eval_freq 25 --dataset slim_pajama_1 --wandb --config_format lora --use_pretrained gpt2 --lora_mlp --lora_causal_self_attention --lora_freeze_all_non_lora --method "fedavg"