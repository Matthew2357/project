#!/bin/bash

python3 ./src/main.py --wandb_project random --wandb_group random --trust_freq 10 --pretraining_rounds 10 --iterations 500\
    --num_clients 3 --eval_freq 25 --dataset matthew-dataset --wandb --config_format lora --use_pretrained gpt2 --lora_mlp --lora_causal_self_attention --lora_freeze_all_non_lora --method "flexlora"