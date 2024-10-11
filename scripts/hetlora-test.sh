#!/bin/bash

python3 ./src/main.py --wandb_project random --wandb_group random --trust_freq 25 --pretraining_rounds 100 --iterations 500\
    --num_clients 4 --eval_freq 25 --dataset agnews_mixed --wandb --config_format lora --use_pretrained gpt2 --lora_mlp --lora_causal_self_attention --lora_freeze_all_non_lora --method "hetlora"