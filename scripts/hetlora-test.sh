#!/bin/bash

python3 ./src/main.py --wandb_project "$wandb_project" --wandb_group "$trust" --trust_freq 10 --pretraining_rounds 10 --iterations 500\
    --num_clients 3 --eval_freq 25 --dataset agnews_mixed --wandb --config_format lora --use_pretrained gpt2 --lora_mlp --lora_causal_self_attention --lora_freeze_all_non_lora --hetlora_ranks 3 3 2 --method "hetlora"