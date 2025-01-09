#!/bin/bash
: <<'EOF'
#!/bin/bash

seed=1

lrs=(0.0003 0.001 0.0001 0.0003)
methods=(fedavg flexlora ffa no_collab)
trust_freqs=(25 25 25 100000000000000)
dirichlet_alpha=$1



for ((j=0; j<4; j++)); do
    
    
    lr="${lrs[j]}"
    method="${methods[j]}"
    trust_freq="${trust_freqs[j]}"
    
    echo "---------------- trust_freq: $trust_freq, method: $method, dirichlet_alpha: $dirichlet_alpha ----------------"
    python -W ignore ./src/main.py --seed $seed --wandb_project "week9tests_${dirichlet_alpha}" --wandb_name "${method}" --lr $lr --lora_rank 8 --lora_alpha 16 --trust_freq $trust_freq --pretraining_rounds 100 --iterations 500 --num_clients 10 --dataset slim_pajama\
    --eval_freq 25 --dirichlet_alpha $dirichlet_alpha --num_tokens_per_client 500000 --method "$method" --wandb --config_format lora --use_pretrained gpt2 --lora_mlp --lora_causal_self_attention --lora_freeze_all_non_lora 
done
EOF



seed=1

lr=0.0003
method=$2
trust_freq=$3
dirichlet_alpha=$1




    
    
    
    
    
    
echo "---------------- trust_freq: $trust_freq, method: $method, dirichlet_alpha: $dirichlet_alpha ----------------"
python -W ignore ../src/main.py --seed $seed --wandb_project "week9tests_${dirichlet_alpha}" --wandb_name "${method}" --lr $lr --lora_rank 8 --lora_alpha 16 --trust_freq $trust_freq --pretraining_rounds 100 --iterations 500 --num_clients 6 --dataset slim_pajama\
    --eval_freq 25 --dirichlet_alpha $dirichlet_alpha --num_tokens_per_client 500000 --method "$method" --wandb --config_format lora --use_pretrained gpt2 --lora_mlp --lora_causal_self_attention --lora_freeze_all_non_lora 




