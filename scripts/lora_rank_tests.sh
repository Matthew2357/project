#!/bin/bash


methods=(hetlora fedavg)
datasets_name=(slim_pajama_1)
num_clients=(10 10 10 10 10 10 10)
lr=(0.0003)
lora_rank=(4 16 32 64 100 150 200)
lora_alpha=(8 32 64 128 200 300 400)

project_name=(lora_rank_tests_hetlora_2 lora_rank_tests_fedavg_1)
wandb_name=(rank4 rank16 rank32 rank64 rank100 rank150 rank200)
#project_name=(wiki_multilingual_2 wiki_multilingual_3 wiki_multilingual_4 wiki_multilingual_5 wiki_multilingual_6)
len=7
#len=5

if [ "$#" -ne 1 ]; then
    runs=1
else
    runs=$1
fi

echo "Starting runs, using $runs runs per experiment"

for((j=0; j<1; j++)); do
    for ((i = 2; i < len; i++)); do
        dataset_name=${datasets_name[0]}
        num_client=${num_clients[i]}
        proj_name=${project_name[j]}
        wandb_name=${wandb_name[i]}
        method=${methods[j]}
        lr=${lr[0]}
        lora_rank=${lora_rank[i]}
        lora_alpha=${lora_alpha[i]}
        echo "---------------- New experiment: $method, $dataset_name, $num_client, $proj_name, $lr, $lora_rank ----------------"
        ./scripts/script.sh "$proj_name" "$wandb_name" "$dataset_name" "$method" "$num_client" $runs $lr $lora_rank $lora_alpha
    done
done