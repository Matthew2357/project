#!/bin/bash

#methods=(ffa flexlora hetlora ffa_inversed fedavg)
methods=(ffa)
datasets_name=(slim_pajama_1)
#datasets_name=(wiki_multilingual_2 wiki_multilingual_3 wiki_multilingual_4 wiki_multilingual_5 wiki_multilingual_6)
num_clients=(4 4 4 4 4 4)
lr=(0.003 0.001 0.0003 0.0001)
#num_clients=(4 4 4 4 4 4)
project_name=(ffa_lr_tests)
#project_name=(wiki_multilingual_2 wiki_multilingual_3 wiki_multilingual_4 wiki_multilingual_5 wiki_multilingual_6)
len=4
#len=5

if [ "$#" -ne 1 ]; then
    runs=1
else
    runs=$1
fi

echo "Starting runs, using $runs runs per experiment"

for method in "${methods[@]}"; do
    for ((i = 0; i < len; i++)); do
        dataset_name=${datasets_name[0]}
        num_client=${num_clients[i]}
        proj_name=${project_name[0]}
        lr=${lr[i]}
        echo "---------------- New experiment: $method, $dataset_name, $num_client, $proj_name, $lr ----------------"
        ./scripts/script.sh "$proj_name" "$dataset_name" "$method" "$num_client" $runs $lr
    done
done