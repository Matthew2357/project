#!/bin/bash

methods=(ffa flexlora hetlora ffa_inversed fedavg)
datasets_name=(wiki_multilingual_1 wiki_multilingual_2 wiki_multilingual_3 wiki_multilingual_4 wiki_multilingual_5 wiki_multilingual_6)
num_clients=(4 4 4 4 4 4)
project_name=(wiki_multilingual_1 wiki_multilingual_2 wiki_multilingual_3 wiki_multilingual_4 wiki_multilingual_5 wiki_multilingual_6)

len=6

if [ "$#" -ne 1 ]; then
    runs=2
else
    runs=$1
fi

echo "Starting runs, using $runs runs per experiment"

for method in "${methods[@]}"; do
    for ((i = 0; i < len; i++)); do
        dataset_name=${datasets_name[i]}
        num_client=${num_clients[i]}
        proj_name=${project_name[i]}
        echo "---------------- New experiment: $method, $dataset_name, $num_client, $proj_name ----------------"
        ./scripts/script.sh "$proj_name" "$dataset_name" "$method" "$num_client" $runs
    done
done