#!/bin/bash

methods=(ffa flexlora hetlora)
datasets_name=(agnews_mixed agnews_specific three_multi_mixed three_multi_specific github_wiki_mixed github_wiki_specific)
num_clients=(4 4 4 4 4 4)
project_name=(cl-AG-M cl-AG-S cl-TW-M cl-TW-S cl-GW-M cl-GW-S)

len=6

if [ "$#" -ne 1 ]; then
    runs=1
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