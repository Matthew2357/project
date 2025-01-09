#!/bin/bash


methods=(fedavg)
datasets_name=(slim_pajama_1)

num_clients=(4 4 4 4 4)
lr=(0.0001 0.0003)
project_name=(fedavg_lr_tests)

len=2



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
        echo "---------------- New experiment: $method, $dataset_name, $num_client, $proj_name, $lr----------------"
        ./scripts/script.sh "$proj_name" "$dataset_name" "$method" "$num_client" $runs $lr
        
    done
done