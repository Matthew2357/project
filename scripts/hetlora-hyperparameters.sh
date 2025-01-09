#!/bin/bash


methods=(hetlora)
datasets_name=(slim_pajama_1)

num_clients=(4 4 4 4 4)
lr=(0.0001 0.001 0.01 0.1)
reg=(0.001 0.01 0.1)
project_name=(flexlora_lr_tests)

len=4
len2=3


if [ "$#" -ne 1 ]; then
    runs=1
else
    runs=$1
fi

echo "Starting runs, using $runs runs per experiment"

for method in "${methods[@]}"; do
    for ((j = 0; j < len2; j++)); do
        for ((i = 0; i < len; i++)) do
            dataset_name=${datasets_name[0]}
            num_client=${num_clients[i]}
            proj_name=${project_name[0]}
            lr=${lr[i]}
            reg_coeff=${reg[j]}
            echo "---------------- New experiment: $method, $dataset_name, $num_client, $proj_name, $lr, $reg_coeff ----------------"
            ./scripts/script.sh "$proj_name" "$dataset_name" "$method" "$num_client" $runs $lr $reg_coeff
        done
    done
done