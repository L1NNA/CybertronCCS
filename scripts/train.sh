#!/bin/bash

echo "How many GPUs:"
read total
echo "Which GPU:"
read index

declare -a models=("GRUC")
declare -a dataset=("Exp_all_m1")

i=0
for model in "${models[@]}"
do
    for data in "${dataset[@]}"
    do
        j=$(($i%$total))
        if [ $j == $index ]
        then
            echo "$model" "$data"
            python -m cybertron --model $model --device $index train --data $data --epoch 20
        fi
        i=$(($i+1))
    done
done

