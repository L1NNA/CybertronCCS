#!/bin/bash

echo "Which GPU:"
read index

echo "Which dataset:"
read data

declare -a filters=("3" "4" "5")
declare -a cells=("128" "256" "512")
model="RLModelKL"

for filter in "${filters[@]}"
do
    for cell in "${cells[@]}"
    do
        echo "$model" "$data" "filters ${filter} cells ${cell}"
        python -m cybertron --model $model --args "{ \"filters\": $filter, \"output_length\": $cell}" --device $index --name "${model}_${filter}_${cell}_6000" train --data $data --epoch 20
    done
done

