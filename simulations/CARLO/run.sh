#!/bin/bash


bias=(0.0 0.05 0.1 0.25)

for bias in ${bias[@]}; do
    noise=(0.0 0.05 0.1 0.25)
    for noise in ${noise[@]}; do
        echo "noise: $noise, bias: $bias"
        python3 test_ours.py --eval --hidden_size 256 --n_eval 250 --noise $noise --bias $bias
        wait
    done
done