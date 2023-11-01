#!/bin/bash
# Run all simulations for experiment 1
# noise=(0.025)

# for noise in ${noise[@]}; do
#     bias=(-0.25 -0.2 -0.1 -0.05 -0.025 0.025 0.05 0.1 0.2 0.25)
#     for bias in ${bias[@]}; do
#         echo "noise: $noise, bias: $bias"
#         python3 test_ours.py --eval --boltzmann --lr 1.0 --n_eval 100 --n_features 2 --env_dim 3 --hidden_size 256 --noise $noise --bias $bias
#         wait
#     done
# done


bias=(0.0 0.025 0.05 -0.025 -0.05)

for bias in ${bias[@]}; do
    noise=(0.0 0.025 0.05 0.1)
    for noise in ${noise[@]}; do
        echo "noise: $noise, bias: $bias"
        python3 test_ours.py --eval --boltzmann --lr 1.0 --n_eval 100 --n_features 2 --env_dim 3 --hidden_size 256 --noise $noise --bias $bias
        wait
    done
done