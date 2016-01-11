#!/bin/bash

datasets=(wikivote BA1000_dataset BA100)
csv_files=(input/datasets/Wiki-Vote_stripped.txt input/datasets/BA1000)
prob_methods=(3 2)
nSamples=5
min_k=1
max_k=1
k_step=1
k_mode=1
init_samples=30
iter_samples=10
probs=(0,.1 0.001,0.1)
for d in 0
do
    for m in 0 1
    do
	rm -f experiments/results/approx-heuristic1/log_${datasets[d]}-prob_method-${prob_methods[m]}-samples-$nSamples
	nohup python test_approx_heuristic.py -k_mode 1 -init_samples $init_samples -iter_samples $iter_samples -dataset ${datasets[d]} -csv ${csv_files[d]} -cores 60 -min_k $min_k -max_k $max_k -k_step $k_step -prob_method ${prob_methods[m]} -prob ${probs[m]} -samples $nSamples >> experiments/results/approx-heuristic1/log-${datasets[d]}-prob_method-${prob_methods[m]}-samples-$nSamples;
    done
done

