#!/bin/bash

prob_methods=(3 2)
probs=(0,.001 0.001,0.0001)
datasets=(BA100 BA500 Kronecker100 gnp08-1000)
csv_files=(input/datasets/BA100.csv input/datasets/BA500.csv input/datasets/Kronecker100.csv input/datasets/gnp08-1000.csv)
nSamples=10
init_samples=(5 10 20 30)
iter_samples=(5 10 20 30)
min_k=0.001
max_k=0.005
k_step=0.001
for d in 3
do
    for m in 0
    do
	for i in 0 1 2 3
	do
	    for j in 0 1 2 3
		     do
	rm -f experiments/results/approx-heuristic1/log_${datasets[d]}-prob_method-${prob_methods[m]}-samples-$nSamples
	nohup python test_approx_heuristic.py -init_samples ${init_samples[i]} -iter_samples ${iter_samples[j]} -dataset ${datasets[d]} -csv ${csv_files[d]} -cores 45 -min_k $min_k -max_k $max_k -k_step $k_step -prob_method ${prob_methods[m]} -prob ${probs[m]} -samples $nSamples >> experiments/results/approx-heuristic1/log_${datasets[d]}-prob_method-${prob_methods[m]}-samples-$nSamples;
	    done
	    
	done
    done
done

