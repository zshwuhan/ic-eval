#!/bin/bash
tau_scales=(0.25 0.1)
prob_methods=(2 3 1)

for t in 0
	 do
	     for m in 2 1 0
	     do
		 rm -f experiments/results/heuristics/log_compare_heuristics-tau_scale-${tau_scales[t]}-prob_method-${prob_methods[m]}.txt
		 echo compare_heuristics_small.py -prob_method ${prob_methods[m]} -tau_scale ${tau_scales[t]} -min_k 0.00002 -max_k 0.00022 -k_step 0.00002 -samples 5 -cores 60
		 nohup compare_heuristics_small.py -prob_method ${prob_methods[m]} -tau_scale ${tau_scales[t]} -min_k 0.00022 -max_k 0.00032 -k_step 0.00002 -samples 5 -cores 60 >> experiments/results/heuristics/log_compare_heuristics-tau_scale-${tau_scales[t]}-prob_method-${prob_methods[m]}.txt;
	     done
done

