#!/bin/bash
tau_scales=(0.5 0.1)
prob_methods=(2 3 1)

for t in 0
	 do
	     for m in 1
	     do
		 rm -f experiments/results/heuristics/log_approximation-tau_scale-${tau_scales[t]}-prob_method-${prob_methods[m]}.txt
		 echo test_approximation2.py -prob_method ${prob_methods[m]} -tau_scale ${tau_scales[t]} -min_k 0.01 -max_k 0.601 -k_step 0.02 -samples 10 -cores 60
		 nohup test_approximation2.py -prob_method ${prob_methods[m]} -tau_scale ${tau_scales[t]} -min_k 0.01 -max_k 0.601 -k_step 0.02 -samples 10 -cores 60 >> experiments/results/heuristics/log_approximation-tau_scale-${tau_scales[t]}-prob_method-${prob_methods[m]}.txt;
	     done
done

