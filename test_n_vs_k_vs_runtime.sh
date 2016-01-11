#!/bin/bash
graph_method=(2 3 4 5)
graph_method_name=(BarabasiAlbert Kronecker SmallWorld ConfigurationModel)
prob_methods=(3 2)
min_n=1000
max_n=5001
n_step=1000
for i in 0 1 2 3
do
    for m in 0
	     do
	     rm -f experiments/report_test_running_time_as_function_of_size-${graph_method_name[i]}-1000-5000-samples-5
	     nohup python test_running_time_as_function_of_size.py -n_min $min_n -prob_method ${prob_methods[m]} -n_max $max_n -n_step $n_step -cores 60 -graph_method ${graph_method[i]} -min 0.01 -max 0.105 -interval 0.01 -samples 5 -output experiments/results/${graph_method_name[i]} -dataset ${graph_method_name[i]} -eps 0 >> experiments/report_test_running_time_as_function_of_size-${graph_method_name[i]}-1000-5000-samples-5
	     done
	     done
