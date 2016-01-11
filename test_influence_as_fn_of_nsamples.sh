#!/bin/bash
rm -f experiments/results/log_test_inf_vs_samples.txt
k_list=(1 10 100)
csv_files=(input/datasets/Wiki-Vote_stripped.txt input/datasets/soc-Slashdot0902.txt input/datasets/Epinions1.txt)
dataset=(wiki-vote slashdot epinions)
undirected=(0 0 0)
prob_methods=(3 2 1)
for m in 1
	 do
	     for i in 0
	     do
		 for k in 0
		 do
	nohup python test_influence_as_fn_of_nsamples.py -csv ${csv_files[i]} -dataset ${dataset[i]} -undirected ${undirected[i]} -cores 50 -output ${dataset[i]} -min_samples 10 -max_samples 5000 -samples_step 50 -k_mode 1 -k ${k_list[k]} -prob_method ${prob_methods[m]} >> experiments/results/log_test_inf_vs_samples.txt;
    done
	     done
done

