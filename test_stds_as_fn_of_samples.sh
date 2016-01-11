#!/bin/bash
rm -f experiments/results/log_test_concentration.txt
k_list=(1 10 100)
csv_files=(input/datasets/Wiki-Vote_stripped.txt input/datasets/soc-Slashdot0902.txt input/datasets/Epinions1.txt input/datasets/com-youtube.ungraph.txt)
dataset=(wiki-vote slashdot epinions youtube)
undirected=(0 0 0 1)
prob_methods=(2 3 1)
for m in 0 1
	 do
	     for i in 0 1 2
	     do
		 for k in 0
		 do
		     echo python test_concentration.py -csv ${csv_files[i]} -dataset ${dataset[i]} -undirected ${undirected[i]} -cores 30 -output ${dataset[i]} -min_samples 10 -max_samples 1001 -samples_step 10 -k_mode 1 -k ${k_list[k]} -prob_method ${prob_methods[m]} -seed_sets 10
	nohup python test_concentration.py -csv ${csv_files[i]} -dataset ${dataset[i]} -undirected ${undirected[i]} -cores 30 -output ${dataset[i]} -min_samples 10 -max_samples 1001 -samples_step 10 -k_mode 1 -k ${k_list[k]} -prob_method ${prob_methods[m]} -seed_sets 10 >> experiments/results/log_test_concentration.txt;
    done
	     done
done

