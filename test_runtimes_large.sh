#!/bin/bash
mkdir experiments/figures
mkdir experiments/results

datasets=(wiki-small wiki-vote epinions slashdot youtube)
csv_files=(input/datasets/wiki-Vote-small.txt input/datasets/Wiki-Vote_stripped.txt input/datasets/Epinions1.txt input/datasets/soc-Slashdot0902.txt input/datasets/com-youtube.ungraph.txt)
output_files=(wiki-small wiki-vote epinions slashdot youtube)
delims=(1 0 0 0 0)
undirected=(0 0 0 0 1)

for i in 1
do
    rm -f experiments/results/report_test_running_times-large-${datasets[i]}.txt
	     nohup python test_running_time_large.py -csv ${csv_files[i]} -undirected ${undirected[i]} -dataset ${datasets[i]} -delim ${delims[i]} -min 0.01 -max 0.6 -interval 0.02 -samples 5 -cores -1 -output ${output_files[i]} >> experiments/results/report_test_running_times-large-${datasets[i]}.txt
done
