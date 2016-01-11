#!/bin/bash
#!/bin/bash
mkdir experiments/figures
mkdir experiments/results

datasets=(wiki-small wiki-vote epinions slashdot youtube)
csv_files=(input/datasets/wiki-Vote-small.txt input/datasets/Wiki-Vote_stripped.txt input/datasets/Epinions1.txt input/datasets/soc-Slashdot0902.txt input/datasets/com-youtube.ungraph.txt)
output_files=(wiki-small wiki-vote epinions slashdot youtube)
delims=(1 0 0 0 0)
undirected=(0 0 0 0 1)
scales=(0.1 0.25 0.5)
for i in 1 2 3
do
    for s in 0 2 3
	     do
		 rm -f experiments/results/report_test_approx_equal_times-${datasets[i]}-tau_scale-${scales[s]}.txt
		 nohup python test_equal_times.py -csv ${csv_files[i]} -tau_scale ${scales[s]} -undirected ${undirected[i]} -dataset ${datasets[i]} -delim ${delims[i]} -k_frac 0.1 -min_alpha 0.00001 -max_alpha 0.001 -alpha_step 0.00005 -samples 5 -cores 35 -output ${output_files[i]} -prob_method 3 >> experiments/results/report_test_approx_equal_times-${datasets[i]}-scale-${scales[s]}.txt;
		 done
done
