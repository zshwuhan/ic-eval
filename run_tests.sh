#!/bin/bash
mkdir experiments/figures
mkdir experiments/results
##### Get running time ratios
datasets=(input/datasets/wiki-vote input/datasets/slashdot input/datasets/epinions)
csv_files=(input/datasets/Wiki-Vote_stripped.txt input/datasets/soc-Slashdot0902.txt input/datasets/Epinions1.txt)
output_files=('wiki-vote' 'slashdot' 'epinions')
#rm -f experiments/results/report_test_running_times-small.txt
#for i in 0
#do
#    nohup python test_running_time_small.py -csv ${csv_files[i]} -dataset ${datasets[i]} -delim 0 -min 0.01 -max 0.6 -interval 0.05 -samples 5 -output ${output_files[i]} >>  experiments/results/report_test_running_times-small.txt;
#done
#### Raunning times for large graphs
#rm report_test_running_times-wikiVote.txt
#nohup python test_running_time_small.py -csv input/datasets/Wiki-Vote_stripped.txt -dataset input/datasets/wikiVote -delim 0 -min 0.01 -max 0.1 -interval 0.01 -samples 10 -output 'experiments/wikiVote' -title 'Wiki-Vote' -eps 0,0.0001,0.001,0.01 >> report_test_running_times-wikiVote.txt &
################
#### Approx ratio tests -- real datasets
#rm -f experiments/report_approx_ratios-epinions.txt
#nohup python test_k_vs_approx.py -csv input/datasets/Epinions1.txt -prob_method 3 -dataset input/datasets/epinions -samples 10 -delim 0 -min 0.01 -max 0.6 -interval 0.01 -output epinions -title 'Epinions'>> report_approx_ratios-epinions.txt &

rm -f experiments/report_approx_ratios-Wiki-Vote.txt
nohup python test_k_vs_approx.py -csv input/datasets/Wiki-Vote_stripped.txt -prob_method 3 -dataset input/datasets/wiki-vote -samples 5 -delim 0 -min 0.01 -max 0.6 -interval 0.02 -output wiki-vote -title 'wiki-Vote'>> experiments/report_approx_ratios-wiki-vote.txt &

#rm -f experiments/report_approx_ratios-slashdot.txt
#nohup python test_k_vs_approx.py -csv input/datasets/soc-Slashdot0902.txt -prob_method 3 -dataset input/datasets/slashdot -samples 10 -delim 0 -min 0.01 -max 0.6 -interval 0.02 -output slashdot -title 'Slashdot'>> experiments/report_approx_ratios-slashdot.txt &

#rm -f experiments/report_approx_ratios-wiki_small.txt
#echo python test_k_vs_approx.py -csv input/datasets/wiki-Vote-small.txt -prob_method 3 -dataset input/datasets/wiki-small -samples 10 -delim 1 -min 0.01 -max 0.6 -interval 0.02 -output wiki-small -title ''
#nohup python test_k_vs_approx.py -csv input/datasets/wiki-Vote-small.txt -prob_method 3 -dataset input/datasets/wiki-small -samples 10 -delim 1 -min 0.01 -max 0.6 -interval 0.02 -output wiki-small -title ''>> experiments/report_approx_ratios-wiki-small.txt;


#rm -f experiments/report_approx_ratios-enron.txt
#nohup python test_k_vs_approx.py -csv input/datasets/email-Enron.txt -undirected 1 -prob_method 3 -dataset input/datasets/enron -samples 10 -delim 0 -min 0.01 -max 0.6 -interval 0.02 -output enron -title ''>> experiments/report_approx_ratios-enron.txt;

#########


