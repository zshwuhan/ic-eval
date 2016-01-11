import time
import argparse
from collections import defaultdict,namedtuple
import os
from link_server import *
from math import log
#import cProfile, pstats,StringIO
from bfs_seq import sequential_estimation
from link_server import LinkServerCP
import cPickle as cp
from common_tools import removeFile
parser = argparse.ArgumentParser()

parser.add_argument('-dataset', type = str, default = 'input/datasets/wiki',
                    help = '-dataset : dataset to use. Default: input/datasets/wiki')
parser.add_argument('-seeds_file', type = str, default = 'experiments/wiki/seeds/wiki-seeds-71.cp',
                    help = '-edge_prob_type : method of generating edge probabilities. 0 - uniform. 1 - 1/in-deg(v), 2 - uid from [0.1,0.01]')
parser.add_argument('-results_file', type = str, default = "experiments/wiki/results-seq-71.txt",
                    help = '-results_file : file name of the results output file')
parser.add_argument('-get_n_reached', type = int, default = 0)
parser.add_argument('-reached_nodes_file', type = str, default = "experiments/wiki/wiki-reached_nodes.txt")
parser.add_argument('-output_mode', type = int, default = 2,
                    help = '-output_mode: 0 - print to screen only, 1 - output results to file, 2 - output to both, 3 - no output')
parser.add_argument('-cores', type = int, default = -1,
                    help = '-cores: number of concurrent processes to start every time. Default : -1 (# of cores)')
parser.add_argument('-nSamples', type = int, default = 0,
                    help = '-nSamples: Number of BFS samples to take (0 = take theoretic n*log(n) samples. Default : 0')

parser.add_argument('-min_samples', type = int, default = -1)
parser.add_argument('-min_relative_standard_error', type = float, default = -1)

def print_out(text):
    if output_mode in [0,2]:
        print text
    if output_mode in [1,2]:
        f_output = open(output_file,'a')
        f_output.write(text+'\n')
        f_output.close()

if __name__ == "__main__":
    parameters = parser.parse_args()
    dataset, seeds_file, results_file, output_mode, nSamples = parameters.dataset, parameters.seeds_file, parameters.results_file,\
      parameters.output_mode, parameters.nSamples

    L = LinkServerCP(dataset)
    f = open(seeds_file, 'r')

    assert (nSamples <= 0 or parameters.min_relative_standard_error <= 0)
    seeds_list = cp.load(f)
    f.close()
    removeFile(parameters.reached_nodes_file)
    if output_mode in [1,2]:
        f = open(results_file, 'w')
    for i, seed_set in enumerate(seeds_list):
        if output_mode in [0,2]:
            print "Sample ", i
        if parameters.get_n_reached == 1:
            try:
                avg, total_samples,l_n_reached = sequential_estimation(L, seed_set, max_samples_cap=nSamples, \
                                                                       nCores=parameters.cores,bReturnValues = True,\
                                                                       min_samples = parameters.min_samples, \
                                                                       min_relative_standard_error = parameters.min_relative_standard_error)
            except:
                r = open('test.txt','w')
                r.write('cannot run bfs\n')
                r.close()
                raise Exception()
        else:
            avg, total_samples = sequential_estimation(L, seed_set, max_samples_cap=nSamples, \
                                                       nCores=parameters.cores,bReturnValues = False, \
                                                       min_samples = parameters.min_samples,\
                                                       min_relative_standard_error = parameters.min_relative_standard_error)

        if output_mode in [1,2]:

            f.write(str(avg) + "\n")
            f.write(str(total_samples) + '\n')
        if output_mode in [0,2]:
            print "Average = %.3f, total # of samples needed = %d"%(avg, total_samples)
        if parameters.get_n_reached == 1:
            g = open(parameters.reached_nodes_file, 'a')
            g.write('\t'.join(str(v) for v in l_n_reached) + '\n')
            g.close()

    if output_mode in [1,2]:    
        f.close()
        
