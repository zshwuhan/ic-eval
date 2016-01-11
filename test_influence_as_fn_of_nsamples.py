from ic_bfs_eval import *
from generate_seeds import LoadNodesFromFile, generateSeedFiles
import subprocess
from link_server import LinkServerCP
from ic_bfs_eval import EstimateInfluence
from bfs_seq import sequential_estimation
import cPickle as cp
from time import time
import argparse
import numpy as np
from plot_running_times import plotRatiosFromFile
from scipy.stats import sem
from math import log
import gc
from math import log
from common_tools import *
import random

parser = argparse.ArgumentParser()
parser.add_argument('-csv', type = str, #required,
                    default = 'input/datasets/Wiki-Vote_stripped.txt')
parser.add_argument('-delim', type = int, default = 0,
                    help = '-delim : which delimiter (for edge-tuples) 0-tab, 1-space,2-comma. Default: 0')
#parser.add_argument('-seed_sets', type = int, default = 5)
#                    help = '-samples : number of samples of given seed set size. Default: 10')
parser.add_argument('-cores', type = int, default = 30,
                    help = '-cores : number of cores of to use. Default: 30')
parser.add_argument('-undirected', type = int, default = 0,
                    help = '-undirected : is the graph undirected')
parser.add_argument('-output', type = str, default = 'wikivote')
parser.add_argument('-dataset', type = str, default = 'input/datasets/wikivote')
parser.add_argument('-min_samples', type = int, default = 10,
                    help = '-min_samples : minimum number of samples of the IC process')
parser.add_argument('-max_samples', type = int, default = 1000)
parser.add_argument('-samples_step', type = int, default = 50)
parser.add_argument('-k_mode', type = int, default = 1,
                    help = '-k_mode : manner of setting k. 0 -- fraction of n, 1 - given value')
parser.add_argument('-k', type = float, default = 1)
parser.add_argument('-prob_method', type = int, default = 2)

def getNumCycles(fname):
    f = open(fname, 'r')
    for line in f.readlines():
        words = line.split(',')
        if 'cycles\n' in words:
            f.close()
            return int(words[0])

if __name__ == "__main__":
    gc.enable()
    parameters = parser.parse_args()
    min_samples, max_samples, samples_step, k_mode, k, edges_csv, delim_option, output, dataset, prob_method, cores= \
      parameters.min_samples, parameters.max_samples, parameters.samples_step, parameters.k_mode, parameters.k,\
      parameters.csv, parameters.delim, parameters.output, parameters.dataset, parameters.prob_method, parameters.cores

    print "Dataset: ", dataset
    delim_dict = {0 : '\t', 1 : ' ', 2 : ','}
    delimeter = delim_dict[delim_option]
    results_dir = 'experiments/results/influence_values/'
    bUndirected = True if parameters.undirected == 1 else False
    bfs_method = 'seq'
    start_time = time()
    print "creating link-server object"
    if delimeter == "\t":
        print "delimeter is tab"
    L=LinkServerCP(dataset, edges_csv, create_new=True, prob_method=parameters.prob_method, prob=[0.1,0.01], delim=delimeter, undirected = bUndirected)
    
    n = L.getNumNodes()
    if k_mode == 0:
        nSeeds = int(n * k)
    else:
        nSeeds = int(k)

    perf_csv_fname = "perf_out" + str(sample(range(1000),1)[0]) + ".csv"
    print "Number of nodes: ", n
    print "nSeeds = ", nSeeds

    results_file = results_dir + output + '-influence_values_samples_min-%d-samples_max-%d-k-%.3f-prob_method-%d'%(min_samples,max_samples, k, prob_method)
    removeFile(results_file)

    seeds_fname = "%s-seeds-%d.cp"%(dataset,nSeeds)
    
    generateSeedFiles(nSeeds, nSeeds+1, 1, range(n), 1, dataset + "-seeds-")
    removeFile(results_file)
    values = []
    for samples in xrange(min_samples, max_samples+1, samples_step):
        print "number of samples = ", samples        
        output_fname = results_dir + 'nReached%d.txt'%random.randint(1,1000)
        subprocess.Popen("python seq_estimation.py -dataset %s -cores %d -seeds_file %s -results_file %s -output_mode 3 -nSamples %d -get_n_reached 1 -reached_nodes_file %s"%(dataset, parameters.cores, seeds_fname, output + "-seq-" + str(nSeeds), samples_step, output_fname), shell=True,stdout=subprocess.PIPE).stdout.read()
        removeFile(output + "-seq-" + str(nSeeds))
        f_values = open(output_fname,'r')
        values += [int(v) for v in f_values.readline().strip().split()]
        f_values.close()
        removeFile(output_fname)
        print "spread values for %d samples: %s"%(samples,str(values))
        f = open(results_file, 'a')
        f.write('%d\t%.5f\t%s\n'%(samples, np.std(values), "\t".join(str(val) for val in values)))
        f.close()
    print "Total runtime: ", (time() - start_time)/60.0
    removeFile(seeds_fname)
