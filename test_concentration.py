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
from scipy.stats import sem
from heuristics import runVanilla
parser = argparse.ArgumentParser()
parser.add_argument('-csv', type = str, #required,
                    default = 'input/datasets/Wiki-Vote_stripped.txt')
parser.add_argument('-delim', type = int, default = 0,
                    help = '-delim : which delimiter (for edge-tuples) 0-tab, 1-space,2-comma. Default: 0')
parser.add_argument('-seed_sets', type = int, default = 5)
#                    help = '-samples : number of samples of given seed set size. Default: 10')
parser.add_argument('-cores', type = int, default = 40,
                    help = '-cores : number of cores of to use. Default: 30')
parser.add_argument('-undirected', type = int, default = 0,
                    help = '-undirected : is the graph undirected')
parser.add_argument('-output', type = str, default = 'out')
parser.add_argument('-dataset', type = str, default = 'input/datasets/wiki')
parser.add_argument('-min_samples', type = int, default = 5,
                    help = '-min_samples : minimum number of samples of the IC process')
parser.add_argument('-max_samples', type = int, default = 1000)
parser.add_argument('-samples_step', type = int, default = 50)
parser.add_argument('-k_mode', type = int, default = 0,
                    help = '-k_mode : manner of setting k. 0 -- fraction of n, 1 - given value')
parser.add_argument('-k', type = float, default = 0.1)
parser.add_argument('-prob_method', type = int, default = 3)

def getNumCycles(fname):
    f = open(fname, 'r')
    for line in f.readlines():
        words = line.split(',')
        if 'cycles\n' in words:
            f.close()
            return int(words[0])

def getTrueValue(dataset, seeds_fname, nBFS_samples, num_cores):
    output_file = 'out' + str(random.randint(1,1000))
    print "Getting true values with nBFS_samples = %d"%nBFS_samples
    subprocess.Popen("python seq_estimation.py -dataset %s -cores %d -seeds_file %s -results_file %s -output_mode 1 -nSamples %d"%(dataset, cores, seeds_fname, output_file, nBFS_samples), shell=True,stdout=subprocess.PIPE).stdout.read()
    
    results = [float(line.strip()) for line in open(output_file,'r').readlines()]
    removeFile(output_file)
    return [line for i, line in enumerate(results) if i%2 == 0]

if __name__ == "__main__":
    global input_dir
    
    gc.enable()
    parameters = parser.parse_args()
    min_samples, max_samples, samples_step, k_mode, k, edges_csv, delim_option, output, dataset, prob_method, cores, seed_sets= \
      parameters.min_samples, parameters.max_samples, parameters.samples_step, parameters.k_mode, parameters.k,\
      parameters.csv, parameters.delim, parameters.output, parameters.dataset, parameters.prob_method, parameters.cores,\
      parameters.seed_sets

    print "Dataset: ", dataset
    delim_dict = {0 : '\t', 1 : ' ', 2 : ','}
    delimeter = delim_dict[delim_option]
    results_dir = 'experiments/results/'
    input_dir = 'input/datasets/'
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

    approx_results_fname = output + "-equal_times-approx-" + str(k) + '-' + str(sample(range(1000),1)[0])
    perf_csv_fname = "perf_out" + str(sample(range(1000),1)[0]) + ".csv"
    nBFS_samples = 1000
    print "Number of nodes: ", n
    print "nSeeds = ", nSeeds

    results_file = results_dir + output + '-influence_concentration_samples_min-%d-samples_max-%d-k-%.3f-prob_method-%d-nSeeds_sets-%d'%(min_samples,max_samples, k, prob_method, seed_sets)
    removeFile(results_file)

    seeds_fname = "%s-seeds-%d.cp"%(results_dir + output, nSeeds)
    generateSeedFiles(nSeeds, nSeeds+1, 1, range(n), seed_sets, results_dir + output + "-seeds-")
    values = []
    for samples in xrange(min_samples, max_samples+1, samples_step):
        print "number of samples = ", samples        
        output_fname = results_dir + 'nReached%d.txt'%random.randint(1,1000)
        subprocess.Popen("python seq_estimation.py -dataset %s -cores %d -seeds_file %s -results_file %s -output_mode 3 -nSamples %d -get_n_reached 1 -reached_nodes_file %s"%(dataset, parameters.cores, seeds_fname, output + "-seq-" + str(nSeeds), samples_step, output_fname), shell=True,stdout=subprocess.PIPE).stdout.read()
        removeFile(output + "-seq-" + str(nSeeds))
        f_values = open(output_fname,'r')
        f = open(results_file, 'a')
        for i, line in enumerate(f_values.readlines()):
            values += [int(v) for v in line.strip().split()]
            f.write('%d\t%.5f\n'%(samples, sem(values)/np.mean(values)))
        f.close()
        f_values.close()
        removeFile(output_fname)
    
    removeFile(seeds_fname)
    
    print "Total runtime: ", (time() - start_time)/60.0
