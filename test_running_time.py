from ic_bfs_eval import *
from generate_seeds import LoadNodesFromFile, generateSeedFiles
import subprocess
from link_server import LinkServerCP
from ic_bfs_eval import EstimateInfluence
from bfs_seq import sequential_estimation
import cPickle as cp
from time import time
import argparse
from plot_running_times import plotRatiosFromFile
import numpy as np
from scipy.stats import sem
from common_tools import *

parser = argparse.ArgumentParser()
parser.add_argument('-csv', type = str, #required,
                    default = 'input/datasets/Wiki-Vote_stripped.txt', help="Name of source csv file (if we generate a new Link Server" )
parser.add_argument('-undirected', type = int, default = 0,
                    help="-undirected : is the graph undirected (0 -- No, 1 -- Yes). Default : 0" )
parser.add_argument('-cores', type = int, default = -1,
                    help="-cores : How many processes to open when multiprocessing. Default : -1 (=# of cores)" )

parser.add_argument('-delim', type = int, default = 0,
                    help = '-delim : which delimiter (for edge-tuples) 0-tab, 1-space,2-comma. Default: 0')
parser.add_argument('-prob_method', type = int, default = 2,
                    help = '-prob_method : how to set the edge weights. 2 -- sample i.i.d. from {0.01,0.1}, 3 -- u.a.r from [0,1] . Default: 2')
parser.add_argument('-min', type = float, default = 0.1,
                     help = '-min : minimum number of seed nodes, k, per set, given as by a fraction k/n. Default: 0.01')
parser.add_argument('-max', type = float, default = 1.0,
                     help = '-max : maximum number of seed nodes, k, per set, given as by a fraction k/n. Default: 0.11')
parser.add_argument('-interval', type = float, default = 0.01,
                     help = '-interval : size of interval. Default: 0.01')
parser.add_argument('-samples', type = int, default = 10,
                    help = '-samples : number of samples of given seed set size. Default: 10')
parser.add_argument('-output', type = str, default = 'experiments/wiki/')
parser.add_argument('-dataset', type = str, default = 'input/datasets/wiki')
parser.add_argument('-title', type = str, default = '',
                    help = '-title : title of the generated plot. Default : blank')

def getCyclesDictFromFile(fname):
    f = open(fname, 'r')
    lines = [float(line.strip()) for line in f.readlines()]
    dCycles = defaultdict(list)
    
    for i in xrange(len(lines)/2):
        dCycles[lines[i*2]].append(lines[i*2+1])
    return dCycles

def getNumCycles(fname):
    f = open(fname, 'r')
    for line in f.readlines():
        words = line.split(',')
        if 'cycles\n' in words:
            f.close()
            return int(words[0])
    
def calculateRuntimes(dRuntimes):
    nSeeds_list = dRuntimes.keys()
    nSeeds_list.sort()

    avg_runtimes = np.array([1.*sum(dRuntimes[nSeeds]) /len(dRuntimes[nSeeds]) for nSeeds in nSeeds_list])

    errs_ratio = 2 * np.array([sem(dRuntimes[nSeeds]) for nSeeds in nSeeds_list])
    
    return nSeeds_list, avg_runtimes, errs_ratio

def plotRuntimes(dRuntimes, dataset_name, fig_name, n=1, log_scale = True):
    runtimes_arrays = []
    errors_arrays = []
    
    nSeeds_array, runtimes, errors = calculateRuntimes(dRuntimes)
    plot2d([1. * nSeeds / n for nSeeds in nSeeds_array], [runtimes], [errors], ['bla'], [r'$\frac{k}{n}$','Running time (in CPU cycles)'], dataset_name, fig_name)
    
if __name__ == "__main__":
    parameters = parser.parse_args()
    min_frac, max_frac, interval, nSamples, edges_csv, delim_option, output, dataset, title, pr_method = parameters.min, \
      parameters.max, parameters.interval, parameters.samples, parameters.csv, parameters.delim, parameters.output, parameters.dataset, parameters.title, \
      parameters.prob_method

    delim_dict = {0 : '\t', 1 : ' ', 2 : ','}
    delimeter = delim_dict[delim_option]
    
    bfs_method = 'seq'
    prob = 0.01
    new = True
    total_time_start = time()
    iters = 0
    if pr_method == 2:
        prob = [0.01,0.1]
    print "min_frac = %.5f, max_frac = %.5f, interval = %.5f"%(min_frac,max_frac,interval)
    print "Number of cores to use: ", parameters.cores
    print "CSV file to be used: ", edges_csv
    print "creating link-server object"
    if delimeter == "\t":
        print "delimeter is tab"
    L=LinkServerCP(dataset, edges_csv, new, pr_method, prob, delim=delimeter, undirected = parameters.undirected)
    V = LoadNodesFromFile(edges_csv, delimeter)
    n = len(V)
    print "Number of nodes: ", n
    running_times_file = output + '-running_times-large-samples-%d-k_frac-%.5f-%.5f'%(nSamples,min_frac,max_frac)

    removeFile(running_times_file)
    dRuntimes = defaultdict(list)
    
    for nSeeds in xrange(int(min_frac * n), int(max_frac * n), int(interval * n)):
        start_time_k = time()
        iters_start = iters
        seeds_fname = output + "-seeds-" + str(nSeeds) + '.cp'
        print "k = ", nSeeds
        for i in xrange(nSamples):
            generateSeedFiles(nSeeds, nSeeds+1, int(interval * n), V, 1, output + "-seeds-")

            removeFile('out.csv')
            subprocess.Popen("perf stat -x, -o out.csv python ic_bfs_eval.py -dataset %s -undirected %d -res_fname %s -seeds %s -output_mode 3 -cores %d"%\
                            (dataset, int(parameters.undirected), output + "-approx-" + str(nSeeds), seeds_fname, parameters.cores), \
                            shell = True, stdout = subprocess.PIPE).stdout.read()
            cycles_approx = getNumCycles('out.csv')
            dRuntimes[nSeeds].append(cycles_approx)
            removeFile('out.csv')

            f = open(running_times_file,'a')
            f.write(str(nSeeds) + '\n')
            f.write(str(1. * cycles_approx / nSamples) + '\n')
            f.close()
            iters += 1
        print "Average time per test for current k value (in minutes): ", (time() - start_time_k)/(60*(iters - iters_start))
    plotRuntimes(dRuntimes,title, running_times_file+'.pdf',n)
    print "Total elapsed time (in minutes) : ", (time() - total_time_start)/60
    print "Average time per run of the algorithm (in minutes): ", (time() - total_time_start)/(60 * iters)
