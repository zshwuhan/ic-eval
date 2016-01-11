from ic_bfs_eval import *
from generate_seeds import LoadNodesFromFile, generateSeedFiles
import subprocess
from link_server import LinkServerCP
from ic_bfs_eval import EstimateInfluence
from bfs_seq import sequential_estimation
import cPickle as cp
from time import time
import argparse

from common_tools import *
from collections import defaultdict
import numpy as np
from scipy.stats import sem
def loadDataFromFiles(approx_fname,seq_fname):
    '''
    Loads results from the two files that contain the results:
    in approx_fname:
    line 1: approximated influence
    line 2: value of k
    line 3: maximal distance from any initial seed

    in seq_fname:
    line 1: estimated influence.
    line 2: number of samples needed.
    '''
    
    approx_estimates = []
    kkt_estimates = []
    f = open(approx_fname, 'r')
    approx = f.readline().strip()
    while not approx == '':
        k = int(f.readline().strip())
        approx_estimates.append(float(approx))
        f.readline()
        approx = f.readline().strip()
    f.close()

    f=open(seq_fname,'r')
    influence_estimate = f.readline().strip()
    while not influence_estimate == '':
        kkt_estimates.append(float(influence_estimate))
        f.readline().strip()
        influence_estimate = f.readline().strip()
        
    f.close()
    return k,approx_estimates, kkt_estimates

def calculateMeanErrorAndSEM(approx_fname,seq_fname):

    k, approx_values, influence_estimates = loadDataFromFiles(approx_fname,seq_fname)
    assert len(approx_values) == len(influence_estimates)
    estimation_errors = []
    for i, approx in enumerate(approx_values):
        ratio = 1.0 * approx / influence_estimates[i]
        if ratio >= 1:
            estimation_errors.append(ratio)
        else:
            estimation_errors.append(1./ratio)
    
    mean = (np.mean(estimation_errors))
    std_err = sem(estimation_errors)
    return k, mean, std_err

    
parser = argparse.ArgumentParser()
parser.add_argument('-csv', type = str, #required,
                    default = 'input/datasets/Epinions1.csv', help="Name of source csv file (if we generate a new Link Server" )
parser.add_argument('-prob_method', type = int, #required,
                    default = 2, help='prob_method : method of setting edge-probabilities.\n 0 - fixed,\n1 - 1/in-deg,\n2 - uniform sample from the set [0.1,0.01].\n3 - u.a.r. Default: 3')
parser.add_argument('-undirected', type = int, #required,
                    default = 0, help='-undirected : is the graph an undirected graph. Default: 0 (no)')
parser.add_argument('-delim', type = int, default = 0,
                    help = '-delim : which delimiter (for edge-tuples) 0-tab, 1-space,2-comma. Default: 0')
parser.add_argument('-min', type = float, default = 0.01,
                    help = '-min : minimum fraction for k. algorithm. Default: 0.01')
parser.add_argument('-max', type = float, default = 0.6,
                    help = '-max : maximum fraction for k. algorithm. Default: 0.6')
parser.add_argument('-interval', type = float, default = 0.03,
                    help = '-interval : size of interval. Default: 0.03')
parser.add_argument('-samples', type = int, default = 5,
                    help = '-samples : number of samples of given seed set size. Default: 5')
parser.add_argument('-output', type = str, default = 'epinions2')
parser.add_argument('-title', type = str, default = '')
parser.add_argument('-dataset', type = str, default = 'input/datasets/epinions2')
if __name__ == "__main__":
    tau_scale_values = [0.5, 0.25, 0.20, 0.1]
    parameters = parser.parse_args()
    min_frac, max_frac, interval, nSamples, edges_csv, delim_option, output, dataset = parameters.min, \
      parameters.max, parameters.interval, parameters.samples, parameters.csv, parameters.delim, parameters.output, parameters.dataset
    
    print "Input file: ", edges_csv
    print "Output file prefix: ", output

    delimiter = delim_dict[delim_option]
    start_time = time()
    bfs_method = 'seq'
    print "creating link-server object"
    if parameters.prob_method == 0:
        edge_prob = 0.2
    else:
        edge_prob = [0.1,0.01]
    L = LinkServerCP(dataset, edges_csv, create_new=True,
                     prob_method=parameters.prob_method,
                     prob=edge_prob, delim=delimiter,
                     undirected=parameters.undirected)
    print "n = ", L.getNumNodes()
    V = LoadNodesFromFile(edges_csv, delimiter)
    n = len(V)

    k_min = int(n * min_frac)
    k_max = int(n * max_frac) + 1
    k_step = int(n * interval)
    print "max_k = ", max_frac
    print "Minimum k value: %d, maximum k value: %d" % (k_min, k_max)
    removeFile(dataset)
    generateSeedFiles(k_min, k_max, k_step, V, nSamples, 'experiments/' + output + "-seeds-")
    results_fname = "experiments/results/" + output + '-approximations-nSamples-%d-k_frac-%.3f-%.3f'%(nSamples, min_frac, max_frac)
    removeFile(results_fname)
    mean_errors, std_errors = [], []
    
    for k in xrange(k_min, k_max,k_step):
        approx_fname = 'experiments/results/' + output + \
          "-approx_errors-k_min-%d-k_max-%d-k-%d-samples-%d"%(k_min,k_max,k,nSamples)
        seq_fname = 'experiments/results/' + output + "-seq-approx-errors-k-%d-samples-%d"%(k,nSamples)
        seeds_fname = 'experiments/' + output + "-seeds-" + str(k) + ".cp"
        
        removeFile(seq_fname)
        print "Running sequential algorithm for k=%d" % k
        subprocess.Popen("python seq_estimation.py -dataset %s -seeds_file %s -cores 40 -results_file %s -output_mode 2 -min_samples 500\
        -min_relative_standard_error .05"%\
                         (dataset, seeds_fname, seq_fname), shell=True,stdout=subprocess.PIPE).stdout.read()
        for scale_factor in tau_scale_values:
            print "Running approx algorithm for k=: %d and scale factor %.2f" % (k, scale_factor)
            removeFile(approx_fname)
            subprocess.Popen("python ic_bfs_eval.py -dataset %s -res_fname %s -seeds %s -cores 40 -output_mode 2 -tau_scale %.3f" % 
                             (dataset, approx_fname, seeds_fname, scale_factor),
                             shell = True, stdout = subprocess.PIPE).stdout.read()
    
            k_val, mean_error, std_error = calculateMeanErrorAndSEM(approx_fname,seq_fname)
            f = open(results_fname,'a')
            f.write("%.3f\t%.3f\t%.3f\t%.3f\n"%(1.*k/n, scale_factor, mean_error, std_error))
            f.close()

    print "Elapsed time: ", time() - start_time
