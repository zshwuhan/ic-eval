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
def loadDataFromFiles(approx_fnames_list,seq_fname):
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
    
    approx_values_lists = [[] for a in range(len(approx_fnames_list))]
    for i, approx_fname in enumerate(approx_fnames_list):
        f = open(approx_fname, 'r')
        approx = f.readline().strip()
        while not approx == '':
            approx_values_lists[i].append(float(approx))
            f.readline()
            f.readline()
            approx = f.readline().strip()
        f.close()

    seq_values = []
    f=open(seq_fname,'r')
    influence_estimate = f.readline().strip()
    while not influence_estimate == '':
        print "adding estimate: ", influence_estimate
        seq_values.append(float(influence_estimate))
        f.readline().strip()
        influence_estimate = f.readline().strip()
        
    f.close()
    return approx_values_lists, seq_values

def calculateAvgErrorPerScale(approx_fnames,seq_fname):

    approx_values_lists, influence_estimate = loadDataFromFiles(approx_fnames,seq_fname)
    avg_list, errs_list = [], []
    for approx_values in approx_values_lists:
        estimation_errors = []
        for j, approx in enumerate(approx_values):
            ratio = 1.0 * approx / influence_estimate[j]
            if ratio >= 1:
                estimation_errors.append(ratio)
            else:
                estimation_errors.append(1./ratio)
    
        avg_list.append(np.mean(estimation_errors))
        errs_list.append(sem(estimation_errors))
    return avg_list, errs_list

def plotErrorRatesForVaryingScales(approx_fnames, seq_fname,min_scale,max_scale,step, pdf_fname, title):
    avg_list, errs_list = calculateAvgErrorPerScale(approx_fnames,seq_fname)
    plot2d(np.array([x for x in drange(min_scale,max_scale,step)]),[avg_list], [errs_list], ['bla'], ['Samples scale factor','Mean approximation ratio'], title, pdf_fname)
    
parser = argparse.ArgumentParser()
parser.add_argument('-csv', type = str, #required,
                    default = 'input/datasets/Wiki-Vote_stripped.txt', help="Name of source csv file (if we generate a new Link Server" )
parser.add_argument('-prob_method', type = int, #required,
                    default = 3, help='prob_method : method of setting edge-probabilities.\n 0 - fixed,\n1 - 1/in-deg,\n2 - uniform sample from the set [0.1,0.01].\n3 - u.a.r. Default: 3')

parser.add_argument('-delim', type = int, default = 0,
                    help = '-delim : which delimiter (for edge-tuples) 0-tab, 1-space,2-comma. Default: 0')
parser.add_argument('-min', type = float, default = 0.01,
                    help = '-min : minimum scale factor of number of samples for the approx. algorithm. Default: 0.01')
parser.add_argument('-max', type = float, default = 0.11,
                    help = '-max : maximum scale factor of number of samples for the approx. algorithm. Default: 1.0')
parser.add_argument('-interval', type = float, default = 0.01,
                    help = '-interval : size of interval. Default: 0.01')
parser.add_argument('-samples', type = int, default = 10,
                    help = '-samples : number of samples of given seed set size. Default: 10')
parser.add_argument('-output', type = str, default = 'experiments/wiki/')
parser.add_argument('-title', type = str, default = '')
parser.add_argument('-dataset', type = str, default = 'input/datasets/wiki')
parser.add_argument('-k_frac', type = float, default = 0.1,
                    help = '-k_frac : fraction of |V| to be used as seeds (uniform selection from V). Default: 0.1')

parser.add_argument('-seq_samples_scale', type = float, default = 1.0)

parser.add_argument('-eps', type = float, default = 0,
                    help = '-eps: convergence threshold. Default: 0 (runs until reached theoretically required # of samples)')

    
if __name__ == "__main__":
    parameters = parser.parse_args()
    k_frac, min_frac, max_frac, interval, nSamples, edges_csv, delim_option, output, dataset, seq_samples_scale, eps = parameters.k_frac, parameters.min, \
      parameters.max, parameters.interval, parameters.samples, parameters.csv, parameters.delim, parameters.output, parameters.dataset, parameters.seq_samples_scale,\
      parameters.eps
    
    print "Running estimation on varying scaling factors of L -- the number of samples to take in each iteration."
    print "Input file: ", edges_csv
    print "Output file prefix: ", output
    print "Seed set size as fraction of n: ", k_frac
    print "Convergence threshold: ", eps
    delimiter = delim_dict[delim_option]
    start_time = time()
    bfs_method = 'seq'
    print "creating link-server object"
    if parameters.prob_method == 0:
        edge_prob = 0.2
    else:
        edge_prob = [0.1,0.01]
    L=LinkServerCP(dataset, edges_csv, create_new=True, prob_method=parameters.prob_method, prob=edge_prob, delim=delimiter)
    print "n = ", L.getNumNodes()
    V = LoadNodesFromFile(edges_csv, delimiter)
    n = len(V)
    k = int(n * k_frac)
    print "n=", n
    print "k=", k
    seeds_fname = output + "-seeds-" + str(k) + ".cp"
    removeFile(seeds_fname)
    removeFile(dataset)
    generateSeedFiles(k, k+1, 1, V, nSamples, output + "-seeds-")

    approx_fnames = ['experiments/results/' + output + "-approx-k-%d-samples-%d-scale-%.5f"%(k,nSamples,scale) for scale in drange(min_frac,max_frac,interval)]
    seq_fname = 'experiments/results/' + output + "-seq-k-%d-samples-%d-eps-%.5f"%(k,nSamples,eps)

    for i, scale in enumerate(drange(min_frac, max_frac, interval)):
        removeFile(approx_fnames[i])
        print "Running approx algorithm for scale factor: ", scale
        subprocess.Popen("python ic_bfs_eval.py -dataset %s -scale %.5f -res_fname %s -seeds %s -output_mode 0"%\
                         (dataset, scale, approx_fnames[i], seeds_fname), \
                         shell = True, stdout = subprocess.PIPE).stdout.read()
    
    print "Running sequential algorithm on seed sets with eps=%.5f"%eps
    removeFile(seq_fname)
    subprocess.Popen("python seq_estimation.py -dataset %s -seeds_file %s -results_file %s -output_mode 2 -min_samples 500 -min_relative_standard_error 0.01 "%\
                     (dataset, seeds_fname, seq_fname), shell=True,stdout=subprocess.PIPE).stdout.read()

    plotErrorRatesForVaryingScales(approx_fnames,seq_fname, min_frac,max_frac, interval, 'experiments/figures/' + output+'-nSamples-%d-scale_samples-errors-%.5f-%.5f-k_frac-%.3f-eps-%.5f.pdf'%(nSamples, min_frac, max_frac,k_frac,eps),parameters.title)

    print "Elapsed time: ", time() - start_time
