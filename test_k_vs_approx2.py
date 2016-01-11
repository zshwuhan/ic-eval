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
            estimation_errors.append(100 * (ratio - 1.0))
        else:
            estimation_errors.append(100 * (1.0 - ratio))
    
    mean = (sum(estimation_errors)/len(estimation_errors))
    std_err = sem(estimation_errors)
    return k, mean, std_err

    
parser = argparse.ArgumentParser()
parser.add_argument('-csv', type = str, #required,
                    default = 'input/datasets/Wiki-Vote_stripped.txt', help="Name of source csv file (if we generate a new Link Server" )
parser.add_argument('-prob_method', type = int, #required,
                    default = 3, help='prob_method : method of setting edge-probabilities.\n 0 - fixed,\n1 - 1/in-deg,\n2 - uniform sample from the set [0.1,0.01].\n3 - u.a.r. Default: 3')
parser.add_argument('-undirected', type = int, #required,
                    default = 0, help='-undirected : is the graph an undirected graph. Default: 0 (no)')
parser.add_argument('-delim', type = int, default = 0,
                    help = '-delim : which delimiter (for edge-tuples) 0-tab, 1-space,2-comma. Default: 0')
parser.add_argument('-min', type = float, default = 0.01,
                    help = '-min : minimum fraction for k. algorithm. Default: 0.01')
parser.add_argument('-max', type = float, default = 0.11,
                    help = '-max : maximum fraction for k. algorithm. Default: 1.0')
parser.add_argument('-interval', type = float, default = 0.01,
                    help = '-interval : size of interval. Default: 0.01')
parser.add_argument('-samples', type = int, default = 10,
                    help = '-samples : number of samples of given seed set size. Default: 10')
parser.add_argument('-output', type = str, default = 'experiments/wiki/')
parser.add_argument('-title', type = str, default = '')
parser.add_argument('-dataset', type = str, default = 'input/datasets/wiki')

    
if __name__ == "__main__":
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
    L=LinkServerCP(dataset, edges_csv, create_new=True, prob_method=parameters.prob_method, prob=edge_prob, delim=delimiter, undirected=parameters.undirected)
    print "n = ", L.getNumNodes()
    V = LoadNodesFromFile(edges_csv, delimiter)
    n = len(V)
    print 'min_frac', min_frac
    k_min = int(n * min_frac)
    k_max = int(n * max_frac) + 1
    k_step = int(n * interval)
    

    generateSeedFiles(k_min, k_max, k_step, V, nSamples, 'experiments/' + output + "-seeds-")
    
    
    mean_errors, std_errors = [], []

    for k in xrange(k_min, k_max,k_step):
        approx_fname = 'experiments/results/' + output + "-approx_errors-k_min-%d-k_max-%d-k-%d-samples-%d"%(k_min,k_max,k,nSamples)
        seq_fname = 'experiments/results/' + output + "-seq-approx-errors-k-%d-samples-%d"%(k,nSamples)
        print approx_fname
        print seq_fname
        removeFile(approx_fname)
        seeds_fname = 'experiments/' + output + "-seeds-" + str(k) + ".cp"
        print "Running approx algorithm for k=: ", k
        subprocess.Popen("python ic_bfs_eval.py -dataset %s -res_fname %s -seeds %s -output_mode 2"%\
                        (dataset, approx_fname, seeds_fname), \
                        shell = True, stdout = subprocess.PIPE).stdout.read()
    
        removeFile(seq_fname)
        subprocess.Popen("python seq_estimation.py -dataset %s -seeds_file %s -results_file %s -output_mode 2"%\
                         (dataset, seeds_fname, seq_fname), shell=True,stdout=subprocess.PIPE).stdout.read()
        
        k, mean_error, std_error = calculateMeanErrorAndSEM(approx_fname,seq_fname)
        removeFile(approx_fname)
        removeFile(seq_fname)
        mean_errors.append(mean_error)
        std_errors.append(std_error)
    f = open("experiments/results/" + output + 'approximations-nSamples-%d-k_frac-%.3f-%.3f'%(nSamples, min_frac, max_frac),'w')
    for k, mean, std_err in zip(xrange(k_min, k_max,k_step), mean_errors, std_errors):
        f.write("%.3f\t%.3f\t%.3f\n"%(1.*k/n,mean,std_err))
    f.close()
        
    ## print "Done running algorithms, now plotting..."
    ## x_axis = [1.*x/n for x in xrange(k_min,k_max,k_step)]
    ## print "x axis: ", x_axis
    ## print "mean errors: ", mean_errors
    ## print "std errors: ", std_errors
    ## print "title: ", parameters.title
    ## plot_fname = 'experiments/figures/' + output + 'approximations-nSamples-%d-k_frac-%.3f-%.3f.pdf'%(nSamples, min_frac, max_frac)
    ## #print "plot file name: ", plot_fname
    ## #plot2d([1.0*x/n for x in xrange(k_min,k_max,k_step)],[mean_errors], [std_errors], ['bla'], [r'$\frac{k}{n}$','Error percentage'], parameters.title, plot_fname)

    print "Elapsed time: ", time() - start_time
