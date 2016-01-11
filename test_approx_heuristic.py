from ic_bfs_eval import *
from generate_seeds import LoadNodesFromFile, generateSeedFiles
import subprocess
from link_server import LinkServerCP
from ic_bfs_eval import EstimateInfluence
from bfs_seq import sequential_estimation

from time import time
import argparse
import numpy as np

from math import log, ceil
import gc
from math import log
from common_tools import *
from random import sample,randint, seed

parser = argparse.ArgumentParser()
parser.add_argument('-csv', type = str, #required,
                    default = 'input/datasets/Wiki-Vote_stripped.txt', help="Name of source csv file (if we generate a new Link Server" )
parser.add_argument('-delim', type = int, default = 0,
                    help = '-delim : which delimiter (for edge-tuples) 0-tab, 1-space,2-comma. Default: 0')
parser.add_argument('-samples', type = int, default = 5,
                    help = '-samples : number of samples of given seed set size. Default: 10')
parser.add_argument('-cores', type = int, default = 50,
                    help = '-cores : number of cores of to use. Default: 50')
parser.add_argument('-undirected', type = int, default = 0,
                    help = '-undirected : is the graph undirected')
parser.add_argument('-output', type = str, default = 'out')
parser.add_argument('-dataset', type = str, default = 'input/datasets/wiki')
parser.add_argument('-init_samples', type = int, default = 5,
                    help = '-init_samples : number of samples to take for initial tau value')
parser.add_argument('-iter_samples', type = int, default = 5,
                    help = '-guess_samples : number of samples to take for initial tau value')
parser.add_argument('-max_k', type = float, default = .01,
                    help = '-max_k : maximum value for k/n. Default: .5')
parser.add_argument('-min_k', type = float, default = .001,
                    help = '-min_k : minimum value for k/n. Default: .05')
parser.add_argument('-k_mode', type = int, default = 0,
                    help = '-k_mode : 0 -- take fractional values, 1 -- min_k, max_k, k_steps are integral')

parser.add_argument('-k_step', type = float, default = .001,
                    help = '-k_step : .05')
parser.add_argument('-prob_method', type = int, default = 3)
parser.add_argument('-prob', type= str, default = '')
parser.add_argument('-tau_scale', type = float, default = 0.1,
                    help = '-tau_scale : scaling factor for tau. Default: 0.1')

def runApproxHeuristic(dataset, seeds_fname, tau_scale, cores, init_samples, iter_samples, nCycles_link_server):
    res_fname = "res" + str(randint(1, 1000))
    perf_fname = "perf" + str(randint(1, 1000))

    subprocess.Popen("perf stat -x, -o %s python seq_estimation.py -dataset %s -seeds_file %s -results_file %s -output_mode 1 -nSamples %d -cores %d"%\
                     (perf_fname, dataset, seeds_fname, res_fname, init_samples, parameters.cores),\
                     shell=True,stdout=subprocess.PIPE).stdout.read()
    
    init_tau = float(open(res_fname, 'r').readline().strip()) / (1 - tau_scale)
    print "Initial tau value set to: ", init_tau
    num_cycles_approx = getNumCycles(perf_fname) - nCycles_link_server
    removeFile(perf_fname)
    removeFile(res_fname)
    subprocess.Popen("perf stat -x, -o %s python ic_bfs_eval.py -tau_scale %.3f -dataset %s -res_fname %s -seeds %s -output_mode 2 -cores %d -init_tau %.3f -iter_samples %d"%\
                     (perf_fname, tau_scale, dataset, res_fname, seeds_fname, cores, init_tau, iter_samples), \
                     shell = True, stdout = subprocess.PIPE).stdout.read()
    estimate_approx = float(open(res_fname, 'r').readline().strip())
    print "Estimated value: ", estimate_approx
    num_cycles_approx += getNumCycles(perf_fname) - nCycles_link_server

    removeFile(res_fname)
    removeFile(perf_fname)

    return (estimate_approx, num_cycles_approx)


def runVanilla(dataset, seeds_fname, nSamples, cores):
    perf_fname = "perf" + str(randint(1,1000))
    res_fname = "res" + str(randint(1,1000))
    subprocess.Popen("perf stat -x, -o %s python seq_estimation.py -dataset %s -seeds_file %s -results_file %s -output_mode 1 -nSamples %d -cores %d"%\
                             (perf_fname, dataset, seeds_fname, res_fname, nSamples, cores),\
                             shell=True,stdout=subprocess.PIPE).stdout.read()

    num_cycles_seq = getNumCycles(perf_fname) - nCycles_link_server
    print "dataset: ", dataset
    print "res_fname: ", res_fname
    print "dataset: ", dataset
    print "seeds file: ", seeds_fname
    estimate_seq_capped = float(open(res_fname, 'r').readline().strip())
    removeFile(perf_fname)
    removeFile(res_fname)
    return estimate_seq_capped, num_cycles_seq

if __name__ == "__main__":
    seed()
    gc.enable()
    parameters = parser.parse_args()
    min_k, max_k, k_step, nSamples, edges_csv, delim_option, output, dataset, tau_scale, init_samples, iter_samples = \
      parameters.min_k, parameters.max_k, parameters.k_step, parameters.samples, parameters.csv, parameters.delim, \
      parameters.output, parameters.dataset, parameters.tau_scale, parameters.init_samples, parameters.iter_samples

    delim_dict = {0 : '\t', 1 : ' ', 2 : ','}
    delimeter = delim_dict[delim_option]
    results_dir = 'experiments/results/approx-heuristic1/'
    bUndirected = True if parameters.undirected == 1 else False
    bfs_method = 'seq'
    start_time = time()
    
    
    if parameters.prob_method == 2:
        if parameters.prob == '':
            prob_values = [0.001, 0.0001]
        else:
            l = parameters.prob.split(',')
            prob_values = [float(x) for x in l]
            
    elif parameters.prob_method == 3:
        if parameters.prob == '':
            prob_values = [0,.01]
        else:
            l = parameters.prob.split(',')
            assert len(l) == 2
            prob_values = [float(x) for x in l]
    print "Probability method: %d, probability values: %s"%(parameters.prob_method, prob_values)
    L=LinkServerCP(dataset, edges_csv, create_new=True, prob_method=parameters.prob_method, prob=prob_values, delim=delimeter, undirected = bUndirected)
    print "Number of seed sets: ", nSamples
    n = L.getNumNodes()
    if parameters.k_mode == 0:
        min_k, max_k, k_step = int(min_k * n), int(max_k * n), int(k_step * n)
    
    perf_csv_fname = "perf_out" + str(randint(1,1000))
    nBFS_samples_theoretic = n * log(n,2)
    nBFS_samples = 1000 #nBFS_samples_theoretic
    
    results_file = results_dir + dataset +\
      '-approx-heuristic-prob_method-%d-k_min-%.4f-k_max-%.4f-tau_scale-%.3f-samples-%d-bfs_samples-%d-init_samples-%d-iter_samples-%d'%\
      (parameters.prob_method, min_k, max_k, tau_scale, nSamples, nBFS_samples, init_samples, iter_samples)
    removeFile(results_file)
       
    # Record time to load link-server file
    subprocess.Popen("perf stat -x, -o %s python load_link_server.py -cp %s"%\
                              (perf_csv_fname, dataset), shell = True, stdout = subprocess.PIPE).stdout.read()
    nCycles_link_server = getNumCycles(perf_csv_fname)
    print "Number of cycles for loading link server: ", nCycles_link_server
    removeFile(perf_csv_fname)
    for k in xrange(int(min_k),int(max_k + 1), int(k_step)):
        print "k = ", k
        seeds_fname = "%s-seeds-%d.cp"%(output, k)
        for i in xrange(nSamples):
            print "sample #",i
            generateSeedFiles(k, k+1, 1, range(n), 1, output + "-seeds-")
            seeds=cp.load(open(seeds_fname,'r'))
            print "Running Vanilla with %d samples"%(nBFS_samples)
            true_value, num_cycles_full = runVanilla(dataset, seeds_fname, nBFS_samples, parameters.cores)            
            print "Done. Number of cycles: %d"%num_cycles_full
            nCycles_per_bfs = 1.*(num_cycles_full - nCycles_link_server) / nBFS_samples
            print "Number of cycles per sample", nCycles_per_bfs
            print "Running approximation algorithm"
            approx_estimate, num_cycles_approx = runApproxHeuristic(dataset, seeds_fname, tau_scale, parameters.cores,\
                                                 parameters.init_samples, parameters.iter_samples, nCycles_link_server)
            print "Number of cycles without link-server loading: ", num_cycles_approx
            print "Done approximating, now running naive sequential algorithm"
            
            nVanilla_samples = int(ceil(1. * num_cycles_approx / nCycles_per_bfs))
            print "Running Vanilla for %s samples"%nVanilla_samples
            seq_estimate, num_cycles_seq = runVanilla(dataset, seeds_fname, nVanilla_samples, parameters.cores)
            num_cycles_seq = num_cycles_seq - nCycles_link_server
            print "Done running capped sequential algorithm"
            print "Number of cycles: ", num_cycles_seq
            print "Ratio of # of cycles_approx to # of cycles_vanilla_capped : ", 1.*num_cycles_approx / num_cycles_seq
            removeFile(seeds_fname)
            f = open(results_file, 'a')
            f.write('%.4f\t%.6f\t%.6f\t%.6f\n'%(1.*k/n if parameters.prob_method==0 else float(k), true_value, approx_estimate, seq_estimate))
            f.close()
            
    removeFile(dataset)
    print "Total runtime: ", (time() - start_time)/60.0
