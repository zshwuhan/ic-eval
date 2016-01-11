from ic_bfs_eval import *
from generate_seeds import LoadNodesFromFile, generateSeedFiles
import subprocess
import numpy as np
from link_server import LinkServerCP
from ic_bfs_eval import EstimateInfluence
from bfs_seq import sequential_estimation
import cPickle as cp
from time import time
import argparse
from plot_running_times import plotRatiosFromFile
from graph_tools import createRandomGraph
from common_tools import *
from random import randint
from math import log
parser = argparse.ArgumentParser()
parser.add_argument('-n_min', type = int, default = 100,
                    help = '-n_min : Minimum number of nodes in the random graph. Default: 100')
parser.add_argument('-n_max', type = int, default = 10000,
                    help = '-n_max : Maximum number of nodes in the random graph. Default: 10000')
parser.add_argument('-n_step', type = int, default = 1000,
                    help = '-n_min : Minimum number of nodes in the random graph. Default: 100')
parser.add_argument('-graph_method', type = int, default = 0,
                    help = '-graph_method : which random graph generation to use. \n0 - Erdos-Renyi graph G(n,p) \n1 - Watts-Strogatz(n,K,beta)\nDefault: G(n,0.2)')
parser.add_argument('-prob_method', type = int, default = 3)
parser.add_argument('-er_p' , type = float, default = 0.2,
                    help = '-er_p : the edge generation probability of the Erdos-Renyi G(n,p) model. Default: 0.2')
parser.add_argument('-ws_k' , type = int, default = 200,
                    help = '-ws_k : The k value (degree) of the Watts Strogatz model. Default: 200')
parser.add_argument('-ws_beta' , type = float, default = 0.3,
                    help = '-ws_beta : The beta value (rewiring probability) of the Watts Strogatz model. Default: 0.3')

parser.add_argument('-min', type = float, default = 0.1,
                     help = '-min : minimum number of seed nodes per set. Default: 0.01')
parser.add_argument('-max', type = float, default = 1.0,
                     help = '-max : maximum number of seed nodes per set. Default: 0.11')
parser.add_argument('-interval', type = float, default = 0.1,
                     help = '-interval : size of interval. Default: 0.01')
parser.add_argument('-samples', type = int, default = 10,
                    help = '-samples : number of samples of given seed set size. Default: 10')
parser.add_argument('-cores', type = int, default = -1,
                    help = '-cores : number of processes to open concurrently. Default: -1')
parser.add_argument('-output', type = str, default = 'experiments/wiki/')
parser.add_argument('-dataset', type = str, default = 'input/datasets/wiki')
parser.add_argument('-title', type = str, default = '',
                    help = '-title : title of the generated plot. Default : blank')
parser.add_argument('-seq_samples_scale', type = float, default = 1.0)
parser.add_argument('-eps', type = str, default = [0],
                    help = '-eps : a list of convergence thresholds. 0 - run until getting n*log(n) samples (not until convergence). Default : [0]')

def getNumCycles(fname):
    f = open(fname, 'r')
    for line in f.readlines():
        words = line.split(',')
        if 'cycles\n' in words:
            f.close()
            return int(words[0])
    print "Error! Could not determine number of cycles."
    print "Content of perf output file:"
    f.seek(0)
    for line in f.readlines():
        print line
    raise Exception("Invalid perf output file")

def loadAvgRatiosFromFile(fname):
    d = defaultdict(list)
    f = open(fname, 'r')
    bDone = False
    while not bDone:
        k = f.readline().strip()
        if k == '':
            bDone = True
        else:
            k=int(k.strip())
            approx_cycles = float(f.readline())
            seq_cycles = float(f.readline())
            d[k].append(1.0 * seq_cycles/approx_cycles)
    k_vals = d.keys()
    k_vals.sort()
    avg_ratios = [sum(d[k])/len(d[k]) for k in k_vals]
    return k_vals, avg_ratios
    
def loadAxesFromFiles(fnames, n_values):
    assert len(fnames) == len(n_values)
    points = []
    for i, n in enumerate(n_values):
        nSeeds_list, avg_ratios = loadAvgRatiosFromFile(fnames[i])
        new_points = zip([n for r in avg_ratios], [1.*k/n for k in nSeeds_list], avg_ratios)
        points += new_points
    n_axis = np.array([point[0] for point in points])
    k_axis = np.array([point[1] for point in points])
    mean_axis = np.array([point[2] for point in points])

    xi = np.linspace(-1,1,20)
    yi = np.linspace(-1,1,20)
    num_n_values = len(n_values)
    num_k_values = len(n_axis) / len(n_values)
    assert len(set([point[1] for point in points])) == num_k_values,\
      "Number of distinct k values is: %d, whereas num_points / num_n_points = %d"%(len(set([point[1] for point in points])), num_k_values)
    
    return points

if __name__ == "__main__":
    parameters = parser.parse_args()
    n_min, n_max, n_step, min_frac, max_frac, interval, nSamples, output, dataset, seq_samples_scale,\
      eps_list, title = parameters.n_min, parameters.n_max, parameters.n_step, parameters.min, \
      parameters.max, parameters.interval, parameters.samples, parameters.output, parameters.dataset, parameters.seq_samples_scale,\
      [float(x) for x in parameters.eps.split(',')], parameters.title
    nBFS_samples = 500
    bfs_method = 'seq'
    perf_output_fname = "out%d.csv"%sample(range(1000),1)[0]
    for n in xrange(n_min, n_max, n_step):
        print "Running tests for n=",n
        
        csv_fname = "random_graph-n-%d-method-%d.csv"%(n, parameters.graph_method)
        running_times_files = [output + '-running_times-n-%d-graph_method-%d-samples-prob_method-%d-%d-eps-%.5f'%(n, parameters.graph_method, nSamples, parameters.prob_method, eps) for eps in eps_list]
        for fname in running_times_files:
            removeFile(fname)
        createRandomGraph(csv_fname, parameters.graph_method, n, parameters.er_p, parameters.ws_beta, parameters.ws_k)
        L=LinkServerCP('input/datasets/' + dataset, csv_fname, create_new=True, prob_method=parameters.prob_method, prob=[0.1,0.01], delim='\t', undirected = 1)
        # record loading time of link-server -- for interpolation
        removeFile(perf_output_fname)
        subprocess.Popen("perf stat -x, -o %s python load_link_server.py -cp %s"%\
                         (perf_output_fname, "input/datasets/" + dataset), shell = True, stdout = subprocess.PIPE).stdout.read()
        
        nCycles_link_server = getNumCycles(perf_output_fname)
        removeFile(perf_output_fname)
        
        #removeFile(csv_fname)
        V = xrange(n)
        
        for nSeeds in xrange(int(min_frac * n), int(max_frac * n), int(interval * n)):
            for i in xrange(nSamples):
                seeds_fname = output + "-seeds-%d-%d.cp"%(i, nSeeds)
                generateSeedFiles(nSeeds, nSeeds+1, int(interval * n), V, 1, output  + "-seeds-%d-"%i)
                subprocess.Popen("perf stat -x, -o %s python ic_bfs_eval.py -dataset %s -cores %d -res_fname %s -seeds %s -output_mode 3 -undirected 1"%\
                                  (perf_output_fname, 'input/datasets/' + dataset, parameters.cores, output + "-approx-" + str(nSeeds), seeds_fname), \
                                  shell = True, stdout = subprocess.PIPE).stdout.read()
                cycles_approx = getNumCycles(perf_output_fname)

                cycles_seq = {}
                print "Done approximating, now running naive sequential algorithm"
                for eps in eps_list:
                    print "Running sequential for eps = ", eps
                    removeFile(perf_output_fname)                    
                    subprocess.Popen("perf stat -x, -o %s python seq_estimation.py -dataset %s -cores %d -seeds_file %s -results_file %s -output_mode 3 -nSamples %d"%(perf_output_fname, 'input/datasets/' + dataset, parameters.cores, seeds_fname, output + "-seq-" + str(nSeeds), nBFS_samples), shell=True,stdout=subprocess.PIPE).stdout.read()
                    cycles_seq[eps] = (n*log(n,2) / nBFS_samples) * (getNumCycles(perf_output_fname) - nCycles_link_server) + nCycles_link_server
                removeFile(perf_output_fname)
                for i, eps in enumerate(eps_list):
                    f = open(running_times_files[i],'a')
                    f.write(str(nSeeds) + '\t' + str(1. * cycles_approx / nSamples) + '\t' + str(1. * cycles_seq[eps] / nSamples) + '\n')
                    ## f.write(str(nSeeds) + '\n')
                    ## f.write(str(1. * cycles_approx / nSamples) + '\n')
                    ## f.write(str(1. * cycles_seq[eps] / nSamples) + '\n')
                    f.close()

        removeFile('input/datasets/' + dataset)
