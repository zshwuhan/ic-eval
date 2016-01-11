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
from random import seed, random
parser = argparse.ArgumentParser()
parser.add_argument('-csv', type = str, #required,
                    default = 'input/datasets/Wiki-Vote_stripped.txt', help="Name of source csv file (if we generate a new Link Server" )
parser.add_argument('-delim', type = int, default = 0,
                    help = '-delim : which delimiter (for edge-tuples) 0-tab, 1-space,2-comma. Default: 0')
parser.add_argument('-min', type = float, default = 0.1,
                     help = '-min : minimum number of seed nodes per set. Default: 0.01')
parser.add_argument('-max', type = float, default = 1.0,
                     help = '-max : maximum number of seed nodes per set. Default: 0.11')
parser.add_argument('-interval', type = float, default = 0.1,
                     help = '-interval : size of interval. Default: 0.01')
parser.add_argument('-samples', type = int, default = 10,
                    help = '-samples : number of samples of given seed set size. Default: 10')
parser.add_argument('-output', type = str, default = 'experiments/wiki/')
parser.add_argument('-dataset', type = str, default = 'input/datasets/wiki')
parser.add_argument('-title', type = str, default = '',
                    help = '-title : title of the generated plot. Default : blank')

#parser.add_argument('-eps', type = str, default = [0],
#                    help = '-eps : a list of convergence thresholds. 0 - run until getting n*log(n) samples (not until convergence). Default : [0]')

def getNumCycles(fname):
    f = open(fname, 'r')
    for line in f.readlines():
        words = line.split(',')
        if 'cycles\n' in words:
            f.close()
            return int(words[0])
def removeFile(fname):
    try:
        os.remove(fname)
    except OSError:
        print "failed to delete the file: ",fname
        pass
    
if __name__ == "__main__":
    parameters = parser.parse_args()
    min_frac, max_frac, interval, nSamples, edges_csv, delim_option, output, dataset= parameters.min, \
      parameters.max, parameters.interval, parameters.samples, parameters.csv, parameters.delim, parameters.output, parameters.dataset#,\
      #[float(x) for x in parameters.eps.split(',')], parameters.title
    seed()
    perf_fname = "runtimes" + str(random())
    delim_dict = {0 : '\t', 1 : ' ', 2 : ','}
    delimeter = delim_dict[delim_option]
    results_dir = 'experiments/results/'
    start_time = time()
    bfs_method = 'seq'

    print "creating link-server object"
    if delimeter == "\t":
        print "delimeter is tab"
    L=LinkServerCP(dataset, edges_csv, create_new=True, prob_method=2, prob=[0.1,0.01], delim=delimeter)
    V = LoadNodesFromFile(edges_csv, delimeter)
    n = len(V)
    print "Number of nodes: ", n
    running_times_file = results_dir + output + '-running_times-ratios_k_min-%.3f-k_max-%.3f-samples-%d'%(min_frac,max_frac,nSamples)
    running_times_file_raw = running_times_file + "-raw"
    removeFile(running_times_file)
    f = open(running_times_file, 'w')
    f_raw = open(running_times_file_raw, 'w')
    for nSeeds in xrange(int(min_frac * n), int(max_frac * n), int(interval * n)):
        print "k = ", nSeeds
        seeds_fname = output + "-seeds-" + str(nSeeds) + '.cp'
        runtimes_approx, runtimes_seq = [], []
        for i in xrange(nSamples):
            generateSeedFiles(nSeeds, nSeeds+1, int(interval * n), V, 1, output + "-seeds-")
            approx_start = time()
            subprocess.Popen("perf stat -x, -o %s python ic_bfs_eval.py -dataset %s -res_fname %s -seeds %s -output_mode 3 -output_results 0"%\
                              (perf_fname, dataset, output + "-approx-" + str(nSeeds), seeds_fname), \
                              shell = True, stdout = subprocess.PIPE).stdout.read()
            print "running time for approximation algorithm", (time()- approx_start)/60
            runtime_approx = getNumCycles(perf_fname)
            runtimes_approx.append(runtime_approx)
            print "Done approximating, now running naive sequential algorithm"
            removeFile(perf_fname)
            seq_start = time()
            subprocess.Popen("perf stat -x, -o %s python seq_estimation.py -dataset %s -seeds_file %s -results_file %s -output_mode 3"%(perf_fname, dataset, seeds_fname, output + "-seq-" + str(nSeeds)), shell=True,stdout=subprocess.PIPE).stdout.read()
            print "running time for sequential algorithm: ", (time() - seq_start)/60
            runtime_seq = getNumCycles(perf_fname)
            runtimes_seq.append(runtime_seq)
            removeFile(perf_fname)
            f_raw.write("%.3f\t%s\t%s\n"%(1.*nSeeds / n, runtime_raw, runtime_approx, runtime_seq))
        print "runtimes_approx: ", runtimes_approx
        print "runtimes_seq: ", runtimes_seq
        ratios = 1. * np.array(runtimes_seq) / np.array(runtimes_approx)
        print "ratios: ", ratios
        f.write("%.3f\t%.3f\t%.3f\n"%(1.0*nSeeds/n, np.mean(ratios), sem(ratios)))
    f.close()
    f_raw.close()
    print "Total running time (in minutes): ", (time() - start_time)/60
