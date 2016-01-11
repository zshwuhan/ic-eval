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
parser.add_argument('-cores', type = int, default = 30,
                    help = '-cores : number of cores of to use. Default: 30')
parser.add_argument('-undirected', type = int, default = 0,
                    help = '-undirected : is the graph undirected')
parser.add_argument('-output', type = str, default = 'out')
parser.add_argument('-dataset', type = str, default = 'input/datasets/wiki')
parser.add_argument('-title', type = str, default = '',
                    help = '-title : title of the generated plot. Default : blank')
parser.add_argument('-seq_samples_scale', type = float, default = 1.0)
parser.add_argument('-prob_method', type = int, default = 3)

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
    gc.enable()
    parameters = parser.parse_args()
    min_frac, max_frac, interval, nSamples, edges_csv, delim_option, output, dataset, seq_samples_scale = parameters.min, \
      parameters.max, parameters.interval, parameters.samples, parameters.csv, parameters.delim, parameters.output, parameters.dataset, parameters.seq_samples_scale#,\
      #[float(x) for x in parameters.eps.split(',')], parameters.title

    delim_dict = {0 : '\t', 1 : ' ', 2 : ','}
    delimeter = delim_dict[delim_option]
    results_dir = 'experiments/results/'
    bUndirected = True if parameters.undirected == 1 else False
    bfs_method = 'seq'
    start_time = time()
    print "creating link-server object"
    if delimeter == "\t":
        print "delimeter is tab"
    L=LinkServerCP(dataset, edges_csv, create_new=True, prob_method=parameters.prob_method, prob=[0.1,0.01], delim=delimeter, undirected = bUndirected)
    n = L.getNumNodes()
    #n = len(V)
    print "Number of nodes: ", n
    nBFS_samples = 1000
    running_times_file = results_dir + output + '-running_times-ratios_k_min-%.3f-k_max-%.3f-samples-%d-bfs_samples-%d-large'%(min_frac,max_frac,nSamples,nBFS_samples)
    running_times_file_raw = running_times_file + "-raw"
    removeFile(running_times_file)
    f = open(running_times_file, 'w')
    f_raw = open(running_times_file_raw,'w')
    nBFS_samples = 1000
    nBFS_samples_theoretic = n * log(n,2)
    for nSeeds in xrange(int(min_frac * n), int(max_frac * n), int(interval * n)):
        print "k = ", nSeeds
        seeds_fname = output + "-seeds-" + str(nSeeds) + '.cp'
        runtimes_approx, runtimes_seq = [], []
        for i in xrange(nSamples):
            generateSeedFiles(nSeeds, nSeeds+1, int(interval * n), range(n), 1, output + "-seeds-")
            perf_csv_fname = dataset + 'runtimes_large.csv'
            subprocess.Popen("perf stat -x, -o %s python ic_bfs_eval.py -dataset %s -res_fname %s -seeds %s -output_mode 3 -cores %d"%\
                              (perf_csv_fname, dataset, output + "-approx-" + str(nSeeds), seeds_fname, parameters.cores), \
                              shell = True, stdout = subprocess.PIPE).stdout.read()
            num_cycles_approx = getNumCycles(perf_csv_fname)
            runtimes_approx.append(num_cycles_approx)
            print "Done approximating, now running naive sequential algorithm"
            removeFile(perf_csv_fname)
            subprocess.Popen("perf stat -x, -o %s python seq_estimation.py -dataset %s -seeds_file %s -results_file %s -output_mode 3 -nSamples %d -cores %d"%(perf_csv_fname, dataset, seeds_fname, output + "-seq-" + str(nSeeds), nBFS_samples, parameters.cores), shell=True,stdout=subprocess.PIPE).stdout.read()
            runtime_seq_samples = getNumCycles(perf_csv_fname)
            theoretic_num_cycles = 1.0 * runtime_seq_samples / nBFS_samples * nBFS_samples_theoretic
            runtimes_seq.append(theoretic_num_cycles)
            f_raw.write('%.3f\t%.3f\t%.3f\n'%(1.*nSeeds/n, num_cycles_approx, theoretic_num_cycles))
            removeFile('runtimes_large_seq.csv')
        print "runtimes_approx: ", runtimes_seq
        print "runtimes_seq: ", runtimes_seq
        ratios = 1. * np.array(runtimes_seq) / np.array(runtimes_approx)
        print "ratios: ", ratios
        f.write("%.3f\t%.3f\t%.3f\n"%(1.0*nSeeds/n, np.mean(ratios), sem(ratios)))
    f.close()
    f_raw.close()
    print "Total runtime: ", (time() - start_time)/60.0
