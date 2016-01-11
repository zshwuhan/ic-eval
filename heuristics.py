from ic_bfs_eval import *
import subprocess
from ic_bfs_eval import EstimateInfluence
from bfs_seq import sequential_estimation
from random import sample,randint, seed
from common_tools import getNumCycles, removeFile

def runApproxHeuristic(dataset, seeds_fname, tau_scale, cores, init_samples = -1, iter_samples = -1):
    res_fname = "res" + str(randint(1, 1000))
    perf_fname = "perf" + str(randint(1, 1000))
    init_tau = -1
    num_cycles_approx = 0
    
    subprocess.Popen("perf stat -x, -o %s python load_link_server.py -cp %s"%\
                     (perf_fname, dataset), shell = True, stdout = subprocess.PIPE).stdout.read()
        
    nCycles_link_server = getNumCycles(perf_fname)
    removeFile(perf_fname)
    
    if init_samples > 0:
        subprocess.Popen("perf stat -x, -o %s python seq_estimation.py -dataset %s -seeds_file %s -results_file %s -output_mode 1 -nSamples %d -cores %d"%\
                         (perf_fname, dataset, seeds_fname, res_fname, init_samples, cores),\
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

def runVanilla(dataset, seeds, nSamples = 0, nCores = -1, min_samples = -1, min_relative_std_error = -1):
    perf_fname = "perf" + str(randint(1,1000))
    res_fname = "res" + str(randint(1,1000))

    subprocess.Popen("perf stat -x, -o %s python load_link_server.py -cp %s"%\
                              (perf_fname, dataset), shell = True, stdout = subprocess.PIPE).stdout.read()
    
    nCycles_link_server = getNumCycles(perf_fname)
    removeFile(perf_fname)
    print "running vanilla with max_samples == %d, min_samples == %d, and min_relative_std_error = %.8f"%(nSamples, min_samples, min_relative_std_error)
    subprocess.Popen("perf stat -x, -o %s python seq_estimation.py -dataset %s -seeds_file %s -results_file %s -output_mode 1 -nSamples %d -cores %d -min_samples %d -min_relative_standard_error %.8f"%\
                             (perf_fname, dataset, seeds, res_fname, nSamples, nCores, min_samples, min_relative_std_error),\
                             shell=True,stdout=subprocess.PIPE).stdout.read()

    num_cycles_seq = getNumCycles(perf_fname)
    print "dataset: ", dataset
    print "res_fname: ", res_fname
    print "dataset: ", dataset
    print "seeds file: ", seeds
    estimate_seq_capped = float(open(res_fname, 'r').readline().strip())
    removeFile(perf_fname)
    removeFile(res_fname)
    return estimate_seq_capped, num_cycles_seq
