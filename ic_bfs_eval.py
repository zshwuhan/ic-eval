
import time
import argparse

from collections import defaultdict,namedtuple
import os
from link_server import *
from math import log
from random import sample
import cProfile, pstats,StringIO
from bfs_seq import IC_Sample

try:
    from mr_ic_bfs_init import MRBFSSampleInit
    from mr_ic_bfs import MRBFSSampleIter
    from mrjob.protocol import JSONProtocol, JSONValueProtocol
    from mr_counter2 import *
    from mrjob.util import log_to_stream
    from globals import *
except:
    pass

#---------
debug_mode = True
#output_mode = 1 # 0 - print to screen only, 1 - print to file only, 2 - print to both file and screen
output_file = 'output.txt'
#debug_mode = False
seeds_file = 'seeds.txt'
interim_file = 'intermediateResults'
dataset_name = 'wiki'
#csv_file = 'input/datasets/Wiki-Vote_stripped.txt'
edge_prob_type = 2
probs = [0.1,0.01]
generate_database = True
sql_mode = False
total_steps_cap = 0
#-----------
parser = argparse.ArgumentParser()
parser.add_argument('-csv', type = str, #required,
                    default = 'input/datasets/Epinions1.csv', help="Name of source csv file (if we generate a new Link Server" )
parser.add_argument('-undirected', type = int, default = 0,
                    help="-undirected : is the graph undirected (0 -- No, 1 -- Yes). Default : 0" )
parser.add_argument('-dataset', type = str, default = 'input/datasets/epinions',
                    help = '-dataset : dataset to use. Default: wiki')
parser.add_argument('-generate_database', type = int, default = 0,
                    help = '-generate_database : generate a new database file (1), or load existing (0)')
parser.add_argument('-edge_prob_type', type = int, default = 1,
                    help = '-edge_prob_type : method of generating edge probabilities. 0 - uniform. 1 - 1/in-deg(v), 2 - uid from [0.1,0.01]')
parser.add_argument('-probs', type = float, default = 0.5,
                    help = '-probs : uniform probabiliy value')
parser.add_argument('-sql', type = int, default = 0,
                     help = '-sql : generate sql (1) database, or use a dictionary')
parser.add_argument('-res_fname', type = str, default = 'results',
                    help = 'results file name')
parser.add_argument('-seeds', type = str, default = 'experiments/wiki/seeds/seeds1000.cp',
                    help = 'number of seeds')
parser.add_argument('-bfs_method', type = str, default = 'seq',
                    help = 'BFS method: emr -- EMR, seq -- Sequential BFS (with parallelization on multiple cores for efficiency)')
parser.add_argument('-cores', type = int, default = -1,
                    help = 'Number of cores to use if bfs-method=seq (if none specified -- use all core)')
parser.add_argument('-output_results', type = int, default = 1,
                    help = '-output_results: write tau value etc. to file. Default: 1 (yes)')
parser.add_argument('-tau_scale', type = float, default = 0.5)


parser.add_argument('-output_mode', type = int, default = 3,
                    help = 'output_mode : how to print output. 0 - print to screen only, 1 - print to file only, 2 - print to both file and screen, 3 - no output (default)')

parser.add_argument('-scale', type = float, default = 1.0,
                    help = '-scale: scale factor for number of samples')

parser.add_argument('-init_tau', type = float, default = -1,
                    help = '-init_tau: initial tau value. Default: -1 (initially n)')
parser.add_argument('-iter_samples', type = int, default = -1,
                    help = '-iter_samples: number of samples to take in each iteration. Default: -1 (theoretical)')

def deleteIntermediateFiles(file, iters):
    for i in xrange(iters+1):
        try:
            os.remove(file+str(i))
        except:
            pass

def createRandomSeedList(link_server, fName, size=1):
    '''
    Creates a file containing a random set of seeds of the given size
    '''
    candidates = link_server.getNodes()

    seeds = sample(candidates, size)
    # delete any old seed files
    try:
        os.remove(fName)
    except:
        pass
    f = open(fName, 'w')
    for s in seeds:
        f.write(str(s)+'\n')

    return seeds

def loadSeedSet(fname):
    seed_set = []
    f = open(fname, 'r')
    for line in f.readlines():
        seed_set.append(int(line.strip()))
    return seed_set

def PrepareNextRound():
    for key in bucket.get_all_keys(prefix=tmp_dir_in_relative):
        key.delete()

    for key in bucket.get_all_keys(prefix=tmp_dir_out_relative):
        key.copy(bucket.name, tmp_dir_in_relative + key.name.split("/")[-1])
        key.delete()

def count(fname, threshold):
    print_out("counting exceeded runs, infected nodes, etc.",output_mode)

    job = Count(args=[fname]+["-c", "~/mrjob.conf",
                      "-r", "emr",
                      "--pool-emr-job-flows",
                      "--threshold", str(threshold)
                      ])
    with job.make_runner() as runner:
        runner.run()
        for line in runner.stream_output():
            _, tup = jvp.read(line.strip())
            print_out("received the following information from count: " + str(tup),output_mode)
    print_out("done",output_mode)
    return tup
def cleanup():
    print_out("cleaning up",output_mode)
    try:
        for key in bucket.list(prefix=tmp_dir_in_relative):
            key.delete()
        for key in bucket.list(prefix=tmp_dir_out_relative):
            key.delete()        
    except:
        print_out("Failed to cleanup S3 folders",output_mode)
        pass
    print_out("done",output_mode)

def print_all_tuples(dir):
    print "Printing contents of folder:"
    for key in bucket.get_all_keys(prefix=dir):
        print key.name
    print "done"
def bfs(database, seeds, threshold, nSamples=1, total_steps_cap = pow(10,30)):
    nDone = 0
    iter = 0
    #first run the initializer to get started
    cleanup()
    print_out("Number of samples: " + str(nSamples),output_mode)

    #seeds_path = "s3://joeloren/input/seeds" + str(nSeeds) + "/"
    mrJob = MRBFSSampleInit(args=[seeds]+["-c", "~/mrjob.conf",
                                               "-r", "emr",
                                               "--no-output",
                                               "--pool-emr-job-flows",
                                               "--samples", str(nSamples),
                                               "--output-dir", tmp_dir_in
                                               ])
    with mrJob.make_runner() as runner:
        runner.run()
    
    
    total_steps_exceeded = 0
    print_out("Finished initialization",output_mode)
    
    while nDone < nSamples and not total_steps_exceeded:
        iter_start_time = time.time()
        print_out('BFS iteration:' + str(iter),output_mode)

        mrJob2 = MRBFSSampleIter(args=[tmp_dir_in]+["-c", "~/mrjob.conf",
                                       "-r", "emr",
                                       "--pool-emr-job-flows",
                                       "--threshold", str(threshold),
                                       '--database', database,
                                       "--no-output",
                                       "--python-archive", "source.tar.gz",
                                       "--output-dir", tmp_dir_out
                                        ])
        
        with mrJob2.make_runner() as runner:
            runner.run()
        print_out("Done one BFS round",output_mode)
        PrepareNextRound()
        nExceeded, nDone, total_infected = count(tmp_dir_in,threshold)
        
        iter += 1
        print_out("Time per iteration: " + str(time.time() - iter_start_time),output_mode)

    print "Total infected, over all samples: ", total_infected, " cap: ", total_steps_cap
    if total_infected > total_steps_cap:
        return "exceeded", iter
    else:
        return (1. * nExceeded / nSamples, iter)

def print_out(text, output_mode = 0):
    if output_mode in [0,2]:
        print text
    if output_mode in [1,2]:
        f_output = open(output_file,'a')
        f_output.write(text+'\n')
        f_output.close()


def EstimateInfluence(link_server, bfs_method, eps, seeds_set, res_fname, \
                      cores, output_mode = 1, scale = 1.0, output_results = False,\
                      init_tau = -1, iter_samples = -1):

    start_time = time.time()
    mr_rounds_total = 0
    
    if bfs_method == 'emr':
        seeds_set = loadSeedSet(seeds)
    
    output_file = "output-nSeeds-%d-method-%s.txt"%(len(seeds_set), bfs_method)
    nNodes = link_server.getNumNodes()
    
    print_out("# nodes: "+str(nNodes) + ", # seeds: " + str(len(seeds_set)),output_mode)
    total_steps_cap = 4 * nNodes * pow(log(nNodes,2),2)
    
    tau = nNodes if init_tau <= 0 else init_tau
    done = False
    while not done and tau >= 1:
        print_out('Current tau value: '+str(tau),output_mode)
        T_vals = []
        T = tau
        p = []
        L = iter_samples
        while T <= nNodes and not done:
            print_out('current T value: ' + str(T),output_mode)
            T_vals.append(T)
            if iter_samples <= 0:
                L =  max(1,int(scale * (1. * T / tau) * eps * log(nNodes, 2) ))
                
            if bfs_method == 'emr':
                frac, mr_rounds = bfs(db_name, seeds, T, L, total_steps_cap)
                mr_rounds_total += mr_rounds
            elif bfs_method == 'seq':
                if not type(seeds_set) is list or len(seeds_set) == 0:
                    print "ERROR!! Problem with seeds set: ", seed
                    raise Exception
                frac, max_distance = IC_Sample(link_server, seeds_set, T, L, total_steps_cap, cores)
                mr_rounds_total += max_distance * 3
            else:
                raise Exception("No valid bfs method was given")
            p.append(frac)
            T *= (1+eps)
            print_out("T_vals = "+str(T_vals),output_mode)
            print_out("p = "+str(p),output_mode)
            if not frac == 'exceeded':
                integral_estimate = sum([T_vals[i] * p[i] for i in xrange(len(p))])
            if frac == "exceeded" or integral_estimate >= tau * (1-2*eps):
                print_out("Summary:",output_mode)
                
                print_out("Estimate of expected spread: " + str(tau),output_mode)
                print_out("BFS method: %s"%bfs_method,output_mode)
                if bfs_method == "emr":
                    print_out("Number of MR Rounds: "+str(mr_rounds_total),output_mode)
                print_out("Elapsed time: "+str(time.time() - start_time),output_mode)
                print_out("Number of nodes: " + str(link_server.n),output_mode)
                if edge_prob_type == 0:
                    print_out("Fixed probability: " + str(probs),output_mode)
                print_out("-------------",output_mode)
                done = True
        if not done:
            tau /= (1+eps)
    stop_time = time.time()
    if output_results:
        f = open(res_fname, 'a')
        f.write(str(tau) + '\n')
        f.write(str(len(seeds_set)) + '\n')
        f.write(str(2 * mr_rounds_total) + '\n')
        f.close()
    if bfs_method == 'emr':
        os.system("python -m mrjob.tools.emr.terminate_idle_job_flows -q")
    return tau

def main():
    global output_file
    global total_steps_cap
    global csv_file
    global generate_database
    global edge_prob_type
    global sql_mode
    global seeds_file
    global probs
    
    parameters = parser.parse_args()
    csv_file, db_name, generate_database, edge_prob_type, sql_mode, res_fname, seeds, bfs_method, cores, output_mode, scale = parameters.csv, parameters.dataset, \
        parameters.generate_database, parameters.edge_prob_type, parameters.sql, parameters.res_fname, parameters.seeds, parameters.bfs_method, parameters.cores, parameters.output_mode, parameters.scale

    if bfs_method == 'emr':
        print_out("setting up logging",output_mode)
        log_to_stream()
        print_out("done",output_mode)
    print_out('Evaluation algorithm. Dataset: %s'%(db_name),output_mode)
    link_server = LinkServerCP(db_name, undirected=parameters.undirected)
    if bfs_method == 'seq':
        print_out('Loading seeds set',output_mode)
        seeds_sets = cp.load(open(seeds, 'r'))
        
        for i in xrange(len(seeds_sets)):
            EstimateInfluence(link_server, bfs_method, parameters.tau_scale, seeds_sets[i], res_fname,\
                               cores, output_mode,scale, parameters.output_results,\
                               init_tau = parameters.init_tau, iter_samples = parameters.iter_samples)
    else:
        EstimateInfluence(link_server, bfs_method, seeds_sets[i], res_fname, cores, output_mode)
if __name__=='__main__':
    res = main()
