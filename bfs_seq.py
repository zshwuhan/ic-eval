#from joblib import Parallel, delayed
from multiprocessing import Pool, cpu_count
from collections import defaultdict
from math import log
import random
from scipy.stats import sem
import numpy as np
#global output_file
#from ic_bfs_eval import print_out

def loadSeedSet(fname):
    seed_set = []
    f = open(fname, 'r')
    for line in f.readlines():
        seed_set.append(int(line.strip()))
    return seed_set


def IC_Instance(seed_set, threshold, sample):
    R = defaultdict(bool)
    distances = defaultdict(int)
    Q = list(seed_set)
    for u in seed_set:
        distances[u] = 0
        R[u] = True
    nReached = len(seed_set)
    while len(Q)>0 and nReached <= threshold:
        # pop a vertex from the queue
        u = Q.pop()
        R[u]=True
        out_neighbourhood = l_server.getOutNeighbourhood(u)
        
        for v, p in out_neighbourhood:
            if not R[v]:
                random.seed()
                t = random.random()
                if t <= p:
                    Q.append(v)
                    R[v] = True
                    nReached += 1
                    distances[v] = distances[u] + 1
    
    max_distance = max(distances.values())
    return nReached, max_distance

def IC_Instance2(input):
    return IC_Instance(input[0],input[1],input[2])

def GenerateParam(input_list, times):
    for i in xrange(int(times)):
        yield input_list + [i]



def IC_Sample(L, seed_set, threshold, nSamples=1, total_steps_cap = pow(10,30), num_cores=-1):
    random.seed()
    global l_server
    l_server = L

    adjacency_lists = L.adj_lists
    
    if num_cores == -1:
        pool = Pool()
    else:
        pool = Pool(num_cores)
        

    #print "Sampling %d instances of the IC process"%nSamples

    results = pool.map(IC_Instance2, GenerateParam([seed_set, threshold], nSamples))
    pool.close()
    pool.join()

    if sum([x[0] for x in results]) > total_steps_cap:
        return "exceeded", 1
    
    nExceeded = sum([1 for x in results if x[0] >= threshold])
    max_rounds = max([x[1] for x in results])
    frac = 1. * nExceeded / nSamples

    if not type(frac) is float:
        print "ERROR! frac isnt a float"
        print "frac = ", frac
        raise ValueError()
    return frac, max_rounds


def sequential_estimation(L, seeds, max_samples_cap = 0, nCores = -1, \
                          bReturnValues = False, min_samples = 0, min_relative_standard_error = -1):
    #global output_file
    global l_server
    #g = open('report_bfs_seq.txt','w')
    #g.write('Seed set: %s, max_samples_cap = %d, nCores = %d'%(str(seeds), max_samples_cap, nCores))
    #g.close()
        #output_file= "
    l_server = L
    nSamples = 0
    n = l_server.getNumNodes()
    
    avg = 0.0000001
    samples_total = 0.0
    bDone = False
    if nCores == -1:
        nCores = int(cpu_count())
    
    rounds = 0

    pool = Pool()

    max_samples = max_samples_cap if max_samples_cap > 0 else int(n * log(n, 2))
    
    spread_values = []
    total_values = 0
    while not bDone:
        num_concurrent_process = min( max_samples - samples_total, nCores)
        #g = open('bfs_seq_report.txt', 'w')
        #g.write('running %d concurrent IC instances on seed set: %s\n'%(num_concurrent_process, str(seeds)))
        #g.close()
        ret_values = pool.map(IC_Instance2, GenerateParam([seeds, l_server.getNumNodes()], num_concurrent_process))
        results = [r[0] for r in ret_values]
        assert all([r <= n for r in results])
        assert len(results) == num_concurrent_process
        
        spread_values += results
        total_values += sum(results)
        samples_total += num_concurrent_process
        bDone = samples_total >= max_samples
        if min_relative_standard_error > 0:
            std_error = sem(spread_values)/np.mean(spread_values)
            if std_error <= min_relative_standard_error:
                if min_samples <= 0 or (samples_total >= min_samples):
                    bDone = True
        
    pool.close()
    pool.join()

    mean = 1. * total_values / samples_total
    if bReturnValues:
        return mean, samples_total, spread_values
    else:
        return mean, samples_total
