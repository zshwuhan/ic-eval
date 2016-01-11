from common_tools import plot2d
from collections import defaultdict
import numpy as np
from scipy.stats import sem
def loadDataFromFile(fname):
    '''
    Loads results from a file containg results for running times:
    line 1: number of seeds
    line 2: number of cycles by approximate evaluation algorithm
    line 3: number of cycles by naive sampling algorithm
    '''
    f = open(fname, 'r')
    bDone = False
    dCycles = defaultdict(list)
    nSeeds = f.readline().strip()
    while not nSeeds == '':
        nSeeds = int(nSeeds)
        cycles_approx = float(f.readline().strip())
        cycles_exact = float(f.readline().strip())
        dCycles[nSeeds].append((cycles_approx,cycles_exact))
        nSeeds = f.readline().strip()
    return dCycles

def calculateRuntimeRatio(dCycles):
    '''
    Returns three lists:
    1. Size of seed sets.
    2. Average Ratio of numbers of cycles: cycles_exact / cycles_approximate
    3. Standard errors
    '''
    nSeeds_list = dCycles.keys()
    nSeeds_list.sort()
    avg_ratio = np.array([sum([tup[1]/tup[0] for tup in dCycles[k]])/len(dCycles[k]) for k in nSeeds_list])
    errs_ratio = 2 * np.array([sem([tup[1]/tup[0] for tup in dCycles[k]]) for k in nSeeds_list])
    print "Average ratios:", avg_ratio
    print "Standard errors:x", errs_ratio
    
    return np.array(nSeeds_list), avg_ratio, errs_ratio

def plotRatiosFromFile(fnames,eps_list, dataset_name, fig_name, n=1):
    ratios_arrays = []
    errors_arrays = []
    eps_values = [r'$\epsilon= $' + str(eps) for eps in eps_list]
    for fname in fnames:
        dCycles = loadDataFromFile(fname)
        nSeeds_array, ratios, errors = calculateRuntimeRatio(dCycles)
        print "nSeeds_array: ",nSeeds_array
        print "ratios: ", ratios
        print "errors: ", errors
        ratios_arrays.append(ratios)
        errors_arrays.append(errors)
    plot2d([1. * nSeeds / n for nSeeds in nSeeds_array],ratios_arrays, errors_arrays, eps_values, [r'$\frac{k}{n}$','Ratio'], dataset_name, fig_name)

if __name__ == "__main__":
    plotRatiosFromFile('experiments/wiki/running_times_wiki','WikiVote', 'wiki_running_time_ratios.pdf')
