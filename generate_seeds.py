from random import sample
import cPickle as cp
import argparse
import os
from common_tools import delim_dict
parser = argparse.ArgumentParser()
parser.add_argument('-csv', type = str, #required,
                    default = 'input/datasets/Wiki-Vote_stripped.txt', help="Name of source csv file (if we generate a new Link Server" )
parser.add_argument('-delim', type = int, default = 0,
                    help = '-delim : which delimiter (for edge-tuples) 0-tab, 1-space,2-comma. Default: 0')
parser.add_argument('-min', type = int, default = 1000,
                    help = '-min : minimum number of seed nodes per set. Default: 1000')
parser.add_argument('-max', type = int, default = 1001,
                    help = '-max : maximum number of seed nodes per set. Default: 1001')
parser.add_argument('-interval', type = int, default = 1000,
                    help = '-interval : size of interval. Default: 1000')
parser.add_argument('-samples', type = int, default = 1,
                    help = '-samples : number of samples of given seed set size. Default: 1')
parser.add_argument('-output', type = str, default = 'experiments/wiki/seeds/wiki')
def LoadNodesFromFile(fname, delim = " ", header_line = False):
    f = open(fname, 'r')
    if header_line == True:
        f.readline()
    nodes_list = []
    for line in f.readlines():
        u, v = line.split(delim)
        u = int(u.strip())
        v = int(v.strip())
        nodes_list += [u, v]
    return set(nodes_list)

def SampleSeeds(V, k):
    return sample(V, k)

def CreateSeedsFile(fname_output, k, samples, V):
    seed_sets = []
   
    for i in xrange(samples):
        seed_sets.append(SampleSeeds(V, k))
    f = open(fname_output, 'w')
    cp.dump(seed_sets, f)
    f.close()

def generateSeedFiles(min_nSeeds, max_nSeeds, interval, V, nSamples, output_fname_prefix):

    for nSeeds in xrange(min_nSeeds, max_nSeeds, interval):        
        try:
            os.remove(output_fname_prefix+"%d.cp"%nSeeds)
        except OSError:
            pass
        CreateSeedsFile(output_fname_prefix+"%d.cp"%nSeeds, nSeeds, nSamples, V)
    
if __name__ == "__main__":
    parameters = parser.parse_args()
    min_nSeeds, max_nSeeds, interval, nSamples, fname_csv_edges, delim, output_fname_prefix = parameters.min, \
      parameters.max, parameters.interval, parameters.samples, parameters.csv, parameters.delim, parameters.output

    
      
    print "loading csv file"
    V = LoadNodesFromFile(fname_csv_edges, delim_dict[delim])
    generateSeedFiles(min_nSeeds, max_nSeeds, interval, V, nSamples, output_fname_prefix)
    
    
        
    
