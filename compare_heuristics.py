from heuristics import *
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-csv', type = str, #required,
                    default = 'input/datasets/Wiki-Vote_stripped.txt', help="Name of source csv file (if we generate a new Link Server" )
parser.add_argument('-delim', type = int, default = 0,
                    help = '-delim : which delimiter (for edge-tuples) 0-tab, 1-space,2-comma. Default: 0')
## parser.add_argument('-samples', type = int, default = 5,
##                     help = '-samples : number of samples of given seed set size. Default: 10')
parser.add_argument('-cores', type = int, default = 50,
                    help = '-cores : number of cores of to use. Default: 50')
parser.add_argument('-undirected', type = int, default = 0,
                    help = '-undirected : is the graph undirected')
parser.add_argument('-output', type = str, default = 'out')
parser.add_argument('-dataset', type = str, default = 'input/datasets/wiki')
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
parser.add_argument('-tau_scale', type = float, default = 0.5,
                    help = '-tau_scale : scaling factor for tau. Default: 0.1')
