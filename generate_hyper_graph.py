# A wrapper for the map-reduce step. Instantiates a bunch of BFS sample runs and runs them all until convergence
import time
import argparse
from ic_bfs_eval_init import MRBFSSampleInit
from mr_ic_bfs import MRBFSSampleIter
#import simplejson as json
from collections import defaultdict,namedtuple
import os
#from subprocess import call
from link_server import *
from math import log
from random import sample
import cProfile, pstats,StringIO
from mrjob.protocol import JSONProtocol, JSONValueProtocol
jvp = JSONValueProtocol()
jp = JSONProtocol()
#---------
debug_mode = True
output_mode = 1
output_file = 'output.txt'
interim_file = 'intermediateResults'
dataset_name = 'input/datasets/wiki'
csv_file = 'input/datasets/Wiki-Vote_stripped.txt'
db_name = 'networks_with_probs'
edge_prob_type = 2
probs = [0.1,0.01]
generate_database = True
sql_mode = False
total_steps_cap = 0
#-----------
parser = argparse.ArgumentParser()
parser.add_argument('-csv', type = str, #required,
                    default = 'input/datasets/Wiki-Vote_stripped.txt', help="Name of source csv file (if we generate a new Link Server" )
parser.add_argument('-dataset', type = str, default = 'input/datasets/wiki',
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

def main():
    global output_file
    global total_steps_cap
    global csv_file
    global generate_database
    global edge_prob_type
    global sql_mode
    global probs

    parameters = parser.parse_args()
    csv_file, db_name, generate_database, edge_prob_type, sql_mode, res_fname = parameters.csv, parameters.dataset, \
        parameters.generate_database, parameters.edge_prob_type, parameters.sql, parameters.res_fname
    
    if edge_prob_type == 0:
        probs = parameters.probs
    
    start_time = time.time()

    

if __name__ == "__main__":
    main()
