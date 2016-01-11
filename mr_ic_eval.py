#! /usr/bin/env python

import os
import re
 
from mrjob.job import MRJob
from mrjob.protocol import RawProtocol, ReprProtocol
from collections import namedtuple
from mrjob.protocol import JSONValueProtocol
from math import sqrt

#import cascadeLib as cl
import random
import sqlite3
dbName = 'networks.db'
tableName = 'p2p_graph'

debug_mode = True
#debug_mode = False
nDFS_samples = 1
dfs_cap_exponent = 0.5
nNodes = 0
edgeProb = 0.5

dfs_output = namedtuple('dfs_output','isCapped node isCompletelyProbed')
total_nodes_reached = 0 # global variable -- total number of nodes reached during iteration T
# a named tuple that will serve in the implementation of the MRSample algorithm (based on Karloff et al.)
mrsample_tuple = namedtuple('mrsample_tuple','node label current_in_neighbour total_in_neighbours active isReached')

# Various functions related to the instantiation and estimation of the independent cascade spread model.
def initLinkServer(databaseName, tableName):
    '''
    Generates an sqlite3 cursor to be used as the "link-server".
    '''
    conn = sqlite3.connect(databaseName)
    link_server = namedtuple('cursor','table n')
    link_server.cursor = conn.cursor()
    link_server.table = tableName

    nodes, adj_lists_str = link_server.cursor.execute("SELECT fromNode,neighbours from '%s'"%(link_server.table))
    total_nodes = set(nodes)
    for adj_list_str in adj_lists_str:
        total_nodes.union(set([int(v.strip()) for v in adj_list_str.split(',')]))

    link_server.n = len(total_nodes)
    print "A total of %d nodes in the graph"%link_server.n
    
    return link_server

def getOutNeighbourhood(link_server, fromNode):
    '''
    Returns the out-neighbourhood of fromNode)
    '''
    out_neighbours = [v[1] for v in link_server.cursor.execute("SELECT * FROM '%s' WHERE fromNode='%s'"%(link_server.table, str(fromNode)))]
    return out_neighbours

def getNnodes(link_server):
    
    


class icEvaluator(MRJob):
    #OUTPUT_PROTOCOL = JSONValueProtocol

    def configure_options(self):
        super(icEvaluator, self).configure_options()
        self.add_file_option('--table')
        
    def mapper_bounded_dfs_init(self):
        self.link_server = initLinkServer(dbName, self.options.table)
        nNodes = getNnodes(self.link_server)
        
    def mapper_bounded_dfs(self, key, seed_node):
        for iSample in xrange(nDFS_samples):  
            yield (iSample, ('seed',seed_node))
            yield (iSample, ('cap', int(sqrt(n))))

    def reducer_bounded_dfs_init(self):
        self.link_server = initLinkServer(dbName, self.options.table)
    
    def reducer_bounded_dfs(self, key, values):
        if debug_mode:
            print "Reducer for sample #: " + str(key)
   
        # cap of the dfs run
        cap = 0
        # Set of reached nodes
        seeds = []
        for tup in values:
            if tup[0] == 'seed':
                seeds.append(tup[1])
            else:
                cap = tup[1]
        # nodes reached and not yet probed (traverse their out-neighbourhoods.
        R = set([])

        Q = list(seeds)
        nReached = 0
        # do a bounded dfs run, starting for the seed nodes
        while len(Q)>0 and nReached < cap:
            # pop a vertex from the queue
            u = Q.pop()
            R.add(u)
            out_neighbourhood = getOutNeighbourhood(self.link_server, u)
            for v in out_neighbourhood:
                if not v in R and v not in Q:
                    random.seed()
                    t = random.uniform(0,1)
                    if t <= edgeProb:
                        Q.append(v)
                        nReached += 1
    
        # if DFS reached cap - return one tuple for each node reached thus far (to be completed in phase 2)
        # otherwise, just return the number of nodes reached.
        if nReached >= cap:
            if debug_mode:
                print "DFS reached cap. nReached = %d, cap = %d"%(nReached, cap)
            for u in R:
                yield (key, dfs_output(isCapped=True, node=u, isCompletelyProbed=True))#('capped', 'probed', u))
            for u in Q:
                yield (key, dfs_output(isCapped=True, node=u, isCompletelyProbed=False))#('capped', 'unprobed', u))
        else:
            # traversal ended prematurely -- no need to extended en subsequent round.
            if debug_mode:
                print "Sample %d, number of reached nodes = %d, cap=%d"%(key, nReached, cap)
            yield (key, dfs_output(isCapped=False, node=nReached, isCompletelyProbed=False))
                   
    # Impementation of the MRSample algorithm
    # ----------- Preparation: creating a tuple for every (node,sample_id) combination ------------------
    def mapper_phase2_init(self):
        self.link_server = initLinkServer(dbName, self.options.table)
        total_nodes_reached = 0

    def reducer_phase2_init(self):
        self.link_server = initLinkServer(dbName, self.options.table)
    
    def mapper_phase2_preparation(self, sample_id, node_tuple):
        '''
        receives a list of tuples of the form (key, node_tuple), where:
        sample_id = the DFS run id from phase 1.
        node_tuple = a tuple of the form (isCapped, node, isCompletelyProbed
        '''
        self.link_server = initLinkServer(dbName, self.options.table)
        # if received number of a dfs run that ended prematurely -- just add to total
        if not node_tuple[isCapped]:
            total_nodes_reached += node_tuple.node
        # otherwise -- the given node will participate in an extension (via MRSample)
        else:
            u = node_tuple.node
            nNeighbours = len(getInNeighbourhood(link_server,u))
            if node_tuple.isCompletelyProbed: # no additional probes available for this node
                yield (sample_id, mrsample_tuple(node=u, label=n+1, current_in_neighbour=nNeighbours,total_in_neighbours=nNeighbours, active=False, isProbed=True))
            else:
                yield (sample_id, mrsample_tuple(node=u, label=u, current_in_neighbour=0,total_in_neighbours=nNeighbours, active=True,isProbed=True))
    
    def reducer_phase2_preparation(self, sample_id, node_tuples):
        '''
        Receives as input the set of nodes tuples corresponding to nodes previously reached, with their current labels (n+1), and the current out-neighbour
        indices (either 0, or the size of their out-neighbourhoods.
        Creates an additional set of tuples, corresponding to nodes not yet reached. Returns the union. 
        '''
        reached_nodes = [node_tuple.node for node_tuple in node_tuples]
        for node_tuple in node_tuples:
            yield (sample_id, node_tuple)
         
        for u in xrange(n):
            if u not in reached_nodes:
                nNeighbours = len(getInNeighbourhood(link_server,u))
                yield (sample_id, mrsample_tuple(node=u,label=u, current_in_neighbour=0,total_in_neighbours=nNeighbours, active=True, isReached=False))
    
    #------------------------ begin main MRSample part-------------------------------
    # Each iteration of  the MRSample algorithm (lines 8-14) is broken down into the following map-reduce steps as follows:
    # Step 1. Line 8: for each key-tuple (sample_id, label=q), request the label of the node Gamma^-(q) for sample_id.
    # Step 2. For each key-tuple (sample_id, node=u) -- return the label of node u, for each label request.
    # Step 3. For each key-tuple (sample_id,label=q) -- update the labels of all nodes 
    def mapper_mrsample_1(sample_id, node_tuple):
        yield((sample_id, node_tuple.label), node_tuple)

    def reducer_mrsample_1(key, node_tuple):
        

    def steps(self):
        return [self.mr(mapper_init=self.mapper_bounded_dfs_init,mapper=self.mapper_bounded_dfs,reducer_init=self.reducer_bounded_dfs_init,reducer=self.reducer_bounded_dfs)]
            
#def main()

if __name__ == "__main__":
    tableName = ''
    icEvaluator.run()
