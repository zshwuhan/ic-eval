import cPickle as cp
import os
import sqlite3
import csv
from random import sample,seed, random, uniform
from collections import defaultdict
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-csv', type = str, #required,
                    default = 'input/datasets/Wiki-Vote_stripped.txt', help="Name of source csv file (if we generate a new Link Server" )
parser.add_argument('-dataset', type = str, #required,
                    default = 'test')
parser.add_argument('-delim', type = int, default = 0,
                    help = '-delim : which delimiter (for edge-tuples) 0-tab, 1-space,2-comma. Default: 0')
parser.add_argument('-undirected', type = int, default = 0,
                    help = '-undirected : is the graph undirected')
parser.add_argument('-prob_method', type = int, default = 3)

def CreateAdjListWithProbs(source, prob_type=0, prob=0.5, delim=' ', undirected = False):
    adjacency_lists = defaultdict(list)
    total_nodes = set()
    in_degs = defaultdict(int)
    max_node_id = 0
    nEdges = 0
    seed()
    if prob_type == 3 and not type(prob) == list or (type(prob) == list and not len(prob) == 2):
        prob = (0,1)
    with open(source, 'r') as csvfile:
        dataset_reader = csv.reader(csvfile, delimiter = delim)
        for row in dataset_reader:
            i, j = int(row[0]), int(row[1])
            total_nodes.add(i)
            total_nodes.add(j)
            max_node_id = max([max_node_id, i, j])
            adjacency_lists[i].append(j)
            if undirected:
                adjacency_lists[j].append(i)
                in_degs[i] += 1
            in_degs[j] += 1
            nEdges += 1
    total_nodes = list(total_nodes)
    total_nodes.sort()
    node_to_index = {}
    for i,node in enumerate(total_nodes):
        node_to_index[node]=i
    probs_lists = defaultdict(list)
    for node in adjacency_lists.keys():
            if prob_type == 0:
                probs_lists[node_to_index[node]] = [(node_to_index[v], prob) for v in adjacency_lists[node]]
            if prob_type == 1:
                probs_lists[node_to_index[node]] = [(node_to_index[v], 1. / in_degs[int(v)]) for v in adjacency_lists[node]]
            if prob_type == 2:
                probs_lists[node_to_index[node]] = [(node_to_index[v], sample(prob, 1)[0]) for v in adjacency_lists[node]]
            if prob_type == 3:
                probs_lists[node_to_index[node]] = [(node_to_index[v], uniform(prob[0],prob[1])) for v in adjacency_lists[node]]
    n = len(total_nodes)
    del total_nodes
    del adjacency_lists
    return (probs_lists, n, nEdges)

class LinkServer:
    def getOutNeighbourhood(self, node):
        '''
        Returns the out-neighbourhood of a node, if the form of a list of tuples: (out_neighbour, infection_probability)
        '''
        raise NotImplementedError( "Should have implemented this" )
    def getNodes(self):
        '''
        Returns a list of the nodes with positive out-degrees.
        '''
        raise NotImplementedError( "Should have implemented this" )
    def getNumNodes(self):
        '''
        Returns the total number of *all* nodes in the graph.
        '''
        raise NotImplementedError( "Should have implemented this" )

class LinkServerSQL(LinkServer):
    def __init__(self, sql_fName, csv_fName='', create_new=False, prob_method=0, prob=0.5):
        sql_fName = sql_fName + '.db'
        if create_new:
            self.createDatabase(sql_fName, csv_fName, prob_method, prob)
        else:
            conn = sqlite3.connect(sql_fName)
            self.cursor = conn.cursor()
            # lazy assignment
            self.n = -1

    def createDatabase(self, sql_fName, csv_fName, prob_method, prob):
        try:
            os.remove(sql_fName)
        except:
            print("Error! Failed to remove the SQL database file: " + sql_fName)
            pass
        
        adj_lists, self.n, self.m = CreateAdjListWithProbs(csv_fName, prob_type=0, prob=0.5)
        # Dump data onto an SQL Database
        conn = sqlite3.connect(sql_fName)
        self.cursor = conn.cursor()

        self.cursor.execute("CREATE TABLE network (fromNode INT, neighbours TEXT, probs TEXT)")
        for node in adj_lists.keys():
            probs_list = [v[1] for v in adj_lists[node]]
            out_neighbours_list = [v[0] for v in adj_lists[node]]
            probs = ','.join(str(e) for e in probs_list)
            out_neighbourhood = ','.join(str(e) for e in out_neighbours_list)
            self.cursor.execute("INSERT INTO network VALUES ('%d','%s','%s')"%(int(node), out_neighbourhood, probs))
        conn.commit()

    def getOutNeighbourhood(self, node):
        query_res = self.cursor.execute("SELECT * FROM network WHERE fromNode='%s'"%(str(node))).fetchone()
        res = []
        if query_res is None:
            "no neighbours!"
        else:
            u, neighbours_str, probs_str = query_res
            neighbours = [int(v) for v in neighbours_str.split(',')]
            probs = [float(p.strip()) for p in probs_str.split(',')]
            res = zip(neighbours, probs)
        return res

    def getNodes(self):
        '''
        Returns a list of all nodes with positive out-degree (potential seeds)
        '''
        V = [v[0] for v in self.cursor.execute('SELECT fromNode FROM network')]
        return V

    def getNumNodes(self):
        if self.n == -1:
            V = self.getNodes()
            total_nodes = set(V)
            for node in V:
                out_neighbourhood = set([v[0] for v in self.getOutNeighbourhood(node)])
                total_nodes = total_nodes.union(out_neighbourhood)
            self.n = len(total_nodes)
        return self.n
            
class LinkServerCP(LinkServer):
    def __init__(self, cp_fName, csv_fName='', create_new=False, prob_method=0, prob=0.5, delim=' ', undirected = False):
        cp_fName = cp_fName + '.cp'
        # lazy assignment
        self.n = -1
        self.m = -1
        self.undirected = undirected
        if create_new:
            d_adj_lists, self.n, self.m = CreateAdjListWithProbs(csv_fName, prob_method, prob, delim, undirected)
            try:
                os.remove(cp_fName)
            except:
                pass
            
            self.adj_lists = [list()] * (1+max(d_adj_lists.keys()))
            for node in d_adj_lists.keys():
                self.adj_lists[node] = d_adj_lists[node]
            #self.adj_lists = d_adj_lists
            data_dict = {'adj_lists': self.adj_lists, 'n' : self.n, 'm' : self.m}
            fCP = open(cp_fName,'w')
            cp.dump(data_dict, fCP)
            fCP.close()
        else:
            fCP = open(cp_fName,'r')
            d = cp.load(fCP)
            self.adj_lists = d['adj_lists']
            self.n = d['n']
            self.m = d['m']
            fCP.close()
    
    def getOutNeighbourhood(self, node):
        try:
            return self.adj_lists[node]
        except:
            return [] # empty out-neighbourhood
            #print "Failed to locate neighbourhood of node: ", node
            #print "Length of adj_list:", len(self.adj_lists)
            #raise IndexError("")

    def getNodes(self):
        #return list(self.adj_lists.keys())
        return [node for node in xrange(len(self.adj_lists))]
    
    def getNumNodes(self):
        if self.n < 0:
            ## nodes = self.adj_lists.keys()
            ## for adj_list in self.adj_lists.values():
            ##     nodes = nodes + [v[0] for v in adj_list]
            
            ## self.n = len(set(nodes))
            self.n = len([1 for v in adj_lists if len(v) > 0])
        return self.n
    
    def getNumEdges(self):
        if self.m < 0:
            self.m = sum([len(adj_list) for adj_list in self.adj_lists])
        return self.m
        

if __name__ == "__main__":
    parameters = parser.parse_args()
    L=LinkServerCP(parameters.dataset, parameters.csv, create_new=True, prob_method=paraameters.prob_method, prob=[0.1,0.01], delim=parameters.delim, undirected = parameters.undirected)
    
