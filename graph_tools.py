#!/usr/bin/env python
import sqlite3
import csv
from collections import defaultdict
from random import uniform, seed, sample
from itertools import chain
from apgl.graph import SparseGraph
from apgl.generator.BarabasiAlbertGenerator import BarabasiAlbertGenerator
from apgl.generator.SmallWorldGenerator import SmallWorldGenerator
from apgl.generator.KroneckerGenerator import KroneckerGenerator
from apgl.generator.ConfigModelGenerator import ConfigModelGenerator
from math import log 
from numpy import array
def  adjListToTuplesList(dE):
    return list(set(chain.from_iterable([(i,j) for j in dE[i]] for i in dE.keys())))
from common_tools import plotHistogram
import networkx as nx
def gnp_graph(n,p):
    '''
    Generates a directed G(n,p) (Erdos-Renyi) graph.
    Output: a list of edges (tuples)
    '''
    seed()
    E = []
    for i in xrange(n):
        for j in xrange(n):
            if not i == j and uniform(0,1) <= p:
                E.append((i,j))
    return E

def ring_lattice(n,k):
    dE = defaultdict(list)    
    for i in xrange(n):
        for j in xrange(n):
            if 0 < abs((j-i))%(n-k/2) <= k/2:
                dE[i].append(j)
    return dE

def rewireEdge(n, dE, edge):
    seed()
    bDone = False
    i, j = edge
    l = range(n)
    l.remove(i)
    l.remove(j)
    new_neighbour = 0
    while not bDone:
        r = sample(l, 1)[0]
        if not r in dE[i] and not i in dE[r]:
            bDone = True

    dE[i].remove(j)
    dE[i].append(r)

def WattsStrogatz(n,k,beta):
    seed()
    dE = ring_lattice(n,k)

    for i in dE.keys():
        neighbours = list(dE[i])
        for j in neighbours:
            if uniform(0,1) <= beta:
                rewireEdge(n,dE,tuple([i,j]))

    return adjListToTuplesList(dE)

def star_graph(nPeripherals):
    l = [(0,i) for i in xrange(1,nPeripherals+1)]
    return l
def path_graph(n):
    l = [(i,i+1) for i in xrange(n)]
    return l
def write_to_csv(L, fName, delim = "\t", strip_first_line = False):
    '''
    Receives a list of lists. Dumps results onto a csv file
    '''
    print "Writing csv file:", fName
    f = open(fName, 'w')

    for i, l in enumerate(L):
        if not strip_first_line or i>0:
            l_str = delim.join(str(e) for e in l)
            f.write('%s\n'%l_str)
    f.close()

def read_edges_from_csv(fName, delim="\t"):
    '''
    Receives the file name (edges given in separate lines, endpoints are comma delimited)
    Returns a list of tuples
    '''
    print "Reading content of the file: ", fName
    edges = []
    with open(fName, 'r') as csvfile:
        dataset_reader = csv.reader(csvfile, delimiter = delim)
        for row in dataset_reader:
            edges.append((int(row[0]), int(row[1])))
    return edges
def createAdjLists(edge_list):
    L = defaultdict(list)
    for i,j in edge_list:
        L[i].append(j)
    
    return [L[k] for k in L.keys()]

                  
def write_to_sql(db_name, table_name, edges):
    conn = sqlite3.connect(db_name)
    c = conn.cursor()
    c.execute("CREATE TABLE '%s' (fromNode INT, toNode INT)"%table_name)

    ## for edge in edges:
    ##     c.execute("INSERT INTO '%s' VALUES('%d', '%d')"%(table_name, edge[0], edge[1]))
    c.executemany("INSERT INTO '%s' VALUES (?,?)"%table_name, edges)
    conn.commit()
    
    conn.close()
def convertAdjListToEdgeList(adj_list):
    edge_list = []
    for i,neighbours in enumerate(adj_list):
        new_edges = [(i,j) for j in neighbours]
        edge_list += new_edges
    return edge_list

def BarabasiAlbertEdgeList(n):
    graph = SparseGraph(n)
    generator = BarabasiAlbertGenerator(10, 10)
    graph = generator.generate(graph)
    l, _ = graph.adjacencyList()
    
    return convertAdjListToEdgeList(l)

def KroneckerEdgeList(n):
    init = SparseGraph(4)
    init[0, 1] = 1
    init[0, 2] = 1
    init[0, 3] = 1
    for i in range(4):
        init[i, i] = 1
    k = int(log(n, 4)) + 1
    generator = KroneckerGenerator(init, k)
    graph = generator.generate()
    l, _ = graph.adjacencyList()
    return convertAdjListToEdgeList(l)

def SmallWorld(n):
    # slow
    graph = SparseGraph(n)
    generator = SmallWorldGenerator(0.3, 50)
    graph = generator.generate(graph)
    l, _ = graph.adjacencyList()
    return convertAdjListToEdgeList(l)

def ConfigurationModel(edges_list):
    deg_dict = defaultdict(int)
    
    
    for u,v in edges_list:
        deg_dict[v] += 1
    l = array(deg_dict.values())
    n = len(l)
    graph = SparseGraph(n)
    generator = ConfigModelGenerator(l)
    graph = generator.generate(graph)
    l, _ = graph.adjacencyList()
    return convertAdjListToEdgeList(l)

def plot_deg_distribution(edges_list, fname):
    deg_dict=defaultdict(int)
    print "first 10 tuples in edges list: ", edges_list[:10]

    for u,v in edges_list:
        deg_dict[u] += 1
    values = deg_dict.values()
    values.sort()
    plotHistogram(values, width = 50, fig_name=fname)
    
def createRandomGraph(fname="input/datasets/tmp.csv", method=2, n=1000,p=0.2, beta=0.3, k=200):
    '''
    Creates a random graph with n nodes, according to the graph generation method specified in the argument 'method'
    Saves the list of edges (tab-delimited) in a csv file (fname)
    '''
    if method == 0:
        E = gnp_graph(n,p)
    if method == 1:
        E = WattsStrogatz(n,k,beta)
    if method == 2:
        print "Creating a Barabasi-Albert graph with %d nodes"%n
        E = BarabasiAlbertEdgeList(n)
    if method == 3:
        print "Creating a Kronecker graph with %d nodes"%n
        E = KroneckerEdgeList(n)
    if method == 4:
        print "Creating a Small-World graph with %d nodes (using apgl)"%n
        E = SmallWorld(n)
    if method == 5:
        print "Creating a configuration model graph based on a Barabasi-Albert graph with %n nodes"
        BA_edges = BarabasiAlbertEdgeList(n)
        E = ConfigurationModel(BA_edges)
    write_to_csv(E, fname, delim = "\t", strip_first_line = False)

def getAverageDegree(csv_fname, delim = '\t', undirected = False):
    E = read_edges_from_csv(csv_fname, delim)
    nodes_indeg_dict = defaultdict(int)
    nodes_outdeg_dict = defaultdict(int)
    for u,v in E:
        nodes_indeg_dict[v] += 1
        nodes_outdeg_dict[u] += 1
    n = len(set(nodes_outdeg_dict.keys()).union(set(nodes_indeg_dict.keys())))

    avg_indeg = 1. / n * sum(nodes_indeg_dict.values())
    avg_outdeg = 1. / n * sum(nodes_outdeg_dict.values())
    if undirected:
        return n, avg_indeg
    else:
        print "directed graph"
        return n, avg_indeg, avg_outdeg
if __name__ == '__main__':
    n = 1000
    l = BarabasiAlbertEdgeList(n)
    plot_deg_distribution(l, 'BA1000-deg_dist.pdf')
    l =  ConfigurationModel(l)
    plot_deg_distribution(l, 'CM1000-deg_dist.pdf')
