#! /usr/bin/env python
from mrjob.job import MRJob
from collections import namedtuple
from collections import defaultdict
#from globals import *
from math import sqrt
from link_server import *
import random
from mrjob.protocol import JSONValueProtocol, JSONProtocol
jvp = JSONValueProtocol()
jp = JSONProtocol()
from boto.s3.key import Key
import boto
from boto.s3.connection import S3Connection
    
class MRBFSSampleIter(MRJob):
    INPUT_PROTOCOL = JSONProtocol
    OUTPUT_PROTOCOL = JSONProtocol

    
    def __init__(self, *args, **kwargs):
        super(MRBFSSampleIter, self).__init__(*args, **kwargs)
        if self.options.server_mode.lower() == 'sql':
            self.link_server = LinkServerSQL(self.options.db_name, False)
        else:
            c = S3Connection('AKIAI4OZ3HY56BTOHA3A',
                 '6isbkZjBM8kt3PIk53EXVIf76VOPxOH8rNleGc6B')

            datasets_bucket = c.get_bucket('joel_datasets')

            k = Key(datasets_bucket)
            k.key = self.options.db_name
            k.get_contents_to_filename(self.options.db_name + '.cp')
            self.link_server = LinkServerCP(self.options.db_name, False)
        
    def configure_options(self):
        super(MRBFSSampleIter, self).configure_options()

        self.add_passthrough_option(
            '--database', dest='db_name', default='networks.db', type='str',
            help='database: file name of the sqlite3 database')
        self.add_passthrough_option(
            '--threshold', dest='threshold', type='int', default=0,
            help='Threshold at which probing should stop.')
        self.add_passthrough_option(
            '--link_server_mode', dest='server_mode', type='str', default='cp',
            help='Should be either sql or cp.')
        
    def mapper_get_node(self, node, val):
         #_, x = jvp.read(val)
        yield node, val

    def reducer_probe(self, node_id, values):
        '''
        Receives a set of tuples corresponding to a node node_id.
        Each tuple is of the form: (sample_id (int), Neighbourhood_exhausted (Boolean))
        Neighbourhood_exhausted denotes whether or not the neighbourhood of that node was already fully probed (i.e.,
        no need to try to infect neighbours).
        '''
        
        node_tuples = [tup for tup in values]
        previously_probed = defaultdict(bool)
        # if this is the reducer for completed samples -- nothing to do
        if node_id == 'd':
            for val in node_tuples:
                yield val[0], ('d', val[1])
            
        else:
            #check for which samples node_id was already probed
            for tup in node_tuples:
                previously_probed[tup[0]] = (previously_probed[tup[0]] or tup[1])
            neighbours = self.link_server.getOutNeighbourhood(node_id)
            try:
                relevant_samples = [tup[0] for tup in node_tuples]
                results = [[node_id, sample, True] for sample in relevant_samples]
            
                for res in results:
                    yield res[1], (res[0],res[2])
            except:
                raise Exception("Failed to yield current node's tuples")
            try:
                for neighbour in neighbours:
                    for sample in relevant_samples:
                        if not previously_probed[sample]:
                            r = random.random()
                            if r <= neighbour[1]:
                                yield sample, (neighbour[0], False)
            except:
                raise Exception("Failed to return out-neighbours tuples")
            
    def reducer_summarize_sample(self, sample, tuples):
        
        tups = [x for x in tuples]

        if tups[0][0] == 'd':
            yield 'd', (sample, tups[0][1])
        
        reached_nodes = set([])
        already_probed = defaultdict(bool)
        for tup in tups:
            already_probed[tup[0]] = already_probed[tup[0]] or tup[1]
            reached_nodes.add(tup[0])
        # if number of reached nodes exceeds the threshold, or all nodes probed already -- done.
        if len(reached_nodes) >= self.options.threshold or all([already_probed[node] for node in already_probed.keys()]):
            yield 'd', (sample, len(already_probed.keys()))
        else:
            for node in already_probed.keys():
                yield node, (sample, already_probed[node])

    def steps(self):
        return [self.mr(reducer=self.reducer_probe),
                self.mr(reducer=self.reducer_summarize_sample)]

if __name__ == "__main__":
    MRBFSSampleIter.run()
