from mrjob.job import MRJob
from mrjob.protocol import JSONProtocol, JSONValueProtocol


class Count(MRJob):

    INPUT_PROTOCOL = JSONProtocol
    OUTPUT_PROTOCOL = JSONValueProtocol

    def __init__(self, *args, **kwargs):
        super(Count, self).__init__(*args, **kwargs)
    
    def configure_options(self):
        super(Count, self).configure_options()
        self.add_passthrough_option(
            '--threshold', dest='threshold', type='int', default=0,
            help='Threshold at which probing should stop.')
    
    def mapper(self, key,val):
        '''
        Counts number of done bfs runs, number of runs that exceeded the threshold,
        and total number of infected nodes, across all samples
        '''
        nExceeded = 1 if key == "d" and val[1] > self.options.threshold else 0
        done = 1 if key == "d" else 0
        total_infected = val[1] if key == "d" else 1

        yield 1, (nExceeded, done, total_infected)

    def reducer(self, key, values):
        tups = [v for v in values]
        nExceeded = sum([v[0] for v in tups])
        nDone = sum([v[1] for v in tups])
        total_infected = sum([v[2] for v in tups])
        yield 1, (nExceeded, nDone, total_infected)
        
if __name__ == "__main__":
    Count.run()
