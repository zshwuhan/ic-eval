from mrjob.job import MRJob
from mrjob.protocol import JSONProtocol, JSONValueProtocol
import sys


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

    def mapper(self, key, val):
        '''
        Counts number of done bfs runs, number of runs that exceeded the threshold,
        and total number of infected nodes, across all samples
        '''
        nExceeded, done, total_infected = 0, 0, 0
        nExceeded = 1 if key == "d" and val[1] > self.options.threshold else 0
        done = 1 if key == "d" else 0
        total_infected = val[1] if key == "d" else 1

        yield None, (nExceeded, done, total_infected)

    def reducer(self, key, values):
        sys.stderr.write("COUNT REDUCER: reducing %s" % list(values))
        yield None, list(values)
if __name__ == "__main__":
    Count.run()
