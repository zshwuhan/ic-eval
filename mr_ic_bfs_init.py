from mrjob.job import MRJob
#import simplejson as json
from mrjob.protocol import JSONValueProtocol, JSONProtocol
#jvp = JSONValueProtocol()
class MRBFSSampleInit(MRJob):
    #DEFAULT_PROTOCOL = 'json'

    OUTPUT_PROTOCOL = JSONProtocol

    def __init__(self, *args, **kwargs):
        super(MRBFSSampleInit, self).__init__(*args, **kwargs)
        #self.file_out = open(self.options.pathName + 'intermediateResults0','a')
        
        
    def configure_options(self):
        super(MRBFSSampleInit, self).configure_options()
        self.add_passthrough_option(
            '--samples', dest='nSamples', type='int', default=1,
            help='samples: number of instantiations of the independent cascade process.')

    def mapper(self, _, seed):
        seed = int(seed)
        for iSample in xrange(self.options.nSamples):
            yield 1, [seed, iSample]

    def reducer(self,_,seed_tuples):
        for tup in seed_tuples:
            yield (tup[0], [tup[1], False])
            
if __name__=='__main__':
    MRBFSSampleInit.run()
        
