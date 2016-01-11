import os
if __name__=='__main__':
    min_nSeeds = 3000
    increment = 1000
    #max_nSeeds = 6100
    max_nSeeds = 3001

    for nSeeds in xrange(min_nSeeds,max_nSeeds, increment):
        os.system("python ic_bfs_eval.py -dataset google+ -res_fname experiments/google+/results%d -seeds experiments/google+/seeds/google-seeds-%d.cp"%(nSeeds, nSeeds))
        #os.system("python -m mrjob.tools.emr.terminate_idle_job_flows -q")
