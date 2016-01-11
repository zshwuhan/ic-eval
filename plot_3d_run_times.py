from test_running_time_as_function_of_size import *
from common_tools import *
if __name__ == "__main__":
    fnames=['experiments/gnprunning_times-n-%d-graph_method-0-samples-10-eps-0.00000'%n for n in range(1000,7000,1000)]
    data=loadAxesFromFiles(fnames,range(1000,7000,1000))
    plot3dWireFrame(data,['n',r'$\frac{k}{n}$','Mean # of cycles'],r'$G(n,0.01)$','experiments/figures/gnp.pdf')
    
