import numpy as np
from scipy.stats import sem
import matplotlib.pyplot as plt
from matplotlib import rc
import matplotlib
import matplotlib.gridspec as gridspec


def loadParallelResults(fName):
    f = open(fName, 'r')
    spreads = []
    rounds = []

    line = f.readline()
    i = 0
    while not line == '':
        i += 1
        spreads.append(int(line.strip()))
        line = f.readline()
        line = f.readline().strip()
        rounds.append(int(line))
        line = f.readline()

    return np.array(spreads), np.array(rounds)

def loadSeqResults(fName):
    f = open(fName, 'r')
    spreads = []
    samples = []

    line = f.readline()
    
    while not line == '':
        spreads.append(float(line.strip()))
        line = f.readline().strip()
        samples.append(float(line))
        line = f.readline()

    return np.array(spreads), np.array(samples)

def plot(x_data,y_data_lists, errs_lists, y_labels, xy_labels, title, fig_name):
    rc('text', usetex=True)
    plt.rc('font', family='serif')
    params = {'legend.fontsize': 20, 'legend.linewidth': 2, 'figure.subplot.hspace' : 0}
    plt.rcParams.update(params)

    styles= [':o','-x','--s', '-.v', '-+', '--*', ':D', '--p']

    fig, axs = plt.subplots(nrows=1, ncols=1)

    axs.set_ylabel(xy_labels[1])
    axs.set_xlabel(xy_labels[0])

    axs.set_title(title)
    axs.set_ylim(0, 3.0)
    axs.set_xlim(0,max(x_data)*1.2)
    l = []
    for i, y_data in enumerate(y_data_lists):
        style = styles[i]
        l.append(axs.errorbar(x_data, y_data, yerr = errs_lists[i], fmt = style))
    axs.legend(l, y_labels, 'lower center', numpoints = 1, ncol = 2, prop={'size':9})

    plt.savefig(fig_name, format = 'pdf', bbox_inches='tight')
    plt.tight_layout()
    plt.show()
    
if __name__ == "__main__":
    results_folder = "experiments/google+/"
    results_parallel_prefix = "results"
    results_seq_prefix = "results-seq-"

    k_values = range(1000,15001,1000)
    mean_apx_list = np.zeros(len(k_values))
    std_errors = np.zeros(len(k_values))
    
    for i, k in enumerate(k_values):
        print "k value: ", k
        spreads_mr, rounds_mr = loadParallelResults(results_folder + results_parallel_prefix + str(k))
        spreads_seq, samples_seq = loadSeqResults(results_folder + results_seq_prefix + str(k))
        print "length of spreads_mr: ", len(spreads_mr)
        print "length of spreads_seq: ", len(spreads_seq)
        apx_ratios = np.divide(1.0 * spreads_seq, spreads_mr)
        
        
        mean_apx_list[i] = np.mean(apx_ratios)
        std_errors[i] =  sem(apx_ratios)
        
    print "Mean approximations: ", mean_apx_list
    print "Standard errors: ", std_errors
    
    plot(k_values, [mean_apx_list], [std_errors], ['Google+'], ['Number of seed nodes', 'Approximation'], 'Approximation ratio of evalutation algorithm for the IC model', 'experiments/google+/google-approximation-plot.pdf')
