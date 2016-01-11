import numpy as np
from scipy.stats import sem
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import rc
import matplotlib
import matplotlib.gridspec as gridspec
from matplotlib.mlab import griddata
delim_dict = {0 : '\t', 1 : ' ', 2 : ','}
import os
from matplotlib import rcParams, cm
def plot2d(x_data,y_data_lists, errs_lists, dataset_labels,
           xy_labels, title, fig_name, log_scale=False,
           log_axis = 1, location = "upper center",
           error_mode = 1, logscale = False,
           plot_legend = True, ylim_scaling_factors = None):

    matplotlib.rcParams['ps.useafm'] = True
    matplotlib.rcParams['pdf.use14corefonts'] = True
    matplotlib.rcParams['text.usetex'] = True

    ## rc('text', usetex=True)
    ## plt.rc('font', family='serif')
    ## params = {'legend.fontsize': 20,'legend.linewidth': 2,'figure.subplot.hspace' : 0}
    ## plt.rcParams.update(params)
	
    styles= [':o','--s', '-.v', '-+', '--*', ':D', '--p']
    styles= [':','--', '-.', '-', '--', ':', '--']

    plt.cla()
    plt.clf()
    if logscale:
        print "setting y_scale(log)"
        plt.yscale('log')
        
    if not type(x_data[0]) is list:
        print "only one list of x_values"
        x = [x_data for i in xrange(len(y_data_lists))]
        x_data = x
    if log_scale:
        if log_axis == 1:
            plt.yscale('log')
        else:
            plt.xscale('log')
    plt.xlim(0,max(x_data[0]))

    y_max = [max(y) for y in y_data_lists]
    y_min = [min(y) for y in y_data_lists]
    if not ylim_scaling_factors is None:
        plt.ylim(ylim_scaling_factors[0] * min(y_min), ylim_scaling_factors[1] * max(y_max))
    for i, y_data in enumerate(y_data_lists):
        print "plotting y_vector number ", i
        if error_mode == 0:
            plt.errorbar(x_data[i], y_data, yerr=errs_lists[i], fmt=styles[i%len(styles)], label=dataset_labels[i])
        if error_mode == 1:
            line_within_colours = [('#1B2ACC','#089FFF'), ('#3F7F4C', '#7EFF99'),
                                   ('#FF9848', '#FFA662'), ('#252319', '#66655E'),
                                   ('#800080','#CEC2E5')]
            styles2 = ['-','--','-.',':']
            plt.plot(x_data[i], y_data, linestyle='-',
                     color = line_within_colours[i%len(line_within_colours)][0],
                     label=dataset_labels[i])
            plt.fill_between(x_data[i], y_data - np.array(errs_lists[i]),
                             y_data+np.array(errs_lists[i]), alpha=.5,
                             edgecolor=line_within_colours[i%len(line_within_colours)][0],
                             facecolor=line_within_colours[i%len(line_within_colours)][1], linewidth=0)
    if len(y_data_lists) > 1:
        legend = plt.legend(loc=location,
                    #bbox_to_anchor=(0.5, 1.10),
                    fancybox=False, shadow=False, ncol=3,
                    prop={'size':8})
        #plt.setp(legend.get_title(),fontsize='xx-small')
    print "Axes labels: ", xy_labels
    plt.tight_layout()
    plt.xlabel(xy_labels[0], fontsize=12)
    plt.ylabel(xy_labels[1], fontsize=12)
    plt.title(title)
    #######
    plt.ioff()
    plt.savefig(fig_name, format = 'pdf', bbox_inches='tight')
    plt.close('All')

def plot3dWireFrame(x_data,y_data,z_data, xyz_labels, title, fig_fname):

    matplotlib.rcParams['ps.useafm'] = True
    matplotlib.rcParams['pdf.use14corefonts'] = True
    matplotlib.rcParams['text.usetex'] = True

    rcParams["font.size"] = 11
    fig = plt.figure()

    
    #ax = fig.add_subplot(111, projection='3d')        
    
    ## Thibauth's code
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_wireframe(x_data, y_data, z_data, linewidth=0.1)
    plt.ticklabel_format(style='sci', axis='z', scilimits=(0, 0))
    
    
    
    ax.invert_xaxis()
    ##

    ax.set_xlabel(xyz_labels[0])
    ax.set_ylabel(xyz_labels[1])
    ax.set_zlabel(xyz_labels[2])

    ax.set_title(title)
    
    plt.savefig(fig_fname, format = 'pdf', bbox_inches='tight')

def plotHistogram(values, width = 10, fig_name="hist.pdf"):
    matplotlib.rcParams['ps.useafm'] = True
    matplotlib.rcParams['pdf.use14corefonts'] = True
    matplotlib.rcParams['text.usetex'] = True

    import pylab as P
    P.figure()
    # the histogram of the data with histtype='step'
    n, bins, patches = P.hist(values, width, normed=1, histtype='stepfilled')
    P.setp(patches, 'facecolor', 'g', 'alpha', 0.75)
    P.savefig(fig_name, format = 'pdf', bbox_inches='tight')
    P.close('All')
    
def removeFile(fname):
    try:
        os.remove(fname)
    except OSError:
        pass

def drange(start, stop, step):
    r = start
    while r <= stop:
        yield r
        r += step
def getNumCycles(fname):
    f = open(fname, 'r')
    for line in f.readlines():
        words = line.split(',')
        if 'cycles\n' in words:
            f.close()
            return int(words[0])


def plotBar(index, means_lists, errors_lists,
            labels, axes_labels, fig_fname,
            y_log = False, plot_legend = True):

    plt.cla()
    plt.clf()

    matplotlib.rcParams['ps.useafm'] = True
    matplotlib.rcParams['pdf.use14corefonts'] = True
    matplotlib.rcParams['text.usetex'] = True

    fig, ax = plt.subplots()
    if y_log:
        ax.set_yscale('log')

        
    bar_width = 0.25
    index = np.array(index)
    opacity = 0.4
    error_config = {'ecolor': '0.3'}
    colors = ['r','b','g','o']
    for i in xrange(len(means_lists)):
        rects1 = plt.bar(index + bar_width * i, means_lists[i], bar_width,
                         alpha=opacity,
                         color=colors[i],
                         yerr=errors_lists[i],
                         error_kw=error_config, label = labels[i])
    

    plt.xlabel(axes_labels[0])
    plt.ylabel(axes_labels[1])
    plt.xticks(index + bar_width, index)
    if plot_legend:
        ax.legend(loc='upper center',
                bbox_to_anchor=(0.5, 1.12),
                fancybox=False, shadow=False, ncol=5,
                prop={'size':8})
        

    plt.tight_layout()
    plt.savefig(fig_fname, format = 'pdf', bbox_inches='tight')

if __name__ == "__main__":
    n_groups = 5

    means_men = (20, 35, 30, 35, 27)
    std_men = (2, 3, 4, 1, 2)

    means_women = (25, 32, 34, 20, 25)
    std_women = (3, 5, 2, 3, 3)

    fig, ax = plt.subplots()

    index = np.arange(n_groups)
    plotBar(index, [means_men,means_women], [std_men, std_women], ['men','women'], ['Group', 'Scores'], 'test.pdf')
