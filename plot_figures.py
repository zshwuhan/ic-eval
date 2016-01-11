#!/usr/bin/env python
from common_tools import *
from graph_tools import *
from link_server import LinkServerCP
from test_k_vs_approx import calculateMeanErrorAndSEM, loadDataFromFiles
import os
from collections import defaultdict
import numpy as np
from scipy.stats import sem
def computeError(value, estimate):
    estimate = max(1, estimate)
    x = 1. * estimate / value
    try:
        if x < 1:
            return 1./x
        else:
            return x
    except:
        print "estimate: %.3f, value: %.3f"%(estimate, value)
        raise Exception("")

def plot_k_vs_approx():
    csv_files = [#'input/datasets/wiki-Vote-small.txt',\
                 #'input/datasets/email-Enron.txt',\
                 'input/datasets/Epinions1.txt',\
                 'input/datasets/Wiki-Vote_stripped.txt']
    work_dir = 'experiments/results/'
    delims = ['\t', '\t']
    n_values = [LinkServerCP('tmp', csv_fname, create_new = True, delim = delims[i]).getNumNodes() for i, csv_fname in enumerate(csv_files)]
    datasets = ['epinions','wiki-vote']
    x_vals_lists, y_vals_lists, sem_lists = [], [], []
    for i, dataset in enumerate(datasets):
        print "dataset: ", dataset
        n = n_values[i]
        min_frac,max_frac,step_frac = 0.01,0.6,0.02
        k_min, k_max, step = int(n * min_frac), int(n * max_frac)+1, int(n * step_frac)
        approx_files = [file for file in os.listdir(work_dir) if file.startswith(dataset + "-approx_errors-k_min-%d-k_max-%d"%(k_min,k_max))]
        seq_files = [file for file in os.listdir(work_dir) if file.startswith(dataset + "-seq-approx-errors-k")]
       
        x_vals,y_data, errs = [], [], []
        for k in range(k_min, k_max, step):
            print "k = ", k
            offsets = range(-30,30)
            candidate_fnames = [dataset + "-approx_errors-k_min-%d-k_max-%d-k-%d-samples-10"%(k_min,k_max,k+offset) for offset in offsets]
            files_exist = [candidate in approx_files for candidate in candidate_fnames]
            assert any(files_exist), "candidate fnames: "+str(candidate_fnames)
            offset_idx = files_exist.index(True)
            approx_fname = candidate_fnames[offset_idx]
            seq_fname = dataset + "-seq-approx-errors-k-" + str(k + offsets[offset_idx]) + "-samples-10"
            print "approx_fname: ", approx_fname
            print "seq_fname: ", seq_fname
            k, mean_err, err = calculateMeanErrorAndSEM(work_dir + approx_fname,work_dir + seq_fname)
            x_vals.append(1.0 * k / n)
            y_data.append(mean_err)
            errs.append(err)
        x_vals_lists.append(x_vals)
        y_vals_lists.append(y_data)
        sem_lists.append(errs)
        assert len(x_vals) == len(y_data)
        assert len(x_vals) == len(errs)
    plot2d(x_vals_lists,y_vals_lists, sem_lists, datasets, [r'$k/n$','Approximation ratio'], '' , 'experiments/figures/k_vs_approx_combined.pdf')

    print "Printing separate plots: "
    for i, dataset in enumerate(datasets):
        x_data = x_vals_lists[i]
        y_data = y_vals_lists[i]
        sem_list = sem_lists[i]
        plot2d(x_data, [y_data], [sem_list], [dataset], [r'$k/n$','Approximation ratio'], \
               '', 'experiments/figures/' + dataset + '-k_vs_approx.pdf')

def plot_k_vs_approx_multiple_epsilons():
    work_dir = 'experiments/results/k_vs_approx_ratio/'
    figs_dir = 'experiments/figures/'
    results_files = [#'wiki-approximations-nSamples-5-k_frac-0.010-0.600',
                     'wiki_improved2-approximations-nSamples-5-k_frac-0.010-0.600',
                     'epinions2-approximations-nSamples-5-k_frac-0.010-0.600']
    delims = ['\t', '\t']
    datasets = ['wiki2', 'epinions2']
    
    print "loading the data"
    
    print "Printing separate plots: "
    for i, dataset in enumerate(datasets):
        x_vals, y_vals_lists, sem_lists = set(), defaultdict(list), defaultdict(list)
        with open(work_dir + results_files[i], 'rb') as f:
            for line in csv.reader(f, delimiter='\t', quotechar='|'):
                k_frac, epsilon, approx_ratio, sem_val = [float(x) for x in line]
                x_vals.add(k_frac)
                y_vals_lists[epsilon].append(approx_ratio)
                sem_lists[epsilon].append(sem_val)
        y_vals_lists.pop(0.500)
        epsilon_values = sorted(y_vals_lists.keys())
        plot2d(sorted(list(x_vals)), [y_vals_lists[eps] for eps in epsilon_values],
               [sem_lists[eps] for eps in epsilon_values],
               [r'$\epsilon = %.2f$'%eps for eps in epsilon_values],
               [r'$k/n$','Approximation ratio'], \
               '', 'experiments/figures/' + dataset + '-k_vs_approx-multiple-epsilons-no-eps-0.5.pdf',
               location = 'upper center', ylim_scaling_factors = (0.9, 1.2))

        
def plot_runtime_ratios_small():
    nSamples, min_frac,max_frac = 10, 0.01, 0.6
    datasets = ['wiki-vote','slashdot','epinions']
    dataset_names = ['Wiki-vote', 'Slashdot','Epinions']
    file_pattern = 'running_times-ratios_k_min-%.3f-k_max-%.3f-samples-%d'%(min_frac,max_frac,nSamples)
    figure_pdf = 'experiments/figures/' + file_pattern + '.pdf'
    x_vals_lists, y_vals_lists, sem_lists = [], [], []
    for dataset in datasets:
        results_file = 'experiments/results/' + dataset + "-" + file_pattern
        print "Loading the file: ", results_file
        f = open(results_file, 'r')
        k_fracs, mean_ratios, stde = zip(*[[float(x) for x in line.strip().split('\t')] for line in f.readlines()])
        x_vals_lists.append(list(k_fracs))
        y_vals_lists.append(list(mean_ratios))
        sem_lists.append(list(stde))
    plot2d(x_vals_lists,y_vals_lists, sem_lists, dataset_names, [r'$k/n$','Running time ratio'], '' , figure_pdf, location="upper center")

def collect_runtimes(fname):
    f = open(fname, 'r')
    nCycles_approx_dict = defaultdict(list)
    nCycles_seq_dict = defaultdict(list)
    for line in f.readlines():
        vals = [float(x) for x in line.strip().split('\t')]
        nCycles_approx_dict[vals[0]].append(vals[1])
        nCycles_seq_dict[vals[0]].append(vals[2])
    k_fracs = nCycles_seq_dict.keys()
    k_fracs.sort()
    
    nCycles_approx = []
    nCycles_approx_errs = []
    nCycles_seq = []
    nCycles_seq_errs = []
    for k_frac in k_fracs:
        nCycles_approx.append(np.mean(np.array(nCycles_approx_dict[k_frac])))
        nCycles_seq.append(np.mean(np.array(nCycles_seq_dict[k_frac])))
        nCycles_approx_errs.append(sem(np.array(nCycles_approx_dict[k_frac])))
        nCycles_seq_errs.append(sem(np.array(nCycles_seq_dict[k_frac])))
    return k_fracs, nCycles_approx,nCycles_approx_errs,nCycles_seq, nCycles_seq_errs

def plot_3d_runtime_ratios():
    datasets = [#('BarabasiAlbert',2,r'Barab\'asi-Albert',3),
                #("Kronecker",3, "Kronecker", 3),
                ('SmallWorld', 4, 'Small World', 3),
                ('ConfigurationModel', 5, 'Configuration Model', 3)
                ]
    min_k, max_k,k_step, min_n,max_n, n_step, nSamples = 0.01, 0.1, 0.01, 1000, 5001, 1000, 5
    print "number of datasets: ", len(datasets)
    for dataset in datasets:
        print "Loading data for dataset: ", dataset[2]
        #file_names = ["experiments/results/n_vs_k_runtime/%s-running_times-n-%d-graph_method-%d-samples-%d-eps-0.00000"
        #              %(dataset[0], n, dataset[1], nSamples)
        #              for n in xrange(min_n, max_n, n_step)]

        file_names = ["experiments/results/n_vs_k_runtime/%s-running_times-n-%d-graph_method-%d-samples-prob_method-%d-%d-eps-0.00000"%(dataset[0], n, dataset[1], nSamples, dataset[3])\
                      for n in xrange(min_n, max_n, n_step)]
        
        n_values = np.arange(min_n,max_n,n_step)
        k_values = np.arange(min_k,max_k+.00000001,k_step)
        x, y = np.meshgrid(k_values, n_values)

        ratios = np.zeros(x.shape)
        for i,n in enumerate(xrange(min_n, max_n, n_step)):
            data = [line.strip().split('\t') for line in open(file_names[i])]
            for tup in data:
                min_k_val, max_k_val, k_step_val = [n * t for t in [min_k,max_k,k_step]]
                ratios[i,int((int(tup[0]) - min_k_val)/k_step_val)] += 1.0 * float(tup[2])/float(tup[1])
        ratios = 1.0/nSamples * ratios
        print "Plotting graph for dataset: ", dataset[2]
        plot3dWireFrame(x[:,:-1], y[:,:-1],ratios[:,:-1], ['k/n','n','Ratio'], '', 'experiments/figures/running_times_ratios-%s-n-%d-%d-prob_method-%d.pdf'%(dataset[0], min_n,max_n, dataset[3]))
        #return x, y, ratios

def plot_runtime_large_separate():
    results_dir = "experiments/results/"
    figures_dir = "experiments/figures/"
    raw_data_fnames = ['wiki-vote-running_times-ratios_k_min-0.010-k_max-0.600-samples-5-bfs_samples-1000-large-raw',
                      'slashdot-running_times-ratios_k_min-0.010-k_max-0.600-samples-5-bfs_samples-1000-large-raw',
                      'youtube-running_times-ratios_k_min-0.010-k_max-0.600-samples-5-bfs_samples-1000-large-raw',
                      'epinions-running_times-ratios_k_min-0.010-k_max-0.600-samples-5-bfs_samples-1000-large-raw',]
    n_values = [7115, 82168, 1134890, 75879]
    figure_pdf_fnames = [figures_dir + source_name + '.pdf' for source_name in raw_data_fnames]
    titles = [r'\textbf{Wiki-Vote}', r'\textbf{Slashdot}', r'\textbf{Youtube}', r'\textbf{Epinions}']
    k_fracs_lists, nCycles_approx_lists, nCycles_approx_errs_lists, nCycles_seq_lists, nCycles_seq_errs_lists = [[] for i in range(5)]
    nCycles_seq_approx_lists = []
    for i, fname in enumerate(raw_data_fnames):
        k_fracs, nCycles_approx,nCycles_approx_errs,nCycles_seq, nCycles_seq_errs = collect_runtimes(results_dir + fname)
        k_fracs_lists.append(k_fracs)
        nCycles_seq_approx_lists.append(1./log(n_values[i], 2) * np.array(nCycles_seq))
        nCycles_approx_lists.append(nCycles_approx)
        nCycles_approx_errs_lists.append(nCycles_approx_errs)
        nCycles_seq_lists.append(nCycles_seq)
        nCycles_seq_errs_lists.append(nCycles_seq_errs)
        
    for i in xrange(len(k_fracs_lists)):
        print "k fracs: ", k_fracs_lists[i]
        print "nCycles_approx", nCycles_approx_lists[i]
        print "nCycles_seq", nCycles_seq_lists[i]
        
        plot2d(k_fracs_lists[i], [nCycles_approx_lists[i], nCycles_seq_lists[i], nCycles_seq_approx_lists[i]], [nCycles_approx_errs_lists[i],nCycles_seq_errs_lists[i],nCycles_seq_errs_lists[i]],\
               [r'\textsc{InfEst}', r'\textsc{MC}', r'\textsc{MC} --- Approx'], [r'$k/n$','Mean number of CPU cycles'], titles[i] , figure_pdf_fnames[i], location="upper left", log_axis = 1, logscale=True)
    
def plot_runtime_ratios_large():
    nBFS_samples, nSamples, min_frac,max_frac = 1000, 5, 0.01, 0.6
    datasets = ['youtube', 'slashdot', 'epinions']
    dataset_names = ['Youtube','Slashdot','Epinions']
    #datasets = ['wiki-small']
    #dataset_names = ['Wiki-small']
    file_pattern = 'running_times-ratios_k_min-%.3f-k_max-%.3f-samples-%d-bfs_samples-%d-large'%(min_frac,max_frac,nSamples,nBFS_samples)
    figure_pdf = 'experiments/figures/' + file_pattern + '-large-' + '-'.join(d for d in datasets) + '.pdf'
    x_vals_lists, y_vals_lists, sem_lists = [], [], []
    for dataset in datasets:
        results_file = 'experiments/results/' + dataset + "-" + file_pattern
        f = open(results_file, 'r')
        k_fracs, mean_ratios, stde = zip(*[[float(x) for x in line.strip().split('\t')] for line in f.readlines()])
        x_vals_lists.append(list(k_fracs))
        y_vals_lists.append(list(mean_ratios))
        sem_lists.append(list(stde))
    plot2d(x_vals_lists,y_vals_lists, sem_lists, dataset_names, [r'$k/n$','Running time ratio'], '' , figure_pdf, location="upper left")

def plot_runtimes():
    plot_runtime_ratios_large()
    datasets = ['wiki-vote', 'epinions','slashdot','youtube']
    results_dir = 'experiments/results/'
    figs_dir = 'experiments/figures/'
    min_frac, max_frac, nSamples, bfs_samples = 0.01, 0.6, 5, 1000
    raw_data_fname_suffix = '-running_times-ratios_k_min-%.3f-k_max-%.3f-samples-%d-bfs_samples-%d-large-raw'%(min_frac, max_frac, nSamples, bfs_samples)
    
    plot_runtime_ratios_large()
    for dataset in datasets:
        pdf_name = figs_dir + dataset + '-running_times-ratios_k_min-%.3f-k_max-%.3f-samples-%d-bfs_samples-%d-large.pdf'%(min_frac, max_frac, nSamples, bfs_samples)
        plot_runtime_large_separate(results_dir + dataset + raw_data_fname_suffix, pdf_name, '', bLog_scale = True)

def plot_approximations_equal_times():
    results_dir = 'experiments/results/approx-heuristic1/'
    figures_dir = 'experiments/figures/'
    file_names = \
      ['wikivote-approx-heuristic-prob_method-2-k_min-0.0010-k_max-0.0100-tau_scale-0.100-samples-5-bfs_samples-1000-init_samples-5-iter_samples-5',\
       'wikivote-approx-heuristic-prob_method-3-k_min-0.0010-k_max-0.0100-tau_scale-0.100-samples-5-bfs_samples-1000-init_samples-5-iter_samples-5',\
      'BA1000_dataset-approx-heuristic-prob_method-2-k_min-0.0010-k_max-0.0100-tau_scale-0.100-samples-5-bfs_samples-1000-init_samples-5-iter_samples-5',\
      'BA1000_dataset-approx-heuristic-prob_method-3-k_min-0.0010-k_max-0.0100-tau_scale-0.100-samples-5-bfs_samples-1000-init_samples-5-iter_samples-5',
      'gnp08-1000-approx-heuristic-prob_method-2-k_min-0.0010-k_max-0.0100-tau_scale-0.100-samples-10-bfs_samples-1000-init_samples-30-iter_samples-30',
      'gnp08-1000-approx-heuristic-prob_method-2-k_min-0.0010-k_max-0.0050-tau_scale-0.100-samples-10-bfs_samples-1000-init_samples-10-iter_samples-10']
    
    titles = ['']*len(file_names)
    approx_errors_apx_d = defaultdict(list)
    approx_errors_seq_d = defaultdict(list)
    for i,fname in enumerate(file_names):
        print "fname: ", fname
        lines = [line.strip().split('\t') for line in open(results_dir + fname, 'r').readlines()]
        for line in lines:
            k_frac, value, apx_value, vanilla_value = float(line[0]), float(line[1]), float(line[2]), float(line[3])
            approx_errors_apx_d[k_frac].append(computeError(value, apx_value))
            approx_errors_seq_d[k_frac].append(computeError(value, vanilla_value))
        k_fracs = approx_errors_apx_d.keys()
        k_fracs.sort()
        errors_apx = [np.mean(approx_errors_apx_d[k_frac]) for k_frac in k_fracs]
        print "errors for apx: ", errors_apx
        sems_apx = [sem(approx_errors_apx_d[k_frac]) for k_frac in k_fracs]
        errors_seq = [np.mean(approx_errors_seq_d[k_frac]) for k_frac in k_fracs]
        print "errors for vanilla: ", errors_seq
        sems_seq = [sem(approx_errors_seq_d[k_frac]) for alpha in k_fracs]
        plot2d(k_fracs,[errors_apx, errors_seq], [sems_apx,sems_seq], [r'INFEST^*', 'Capped MC'], [r'$k/n$','Estimation error'], titles[i] , figures_dir + fname+'.pdf', location="upper left")

def plotStdDeviations():
    results_dir = 'experiments/results/influence_std/'
    figures_dir = 'experiments/figures/'
    #files = ['wikivote-influence_concentration_samples_min-10-samples_max-1000-k-100.000-prob_method-2-nSeeds_sets-100',\
    #         'wikivote-influence_concentration_samples_min-10-samples_max-1000-k-10.000-prob_method-2-nSeeds_sets-100',\
    #         'wikivote-influence_concentration_samples_min-10-samples_max-1000-k-1.000-prob_method-2-nSeeds_sets-100']
    files = ['wiki-vote-influence_concentration_samples_min-10-samples_max-1001-k-1.000-prob_method-2-nSeeds_sets-10',
             'epinions-influence_concentration_samples_min-10-samples_max-200-k-1.000-prob_method-2-nSeeds_sets-10',
             'slashdot-influence_concentration_samples_min-10-samples_max-1001-k-1.000-prob_method-2-nSeeds_sets-10']
    #titles = [r"Wiki-vote, k=1, prob method --- $p_e \in_R \{ 0.1,0.01 \}, \forall e \in V$"]
    titles = ['','','']
    #r"Wiki-vote, k=1, prob method --- u.a.r"]#,     
    std_lists = defaultdict(list)
    for i, filename in enumerate(files):
        data = [x.strip().split('\t') for x in open(results_dir + filename, 'r').readlines()]
        for line in data:
            std_lists[int(line[0])].append(float(line[1]))
        l_nSamples = std_lists.keys()
        l_nSamples.sort()
        l_mean_stds = []
        l_mean_std_errors = []
        for nSamples in l_nSamples:
            l_mean_stds.append(np.mean(std_lists[nSamples]))
            l_mean_std_errors.append(sem(std_lists[nSamples]))
        plot2d(l_nSamples,[l_mean_stds], [l_mean_std_errors], [''], [r'\# of samples','Mean Relative std error'], titles[i] , figures_dir + filename+'.pdf', location="lower left")
        plot2d(l_nSamples,[l_mean_stds], [l_mean_std_errors], [''], [r'\# of samples','Mean Relative std error'], titles[i] , figures_dir + filename+ "-x_log_scale" +'.pdf', log_scale = True, log_axis = 0, location="lower left")

def plotSamplesVSInfluence():
    results_dir = 'experiments/results/influence_values/'
    figures_dir = 'experiments/figures/'
    files = ['wiki-vote-influence_values_samples_min-10-samples_max-5000-k-1.000-prob_method-1',\
             'wiki-vote-influence_values_samples_min-10-samples_max-5000-k-1.000-prob_method-2',\
             'wiki-vote-influence_values_samples_min-10-samples_max-5000-k-1.000-prob_method-3']
    #titles = [r"Wiki-vote, k=1, prob method --- u.a.r", \
     #         r"Wiki-vote, k=1, prob method --- $p_e \in_R \{ 0.1,0.01 \}, \forall e \in V$"]
    titles = [''] * len(files)
    
    for i, filename in enumerate(files):
        data = [x.strip().split('\t') for x in open(results_dir + filename, 'r').readlines()]
        samples_list = []
        influence_values = []
        influence_sems = []
        for line in data:
            values = [int(val) for val in line[2:]]
            samples_list.append(len(values))
            influence_values.append(np.mean(values))
            influence_sems.append(sem(values))

        plot2d(samples_list,[influence_values], [influence_sems], [''], [r'\# of samples','Mean influence value'], titles[i] , figures_dir + filename+'.pdf', location="lower left")
        plot2d(samples_list,[influence_values], [influence_sems], [''], [r'\# of samples','Mean influence value'], titles[i] , figures_dir + filename + "-x_log_scale"+'.pdf', log_scale = True, log_axis = 0, location="lower left")



def translate_label(label):
    if label.startswith('Vanilla'):
        return label.replace('Vanilla', r"\textsc{MC'}")

    ## INFEST label
    init_samples, iter_samples = label[7:-1].split(',')
    if init_samples == '-1':
        if iter_samples == '-1':
            return r'\textsc{INF}'
        else:
            return r"\textsc{INF'(%s)}"%iter_samples
    else:
        return r"\textsc{INF''(%s, %s)}"%(init_samples, iter_samples)
def filterList(l, indices):
    return [x for i,x in enumerate(l) if i in indices]
def plot_heuristics_small():
    results_dir = 'experiments/results/heuristics/'
    results_dir1 = 'experiments/results/'
    figures_dir = 'experiments/figures/'
    file_names = \
      [
      results_dir +'epinions-heuristics_comparison-min_k-0.000020-max_k-0.000220-nSamples-5-prob_method-1-tau_scale-0.250',
      results_dir +'WikiVote-heuristics_comparison-min_k-0.000200-max_k-0.002200-nSamples-5-prob_method-1-tau_scale-0.250',
      ]
    figures_file_names = \
      [
      figures_dir +'epinions-heuristics_comparison-min_k-0.000020-max_k-0.000220-nSamples-5-prob_method-1-tau_scale-0.250',
      figures_dir +'WikiVote-heuristics_comparison-min_k-0.000200-max_k-0.002200-nSamples-5-prob_method-1-tau_scale-0.250',
      ]
    n_values = [75879, 7115]
    titles = ['']*len(file_names)
    k_values = []
    for i, filename in enumerate(file_names):
        print "Plotting results for the file: ", filename
        print "number of nodes: ", n_values[i]
        lines = open(filename, 'r').readlines()
        infest_labels = [translate_label(label) for label in lines[1].split('\t')]
        vanilla_labels = []
        if lines[2].startswith("Vanilla"):
            vanilla_labels = [translate_label(label) for label in lines[2].strip().split('\t')]
            data_start = 3

        print "INFEST Labels: ", infest_labels
        print "Vanilla Labels: ", vanilla_labels
        
        data = [x.strip().split('\t') for x in lines[3:]]
        assert all([len(line) == 1 + 2 * (len(infest_labels) + len(vanilla_labels) + 1) for line in data])
        estimates_dicts = [defaultdict(list) for heuristic in infest_labels + vanilla_labels]
        running_times_dicts = [defaultdict(list) for heuristic in infest_labels + vanilla_labels]
        #print "data: ", data
        for line in data:
            k_frac = float(line[0])
            true_value = float(line[1])
            estimates = [int(val) for j, val in enumerate(line[3:]) if j%2 == 0]
            running_times = [int(val) for j, val in enumerate(line[3:]) if not j%2 == 0]
            for j, estimate in enumerate(estimates):
                estimates_dicts[j][k_frac].append(computeError(float(true_value), estimate))
            for j, runtime in enumerate(running_times):
                running_times_dicts[j][k_frac].append(runtime)
        estimate_errors = []
        estimate_error_sem = []
        k_fracs = []
        running_times_lists = []
        running_times_sems = []
        
        for estimates_dict in estimates_dicts:
            k_fracs = estimates_dict.keys()
            k_fracs.sort()
            errors = []
            sems = []
            for k_frac in k_fracs:
                    errors.append(np.mean(estimates_dict[k_frac]))
                    sems.append(sem(estimates_dict[k_frac]))
            estimate_errors.append(errors)
            estimate_error_sem.append(sems)

        for running_times_dict in running_times_dicts:
             runtimes = []
             sems = []
             for k_frac in k_fracs:
                     runtimes.append(np.mean(running_times_dict[k_frac]))
                     sems.append(sem(running_times_dict[k_frac]))
             running_times_lists.append(runtimes)
             running_times_sems.append(sems)

        labels = infest_labels + vanilla_labels
        
        k_values = [int(k_frac * n_values[i]) for k_frac in k_fracs]
        if i == 0:
            k_values[len(k_values) - 1] = 14
        plot_legend = False if i > 0 else True
        plot2d(k_values,estimate_errors, estimate_error_sem, labels, [r'\# of seeds','Approximation ratio'], '' , figures_file_names[i] + '.pdf', location="upper center",plot_legend = plot_legend)
        plot2d(k_values,estimate_errors, estimate_error_sem, labels, [r'\# of seeds','Approximation ratio'], '' , figures_file_names[i] + '_log_y_axis' +'.pdf', location="upper center",log_scale=True, log_axis = 1, plot_legend = plot_legend)
        print "k values: ", k_values
        print "approximation ratios:", estimate_errors
        plotBar(k_values, estimate_errors, estimate_error_sem, labels, [r'\# of seeds','Approximation ratio'], figures_file_names[i] + '_bars' + '.pdf', plot_legend = plot_legend)

        #running times
        plot2d(k_values,running_times_lists, running_times_sems, labels, [r'\# of seeds','Running time'], '' , figures_file_names[i] + '-running_times' + '.pdf', location="upper center",logscale = True, plot_legend = plot_legend)
        
        plotBar(k_values, running_times_lists, running_times_sems, labels, [r'\# of seeds','Running time'], figures_file_names[i] + '-running_times-bars' + '.pdf', plot_legend = plot_legend)
    

if __name__ == "__main__":
    #plot_runtime_ratios_small()
    #plot_runtime_large_separate()
    #plotSamplesVSInfluence()
    #plotStdDeviations()
    #plot_k_vs_approx()
    #plot_3d_runtime_ratios()
    plot_heuristics_small()
    #plot_k_vs_approx_multiple_epsilons()
