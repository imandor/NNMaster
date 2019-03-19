import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from matplotlib.colors import BoundaryNorm
from src.settings import save_as_pickle, load_pickle
import glob
import numpy as np
from scipy.interpolate import interp1d
from matplotlib.patches import Patch
from statsmodels.nonparametric.smoothers_lowess import lowess
from matplotlib import rc


def edit_axis(model_path_list,ax_label,color,ax=None,plot_type="ape"):
    ax = ax or plt.gca()
    ape_avg_list = []
    for i,path in enumerate(model_path_list):
        # Plot parameters
        network_output_list = load_position_network_output(path)
        all_samples = [a.metric_by_cvs for a in network_output_list]
        if plot_type =="ape":
            ape_avg_list_i = [np.average([b.ape_by_epoch[-1] for b in a.metric_by_cvs]) for a in network_output_list]
        if plot_type == "r2":
            ape_avg_list_i = [np.average([b.r2_by_epoch[-1] for b in a.metric_by_cvs]) for a in network_output_list]
        ape_avg_list.append(ape_avg_list_i[0])
    time_shift_list = np.arange(0,len(ape_avg_list))
    if ape_avg_list is not None:
        ax.plot(time_shift_list, ape_avg_list, label=ax_label, color=color, marker="None",linestyle=":") #,linestyle="None"
    ax.legend(fontsize=fontsize)
    ax.grid(c='k', ls='-', alpha=0.3)

    return ax

def load_position_network_output(path):
    dict_files = glob.glob(path + "output/" + "*.pkl")
    net_out = []
    sorted_list = []
    if len(dict_files) == 0:
        raise OSError("Warning: network Directory is empty")
    for i, file_path in enumerate(dict_files):
        net_dict_i = load_pickle(file_path)
        sorted_list.append([file_path, net_dict_i.net_data.time_shift])
    sorted_list = sorted(sorted_list, key=lambda x: x[1])
    dict_files = [i[0] for i in sorted_list]
    for file_path in dict_files:
        # print("processing", file_path)
        net_dict_i = load_pickle(file_path)
        net_out.append(net_dict_i)
    return net_out


if __name__ == '__main__':

    # Settings
    fontsize = 24
    plot_error_bars = True
    epoch = -1 # raw data contains information about all epochs, we only want the newest iteration
    # plot_type="r2"
    plot_type="ape"

    # ]

    plot_error_bars =False
    neuron_filter_list = ["56","44","33","22","11","1"]
    directory = "C:/Users/NN/Desktop/Master/experiments/Experiments for thesis 2/neuron filter/"
    ax_label_list = ["hc","pfc","chc","cpfc"]
    data_set = ax_label_list[0]
    path_list_1 = [
        directory + data_set + neuron_filter_list[0] + "/",
        directory + data_set + neuron_filter_list[1] + "/",
        directory + data_set + neuron_filter_list[2] + "/",
        directory + data_set + neuron_filter_list[3] + "/",
        directory + data_set + neuron_filter_list[4] + "/",
        directory + data_set + neuron_filter_list[5] + "/",
    ]
    data_set = ax_label_list[1]
    path_list_2 = [
        directory + data_set + neuron_filter_list[0] + "/",
        directory + data_set + neuron_filter_list[1] + "/",
        directory + data_set + neuron_filter_list[2] + "/",
        directory + data_set + neuron_filter_list[3] + "/",
        directory + data_set + neuron_filter_list[4] + "/",
        directory + data_set + neuron_filter_list[5] + "/",
    ]
    data_set = ax_label_list[2]
    path_list_3 = [
        directory + data_set + neuron_filter_list[0] + "/",
        directory + data_set + neuron_filter_list[1] + "/",
        directory + data_set + neuron_filter_list[2] + "/",
        directory + data_set + neuron_filter_list[3] + "/",
        directory + data_set + neuron_filter_list[4] + "/",
        directory + data_set + neuron_filter_list[5] + "/",
    ]
    data_set = ax_label_list[3]
    path_list_4 = [
        directory + data_set + neuron_filter_list[0] + "/",
        directory + data_set + neuron_filter_list[1] + "/",
        directory + data_set + neuron_filter_list[2] + "/",
        directory + data_set + neuron_filter_list[3] + "/",
        directory + data_set + neuron_filter_list[4] + "/",
        directory + data_set + neuron_filter_list[5] + "/",
    ]
    color_code_list = ["red","blue","green","orange"]

    rc('font',**{'family':'serif','serif':['Palatino']})
    rc('text', usetex=True)
    rc('xtick', labelsize=fontsize)
    rc('ytick', labelsize=fontsize)
    rc('axes', labelsize=fontsize)
    fig, ax = plt.subplots()
    axis_label_x = 'Time shift [ms]'
    if plot_type == "ape":
        axis_label_y = r'$\o$ ape'
    if plot_type == "r2":
        axis_label_y = r'$\o$ r2 score'
    # plt.rc('font', family='serif', serif='Times')
    # plt.rc('xtick', labelsize=fontsize)
    # plt.rc('ytick', labelsize=fontsize)
    # plt.rc('axes', labelsize=fontsize)
    width = 3.5
    height = width / 1.5
    fig.set_size_inches(width, height)
    edit_axis(path_list_1,ax_label_list[0],color=color_code_list[0],ax=ax,plot_type=plot_type)
    # ax1.tick_params(labelsize=fontsize)
    # ax2.tick_params(labelsize=fontsize)
    # ax3.tick_params(labelsize=fontsize)
    # ax4.tick_params(labelsize=fontsize)
    ax.set_ylabel(axis_label_y,fontsize=fontsize)
    if plot_type == "r2":
        ax.axhline(0)
        ax.axhline(0)
        ax.axhline(0)
        ax.axhline(0)
    else:
        ax.set_ylim(20,80)
        ax.set_ylim(20,80)
        ax.set_ylim(20,80)
        ax.set_ylim(20,80)

    plt.show()
    plt.close()