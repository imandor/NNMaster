import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from matplotlib.colors import BoundaryNorm
from src.settings import save_as_pickle, load_pickle
import glob
import numpy as np
from scipy.interpolate import interp1d
from matplotlib.patches import Patch
from statsmodels.nonparametric.smoothers_lowess import lowess


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


def get_c(i):
    if i == 0:
        c = "b"
    if i == 1:
        c = "g"
    if i == 2:
        c = "r"
    if i == 3:
        c = "c"
    if i == 4:
        c = "m"
    if i == 5:
        c = "y"
    if i == 6:
        c = "k"
    if i == 7:
        c = "g"
    if i == 8:
        c = "sandybrown"
    if i == 9:
        c = "goldenrod"
    return c
if __name__ == '__main__':

    # Settings
    fontsize = 18
    plot_error_bars = False
    barcolor = "darkred"
    epoch = -1 # raw data contains information about all epochs, we only want the newest iteration
    dir = "C:/Users/NN/Desktop/Master/experiments/position decoding/"
    # model_path_list = "C:/Users/NN/Desktop/Master/experiments/position decoding/MLP_HC_early_stopping/"
    # model_path_list = [
    #     dir + "MLP_C/",
    #     dir + "MLP_CHC/",
    #     dir + "MLP_CPFC/",
    #     dir + "MLP_HC/",
    #     dir + "MLP_PFC/",
    #     dir + "MLP_HC_naive/",
    #     dir + "MLP_PFC_naive/",
    #     dir + "MLP_PFC_k1/",
    #     dir + "hc_1d/"
    #
    # ]
    model_path_list = [
        dir + "pfc_1d_80/",
        dir + "pfc_1d_60/",
        dir + "pfc_1d_40/",
        dir + "pfc_1d_20/",

    ]
    ax_label_list = ["PFC 80","PFC 60","PFC 40","PFC 20"]
    save_path ="C:/Users/NN/Desktop/Master/experiments/position decoding/ape.png"
    fig, ax = plt.subplots()
    axis_label_x = 'Time shift [ms]'
    axis_label_y = r'$\varnothing$ absolute position error [cm]'
    # plt.rc('font', family='serif', serif='Times')
    plt.rc('text', usetex=True)
    plt.rc('xtick', labelsize=fontsize)
    plt.rc('ytick', labelsize=fontsize)
    plt.rc('axes', labelsize=fontsize)
    width = 3.5
    height = width / 1.5
    fig.set_size_inches(width, height)

    for i,path in enumerate(model_path_list):
        # Plot parameters
        network_output_list = load_position_network_output(path)
        ape_avg_list = [np.average([b.ape_by_epoch[-1] for b in a.metric_by_cvs]) for a in network_output_list]
        all_samples = [a.metric_by_cvs for a in network_output_list]
        y_all = [[a.ape_by_epoch[epoch] for a in metric] for metric in all_samples]
        time_shift_list = [a.net_data.time_shift for a in network_output_list]

        errorbars = [np.std(y) for y in y_all]
        # for i in range(len(y_all[0])):
        #     y_i = [a[i] for a in y_all]
        #     c = get_c(i)
        #     ax.plot(time_shift_list,y_i,color=c)#label="cv "+str(i+1)+"/10",
        if ape_avg_list is not None:
            if plot_error_bars is False:
                ax.plot(time_shift_list, ape_avg_list, label=ax_label_list[i], color=get_c(i), marker="o",linestyle=":") #,linestyle="None"
            else:
                ax.errorbar(x=time_shift_list,y=ape_avg_list,yerr=errorbars,capsize=2,label=ax_label_list[i], color=get_c(i), marker="o",linestyle=":")
    ax.legend()
    ax.grid(c='k', ls='-', alpha=0.3)
    ax.set_xlabel(axis_label_x)
    ax.set_ylabel(axis_label_y)
    plt.show()
    plt.savefig(save_path)
    plt.close()