import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from matplotlib.colors import BoundaryNorm
from src.settings import save_as_pickle, load_pickle
import glob
import numpy as np
from scipy.interpolate import interp1d
from matplotlib.patches import Patch
from statsmodels.nonparametric.smoothers_lowess import lowess
def edit_axis(model_path_list,ax_label_list,color_list,ax=None,plot_error_bars=False,plot_type="ape"):
    ax = ax or plt.gca()
    for i,path in enumerate(model_path_list):
        # Plot parameters
        network_output_list = load_position_network_output(path)
        all_samples = [a.metric_by_cvs for a in network_output_list]
        if plot_type =="ape":
            ape_avg_list = [np.average([b.ape_by_epoch[-1] for b in a.metric_by_cvs]) for a in network_output_list]
            y_all = [[a.ape_by_epoch[epoch] for a in metric] for metric in all_samples]
        if plot_type == "r2":
            ape_avg_list = [np.average([b.r2_by_epoch[-1] for b in a.metric_by_cvs]) for a in network_output_list]
            y_all = [[a.r2_by_epoch[epoch] for a in metric] for metric in all_samples]

        time_shift_list = [a.net_data.time_shift for a in network_output_list]

        errorbars = [np.std(y) for y in y_all]
        # for i in range(len(y_all[0])):
        #     y_i = [a[i] for a in y_all]
        #     c = get_c(i)
        #     ax.plot(time_shift_list,y_i,color=c)#label="cv "+str(i+1)+"/10",
        if ape_avg_list is not None:
            if plot_error_bars is False:
                ax.plot(time_shift_list, ape_avg_list, label=ax_label_list[i], color=color_list[i], marker="None") #,linestyle="None"
            else:
                ax.errorbar(x=time_shift_list,y=ape_avg_list,yerr=errorbars,capsize=2,label=ax_label_list[i], color=color_list[i], marker="None")
    ax.legend()
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
    fontsize = 20
    plot_error_bars = True
    epoch = -1 # raw data contains information about all epochs, we only want the newest iteration
    # plot_type="r2"
    plot_type="ape"

    # Neuron filter 1
    # dir = "C:/Users/NN/Desktop/Master/experiments/Experiments for thesis/neuron filter/filter fraction/"
    # model_path_list_1 = [
    #     dir + "pfc/",
    #     dir + "pfc_1d_80/",
    #     dir + "pfc_1d_60/",
    #     dir + "pfc_1d_40/",
    #     dir + "pfc_1d_20/",
    # ]
    # model_path_list_2 = [
    #     dir + "hc/",
    #     dir + "hc_1d_80/",
    #     dir + "hc_1d_60/",
    #     dir + "hc_1d_40/",
    #     dir + "hc_1d_20/",
    # ]
    # model_path_list_3 = [
    #     dir + "cpfc/",
    #     dir + "cpfc_1d_80/",
    #     dir + "cpfc_1d_60/",
    #     dir + "cpfc_1d_40/",
    #     dir + "cpfc_1d_20/",
    # ]
    # model_path_list_4 = [
    #     dir + "chc/",
    #     dir + "chc_1d_80/",
    #     dir + "chc_1d_60/",
    #     dir + "chc_1d_40/",
    #     dir + "chc_1d_20/",
    # ]
    # ax_label_list_1 = ["PFC","117","88","58","29"]
    # ax_label_list_2 = ["HC", "44","33","22","11"]
    # ax_label_list_3 = ["CPFC","28","21","14","7"]
    # ax_label_list_4 = ["CHC","60","45","30","15"]
    #
    # color_code_list_1 = ["black","red","firebrick","darkred","maroon"]
    # color_code_list_2 = ["black","orange","darkorange","orangered","coral"]
    # color_code_list_3 = ["black","blue","mediumblue","darkblue","navy"]
    # color_code_list_4 = ["black","slateblue","mediumslateblue","darkslateblue","midnightblue"]
    # color_code_list_5 = ["black","green","limegreen","forestgreen","lightgreen"]

    # regular decoding
    dir = "C:/Users/NN/Desktop/Master/experiments/Experiments for thesis 2/Position decoding/"
    model_path_list_1 = [
        dir + "pfc/",
    ]
    model_path_list_2 = [
        dir + "cpfc/"
    ]
    model_path_list_3 = [
        dir + "hc/"
    ]
    model_path_list_4 = [
        dir + "chc/"
    ]
    ax_label_list_1 = ["PFC"]
    ax_label_list_2 = ["CPFC"]
    ax_label_list_3 = ["HC"]
    ax_label_list_4 = ["CHC"]

    color_code_list_1 = ["red","firebrick","darkred","maroon"]
    color_code_list_2 = ["orange","darkorange","orangered","coral"]
    color_code_list_3 = ["blue","mediumblue","darkblue","navy"]
    color_code_list_4 = ["slateblue","mediumslateblue","darkslateblue","midnightblue"]
    color_code_list_5 = ["green","limegreen","forestgreen","lightgreen"]

    # at lickwell vs not at lickwell
    # dir = "C:/Users/NN/Desktop/Master/experiments/Experiments for thesis/behavior component test/at lickwell/"
    # model_path_list_1 = [
    #     dir + "pfc_bc_at_lickwell/",
    #     dir + "pfc_bc_not_at_lickwell/",
    #
    # ]
    # model_path_list_2 = [
    #     dir + "cpfc_bc_at_lickwell/",
    #     dir + "cpfc_bc_not_at_lickwell/"
    #
    # ]
    # model_path_list_3 = [
    #     dir + "hc_bc_at_lickwell/",
    #     dir + "hc_bc_not_at_lickwell/"
    # ]
    # model_path_list_4 = [
    #     dir + "chc_bc_at_lickwell/",
    #     dir + "chc_bc_not_at_lickwell/"
    # ]
    # ax_label_list_1 = ["PFC at well","PFC not at well"]
    # ax_label_list_2 = ["CPFC at well", "CPFC not at well"]
    # ax_label_list_3 = ["HC at well", "HC not at well"]
    # ax_label_list_4 = ["CHC at well", "CHC not at well"]
    #
    # color_code_list_1 = ["red","maroon"]
    # color_code_list_2 = ["orange","coral"]
    # color_code_list_3 = ["blue","navy"]
    # color_code_list_4 = ["slateblue","midnightblue"]
    # color_code_list_5 = ["green","lightgreen"]



    save_path ="C:/Users/NN/Desktop/Master/experiments/position decoding/ape.png"
    fig, ((ax1,ax2),(ax3,ax4)) = plt.subplots(nrows=2,ncols=2,sharey='all')
    axis_label_x = 'Time shift [ms]'
    if plot_type == "ape":
        axis_label_y = r'$\varnothing$ abs. position error [cm]'
    if plot_type == "r2":
        axis_label_y = r'$\varnothing$ r2 score'

    # plt.rc('font', family='serif', serif='Times')
    plt.rc('text', usetex=True)
    plt.rc('xtick', labelsize=fontsize)
    plt.rc('ytick', labelsize=fontsize)
    plt.rc('axes', labelsize=fontsize)
    width = 3.5
    height = width / 1.5
    fig.set_size_inches(width, height)
    edit_axis(model_path_list_1,ax_label_list_1,color_list=color_code_list_1,ax=ax1,plot_error_bars=plot_error_bars,plot_type=plot_type)
    edit_axis(model_path_list_2,ax_label_list_2,color_list=color_code_list_2,ax=ax2,plot_error_bars=plot_error_bars,plot_type=plot_type)
    edit_axis(model_path_list_3,ax_label_list_3,color_list=color_code_list_3,ax=ax3,plot_error_bars=plot_error_bars,plot_type=plot_type)
    edit_axis(model_path_list_4,ax_label_list_4,color_list=color_code_list_4,ax=ax4,plot_error_bars=plot_error_bars,plot_type=plot_type)
    ax1.set_ylabel(axis_label_y,fontsize=fontsize)
    ax3.set_ylabel(axis_label_y,fontsize=fontsize)
    ax3.set_xlabel(axis_label_x,fontsize=fontsize)
    ax4.set_xlabel(axis_label_x,fontsize=fontsize)
    if plot_type == "r2":
        ax1.axhline(0)
        ax2.axhline(0)
        ax3.axhline(0)
        ax4.axhline(0)

    plt.show()
    plt.savefig(save_path)
    plt.close()