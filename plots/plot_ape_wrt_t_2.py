import matplotlib.pyplot as plt
from src.settings import save_as_pickle, load_pickle
import glob
import numpy as np
from matplotlib import rc

def edit_axis_special(model_path_list,ax_label_list,color_list,ax=None,plot_error_bars=False,plot_type="ape"):
    ax = ax or plt.gca()
    for i, path in enumerate(model_path_list):
        # Plot parameters
        network_output_list = load_position_network_output(path)
        all_samples = [a.metric_by_cvs for a in network_output_list]
        if plot_type == "ape":
            ape_avg_list = [np.average([b.ape_by_epoch[-1] for b in a.metric_by_cvs]) for a in network_output_list]
            y_all = [[a.ape_by_epoch[epoch] for a in metric] for metric in all_samples]
        if plot_type == "r2":
            ape_avg_list = [np.average([b.r2_by_epoch[-1] for b in a.metric_by_cvs]) for a in network_output_list]
            y_all = [[a.r2_by_epoch[epoch] for a in metric] for metric in all_samples]
        time_shift_list = [a.net_data.time_shift for a in network_output_list]
        errorbars = [np.std(y) for y in y_all]
        if ape_avg_list is not None:
            if plot_error_bars is False:
                ax.plot(time_shift_list, ape_avg_list, label=ax_label_list[i], color=color_list[i], marker="None",
                        linestyle=":")  # ,linestyle="None"
            else:
                ax.errorbar(x=time_shift_list, y=ape_avg_list, yerr=errorbars, capsize=2, label=ax_label_list[i],
                            color=color_list[i], marker="None", linestyle=":")
    ax.legend(fontsize=fontsize - 3)
    ax.grid(c='k', ls='-', alpha=0.3)
    return ax


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
        if ape_avg_list is not None:
            if plot_error_bars is False:
                ax.plot(time_shift_list, ape_avg_list, label=ax_label_list[i], color=color_list[i], marker="None",linestyle=":") #,linestyle="None"
            else:
                ax.errorbar(x=time_shift_list,y=ape_avg_list,yerr=errorbars,capsize=2,label=ax_label_list[i], color=color_list[i], marker="None",linestyle=":")
    ax.legend(fontsize=fontsize-3)
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
    plot_error_bars = False
    epoch = -1 # raw data contains information about all epochs, we only want the newest iteration but it can also be set to an arbitrary epoch
    combined_comparison = False # used for comparison in combined data set so values are added
    # plot_type="r2"
    plot_type="ape"

    # comparison in combined data set
    dir = "C:/Users/NN/Desktop/Master/experiments/Experiments for thesis 2/Position decoding/"

    plot_error_bars =False
    model_path_list_1 = [
        dir + "cpfc/",
    ]
    model_path_list_2 = [
        dir + "chc/"
    ]
    model_path_list_3 = [
        dir + "c/"
    ]
    model_path_list_4 = [
        dir + "c/",
        dir + "chc/",
        dir + "cpfc/"
    ]
    combined_comparison = True

    ax_label_list_1 = ["CPFC"]
    ax_label_list_2 = ["CHC"]
    ax_label_list_3 = ["Combined"]
    ax_label_list_4 = ["Combined","CPFC","CHC"]
    # ax_label_list_1 = ["PFC"]
    # ax_label_list_2 = ["CPFC"]
    # ax_label_list_3 = ["HC"]
    # ax_label_list_4 = ["CHC"]

    color_code_list_1 = ["red","maroon","firebrick","darkred"]
    color_code_list_2 = ["orange","orangered","darkorange","coral"]
    color_code_list_3 = ["aqua","navy","mediumblue","darkblue"]
    color_code_list_5 = ["purple","lightgreen","orange","darkslateblue"]
    color_code_list_4 = ["lightgreen","forestgreen","limegreen","green"]




    # regular decoding
    # dir = "C:/Users/NN/Desktop/Master/experiments/Experiments for thesis 2/Position decoding/"
    # dir = "C:/Users/NN/Desktop/Master/experiments/Experiments for thesis 2/naive test/"

    # plot_error_bars =False
    # model_path_list_1 = [
    #     dir + "pfc/",
    # ]
    # model_path_list_2 = [
    #     dir + "cpfc/"
    # ]
    # model_path_list_3 = [
    #     dir + "hc/"
    # ]
    # model_path_list_4 = [
    #     dir + "chc/"
    # ]
    # dir = "C:/Users/NN/Desktop/Master/experiments/Experiments for thesis 2/behavior component test/at lickwell/"
    # dir = "C:/Users/NN/Desktop/Master/experiments/Experiments for thesis 2/behavior component test/correct trials/"
    # model_path_list_1 = [
    #     dir + "pfc_correct_trials/",
    #     dir + "pfc_incorrect_trials/",
    #
    # ]
    # model_path_list_2 = [
    #     dir + "cpfc_correct_trials/",
    #     dir + "cpfc_incorrect_trials/",
    # ]
    # model_path_list_3 = [
    #     dir + "hc_correct_trials/",
    #     dir + "hc_incorrect_trials/",
    # ]
    # model_path_list_4 = [
    #     dir + "chc_correct_trials/",
    #     dir + "chc_incorrect_trials/",
    # ]
    #
    # ax_label_list_1 = ["PFC correct trials","incorrect trials"]
    # ax_label_list_2 = ["CPFC correct trials","incorrect trials"]
    # ax_label_list_3 = ["HC correct trials","incorrect trials"]
    # ax_label_list_4 = ["CHC correct trials","incorrect trials"]
    # # ax_label_list_1 = ["PFC"]
    # # ax_label_list_2 = ["CPFC"]
    # # ax_label_list_3 = ["HC"]
    # # ax_label_list_4 = ["CHC"]
    #
    # color_code_list_1 = ["red","maroon","firebrick","darkred"]
    # color_code_list_2 = ["orange","orangered","darkorange","coral"]
    # color_code_list_3 = ["aqua","navy","mediumblue","darkblue"]
    # color_code_list_5 = ["slateblue","midnightblue","mediumslateblue","darkslateblue"]
    # color_code_list_4 = ["lightgreen","forestgreen","limegreen","green"]

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
    rc('font',**{'family':'serif','serif':['Palatino']})
    rc('text', usetex=True)
    rc('xtick', labelsize=fontsize)
    rc('ytick', labelsize=fontsize)
    rc('axes', labelsize=fontsize)
    fig, ((ax1,ax2),(ax3,ax4)) = plt.subplots(nrows=2,ncols=2)
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
    edit_axis(model_path_list_1,ax_label_list_1,color_list=color_code_list_1,ax=ax1,plot_error_bars=plot_error_bars,plot_type=plot_type)
    edit_axis(model_path_list_2,ax_label_list_2,color_list=color_code_list_2,ax=ax2,plot_error_bars=plot_error_bars,plot_type=plot_type)
    edit_axis(model_path_list_3,ax_label_list_3,color_list=color_code_list_3,ax=ax3,plot_error_bars=plot_error_bars,plot_type=plot_type)
    if combined_comparison is False:
        edit_axis(model_path_list_4,ax_label_list_4,color_list=color_code_list_4,ax=ax4,plot_error_bars=plot_error_bars,plot_type=plot_type)
    else:
        edit_axis_special(model_path_list_4,ax_label_list_4,color_list=color_code_list_5,ax=ax4,plot_error_bars=plot_error_bars,plot_type="r2")
    # ax1.tick_params(labelsize=fontsize)
    # ax2.tick_params(labelsize=fontsize)
    # ax3.tick_params(labelsize=fontsize)
    # ax4.tick_params(labelsize=fontsize)
    ax1.set_ylabel(axis_label_y,fontsize=fontsize)
    ax3.set_ylabel(axis_label_y,fontsize=fontsize)
    ax3.set_xlabel(axis_label_x,fontsize=fontsize)
    ax4.set_xlabel(axis_label_x,fontsize=fontsize)

    if plot_type == "r2":
        pass
        ax1.axhline(0)
        ax2.axhline(0)
        ax3.axhline(0)
        ax4.axhline(0)
    else:
        ax1.set_ylim(0,100)
        ax2.set_ylim(0,100)
        ax3.set_ylim(0,100)
        if combined_comparison is False:
            ax4.set_ylim(0,100)
        else:
            ax4.set_ylim(-1,1)
            ax4b = ax4.twinx()
            ax4b.set_ylabel("r2-score")
            ax4.legend(fontsize=fontsize-6)
            ax4.axhline(0)
            ax4b.set_yticks([])
    plt.show()
    plt.savefig(save_path)
    plt.close()