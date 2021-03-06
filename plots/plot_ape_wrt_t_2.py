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
        if plot_type == "ape":# or plot_type=="rpe":
            ape_avg_list = [np.average([b.ape_by_epoch[-1] for b in a.metric_by_cvs]) for a in network_output_list]
            y_all = [[a.ape_by_epoch[epoch] for a in metric] for metric in all_samples]
        if plot_type == "r2":
            ape_avg_list = [np.average([b.r2_by_epoch[-1] for b in a.metric_by_cvs]) for a in network_output_list]
            y_all = [[a.r2_by_epoch[epoch] for a in metric] for metric in all_samples]
        time_shift_list = [a.net_data.time_shift for a in network_output_list]
        errorbars = [np.std(y) for y in y_all]
        # if plot_type=="rpe":
        #     ape_avg_list == [x/divide_by for x in ape_avg_list]
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


def edit_axis(model_path_list,ax_label_list,color_list,ax=None,plot_error_bars=False,plot_type="ape",divide_by=1):
    ax = ax or plt.gca()
    if plot_type=="rpe":
        plot_error_bars=False
    for i,path in enumerate(model_path_list):
        # Plot parameters
        network_output_list = load_position_network_output(path)
        all_samples = [a.metric_by_cvs for a in network_output_list]
        if plot_type =="ape" or plot_type=="rpe":
            ape_avg_list = [np.average([b.ape_by_epoch[-1] for b in a.metric_by_cvs]) for a in network_output_list]
            y_all = [[a.ape_by_epoch[epoch] for a in metric] for metric in all_samples]
        if plot_type == "r2":
            ape_avg_list = [np.average([b.r2_by_epoch[-1] for b in a.metric_by_cvs]) for a in network_output_list]
            y_all = [[a.r2_by_epoch[epoch] for a in metric] for metric in all_samples]
        time_shift_list = [a.net_data.time_shift for a in network_output_list]
        errorbars = [np.std(y) for y in y_all]
        if plot_type=="rpe":
            ape_avg_list = [x / divide_by for x in ape_avg_list]

        if ape_avg_list is not None:
            if plot_error_bars is False or plot_type =="r2":
                ax.plot(time_shift_list, ape_avg_list, label=ax_label_list[i], color=color_list[i], marker="None",linestyle="-") #,linestyle="None"
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
    plot_error_bars = True
    epoch = -1 # raw data contains information about all epochs, we only want the newest iteration but it can also be set to an arbitrary epoch
    combined_comparison = True # used for comparison in combined data set so values are added
    # plot_type="r2"
    plot_type="ape"
    # plot_type="rpe" # ape divided by chance error

    # comparison in combined data set
    dir = "C:/Users/NN/Desktop/Master/experiments/Experiments for thesis 2/Position decoding/"
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
    # # ax_label_list_1 = ["PFC"]
    # # ax_label_list_2 = ["CPFC"]
    # # ax_label_list_3 = ["HC"]
    # # ax_label_list_4 = ["CHC"]
    #
    color_code_list_1 = ["red","maroon","firebrick","darkred"]
    color_code_list_2 = ["orange","orangered","darkorange","coral"]
    color_code_list_3 = ["aqua","navy","mediumblue","darkblue"]
    color_code_list_5 = ["purple","lightgreen","orange","darkslateblue"]
    color_code_list_4 = ["lightgreen","forestgreen","limegreen","green"]




    # regular decoding
    # dir = "C:/Users/NN/Desktop/Master/experiments/Experiments for thesis 2/Position decoding/"
    # dir = "C:/Users/NN/Desktop/Master/experiments/Experiments for thesis 2/naive test/"
    # dir = "C:/Users/NN/Desktop/Master/experiments/Experiments for thesis 2/Position decoding/"
    # #
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
    # dir = "C:/Users/NN/Desktop/Master/experiments/Experiments for thesis 2/behavior component test/speed/"
    # model_path_list_1 = [
    #     dir + "pfc_move/",
    #     dir + "pfc_rest/",
    #
    # ]
    # model_path_list_2 = [
    #     dir + "cpfc_move/",
    #     dir + "cpfc_rest/",
    # ]
    # model_path_list_3 = [
    #     dir + "hc_move/",
    #     dir + "hc_rest/",
    # ]
    # model_path_list_4 = [
    #     dir + "chc_move/",
    #     dir + "chc_rest/",
    # ]
    #
    # ax_label_list_1 = ["PFC","rest"]
    # ax_label_list_2 = ["CPFC","rest"]
    # ax_label_list_3 = ["HC","rest"]
    # ax_label_list_4 = ["CHC","rest"]

    # dir = "C:/Users/NN/Desktop/Master/experiments/Experiments for thesis 2/behavior component test/correct trials/"
    # model_path_list_1 = [
    #     dir + "pfc_correct_trials/",
    #     dir + "pfc_incorrect_trials/",
    #
    # ]
    # model_path_list_2 = [
    #     dir + "cpfc_correct_trials/",
    #     dir + "pfc_incorrect_trials/",
    # ]
    # model_path_list_3 = [
    #     dir + "hc_correct_trials/",
    #     dir + "pfc_incorrect_trials/",
    # ]
    # model_path_list_4 = [
    #     dir + "chc_correct_trials/",
    #     dir + "pfc_incorrect_trials/",
    # ]
    #
    # ax_label_list_1 = ["PFC correct trials","incorrect trials"]
    # ax_label_list_2 = ["CPFC correct trials","incorrect trials"]
    # ax_label_list_3 = ["HC correct trials","incorrect trials"]
    # ax_label_list_4 = ["CHC correct trials","incorrect trials"]
    # #
    # ax_label_list_1 = ["PFC"]
    # ax_label_list_2 = ["CPFC"]
    # ax_label_list_3 = ["HC"]
    # ax_label_list_4 = ["CHC"]
    # #
    # color_code_list_1 = ["red","maroon"]
    # color_code_list_2 = ["orange","coral"]
    # color_code_list_3 = ["blue","navy"]
    # color_code_list_4 = ["lightblue","midnightblue"]
    # color_code_list_5 = ["darkgreen","lightgreen"]

    # at lickwell vs not at lickwell
    # dir = "C:/Users/NN/Desktop/Master/experiments/Experiments for thesis 2/behavior component test/at lickwell/"
    # model_path_list_1 = [
    #     dir + "pfc_at_lickwell/",
    #     dir + "pfc_not_at_lickwell/",
    #
    # ]
    # model_path_list_2 = [
    #     dir + "cpfc_at_lickwell/",
    #     dir + "cpfc_not_at_lickwell/"
    #
    # ]
    # model_path_list_3 = [
    #     dir + "hc_at_lickwell/",
    #     dir + "hc_not_at_lickwell/"
    # ]
    # model_path_list_4 = [
    #     dir + "chc_at_lickwell/",
    #     dir + "chc_not_at_lickwell/"
    # ]
    # ax_label_list_1 = ["PFC at well","PFC not at well"]
    # ax_label_list_2 = ["CPFC at well", "CPFC not at well"]
    # ax_label_list_3 = ["HC at well", "HC not at well"]
    # ax_label_list_4 = ["CHC at well", "CHC not at well"]

    # color_code_list_1 = ["red","maroon"]
    # color_code_list_2 = ["orange","coral"]
    # color_code_list_3 = ["blue","navy"]
    # color_code_list_4 = ["lightblue","midnightblue"]
    # color_code_list_5 = ["darkgreen","lightgreen"]



    save_path ="C:/Users/NN/Desktop/Master/experiments/position decoding/ape.png"
    rc('font',**{'family':'serif','serif':['Palatino']})
    rc('text', usetex=True)
    rc('xtick', labelsize=fontsize)
    rc('ytick', labelsize=fontsize)
    rc('axes', labelsize=fontsize)
    fig, ((ax1,ax2),(ax3,ax4)) = plt.subplots(nrows=2,ncols=2,sharey=False)
    axis_label_x = 'Time shift [ms]'
    if plot_type == "ape" or plot_type=="rpe":
        axis_label_y = r'$\o$ ape [cm]'
    if plot_type == "r2":
        axis_label_y = r'$\o$ r2 score'
    # plt.rc('font', family='serif', serif='Times')
    # plt.rc('xtick', labelsize=fontsize)
    # plt.rc('ytick', labelsize=fontsize)
    # plt.rc('axes', labelsize=fontsize)
    width = 3.5
    height = width / 1.5
    fig.set_size_inches(width, height)
    edit_axis(model_path_list_1,ax_label_list_1,color_list=color_code_list_1,ax=ax1,plot_error_bars=plot_error_bars,plot_type=plot_type,divide_by=57.3)
    edit_axis(model_path_list_2,ax_label_list_2,color_list=color_code_list_2,ax=ax2,plot_error_bars=plot_error_bars,plot_type=plot_type,divide_by=61.4)
    edit_axis(model_path_list_3,ax_label_list_3,color_list=color_code_list_3,ax=ax3,plot_error_bars=plot_error_bars,plot_type=plot_type,divide_by=63.5)
    if combined_comparison is False:
        edit_axis(model_path_list_4,ax_label_list_4,color_list=color_code_list_4,ax=ax4,plot_error_bars=plot_error_bars,plot_type=plot_type,divide_by=61.4)
    else:
        edit_axis_special(model_path_list_4,ax_label_list_4,color_list=color_code_list_5,ax=ax4,plot_error_bars=False,plot_type="r2")
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
        ax1.set_ylim(-30,2)
        ax2.set_ylim(-30,2)
        ax3.set_ylim(-30,2)
    if plot_type=="ape":
        # speed
        # ax1.axhline(36.6,color=color_code_list_1[1])
        # ax1.axhline(60.2,color=color_code_list_1[0])
        # ax2.axhline(64.7,color=color_code_list_2[1])
        # ax2.axhline(60.6,color=color_code_list_2[0])
        # ax3.axhline(52.7,color=color_code_list_3[1])
        # ax3.axhline(64.5,color=color_code_list_3[0])
        # ax4.axhline(64.7,color=color_code_list_4[1])
        # ax4.axhline(60.6,color=color_code_list_4[0])
        # at well
        # ax1.axhline(59.8,color=color_code_list_1[0])
        # ax1.axhline(52.4,color=color_code_list_1[1])
        # ax2.axhline(60.1,color=color_code_list_2[0])
        # ax2.axhline(67.1,color=color_code_list_2[1])
        # ax3.axhline(62.1,color=color_code_list_3[0])
        # ax3.axhline(65.5,color=color_code_list_3[1])
        # ax4.axhline(60.1,color=color_code_list_4[0])
        # ax4.axhline(67.1,color=color_code_list_4[1])
        # correct vs incorrect trial
        # ax1.axhline(54.4,color=color_code_list_1[0])
        # ax1.axhline(82.7,color=color_code_list_1[1])
        # ax2.axhline(67.9,color=color_code_list_2[0])
        # ax2.axhline(42.5,color=color_code_list_2[1])
        # ax3.axhline(64.3,color=color_code_list_3[0])
        # ax3.axhline(62.3,color=color_code_list_3[1])
        # ax4.axhline(67.9,color=color_code_list_4[0])
        # ax4.axhline(42.5,color=color_code_list_4[1])
        # regular
        # ax1.axhline(57.3,color=color_code_list_1[0])
        # ax2.axhline(61.4,color=color_code_list_2[0])
        # ax3.axhline(63.5,color=color_code_list_3[0])
        # ax4.axhline(61.4,color=color_code_list_4[0])

        ax1.set_ylim(0,100)
        ax2.set_ylim(0,100)
        ax3.set_ylim(0,100)
    if plot_type =="rpe":
        ax1.set_ylim(0.4,1.5)
        ax2.set_ylim(0.4,1.5)
        ax3.set_ylim(0.4,1.5)
        ax4.set_ylim(0.4,1.5)
        ax1.set_ylabel("ape/chance")
        # ax2.set_ylabel("ape/chance")
        ax3.set_ylabel("ape/chance")
        # ax4.set_ylabel("ape/chance")
        ax1.axhline(1)
        ax2.axhline(1)
        ax3.axhline(1)
        ax4.axhline(1)

    else:
        if combined_comparison is False and plot_type=="ape":
            ax4.set_ylim(0,100)
        else:
            if combined_comparison is True:
                ax4.set_ylim(-1,1)
                ax4b = ax4.twinx()
                ax4b.set_ylabel("r2-score")
                ax4.legend(fontsize=fontsize-6)
                ax4.axhline(0)
                ax4b.set_yticks([])

    plt.show()
    plt.savefig(save_path)
    plt.close()