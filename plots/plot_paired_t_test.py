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
from scipy import stats



def create_t_shape_array(network_output_list,dir=1):
    if dir == -1:
        network_output_list= list(reversed(network_output_list))
    network_output_list = network_output_list[1:]
    return_list = np.zeros((len(network_output_list),len(network_output_list[0].metric_by_cvs)))
    for j,a in enumerate(network_output_list):
        metric_cvs = a.metric_by_cvs
        for i,m in enumerate(metric_cvs):
            return_list[j][i] = m.ape_by_epoch[-1]
    return return_list


def edit_axis(model_path_list,ax_label_list,color_list,ax=None,plot_error_bars=False,plot_type="ape"):
    """

    :param model_path_list:
    :param ax_label_list:
    :param color_list:
    :param ax:
    :param plot_error_bars:
    :param plot_type:
    :return: edits axis to give paired t-test p values to axis
    """
    ax = ax or plt.gca()
        # Plot parameters
    network_output_list = load_position_network_output(model_path_list[0])
    all_samples = []
    asd1 = network_output_list[len(network_output_list)//2:]
    asd2 = network_output_list[:len(network_output_list)//2+1]
    array_1 = create_t_shape_array(asd1,1)
    array_2 = create_t_shape_array(asd2,-1)
    a = stats.ttest_ind(array_1, array_2,axis=1)
    # a.pvalue[0] = 0
    time_shift_list = [a.net_data.time_shift for a in network_output_list][1:]
    time_shift_list = time_shift_list[len(time_shift_list)//2:]

    ax.plot(time_shift_list, a.pvalue, label=ax_label_list[0], color=color_list[0], marker="None",linestyle=":") #,linestyle="None"
    ax.legend(fontsize=fontsize)
    ax.grid(c='k', ls='-', alpha=0.3)
    return ax


def edit_axis_2(model_path_list, ax_label_list, color_list, ax=None, plot_error_bars=False, plot_type="ape"):
    """

    :param model_path_list:
    :param ax_label_list:
    :param color_list:
    :param ax:
    :param plot_error_bars:
    :param plot_type:
    :return: edits axis to give average value back
    """
    ax = ax or plt.gca()
    for i, path in enumerate(model_path_list):
        # Plot parameters
        network_output_list = load_position_network_output(path)
        all_samples = []
        asd1 = network_output_list[len(network_output_list) // 2:]
        asd2 = network_output_list[:len(network_output_list) // 2 + 1]
        array_1 = create_t_shape_array(asd1, 1)
        array_2 = create_t_shape_array(asd2, -1)
        a = stats.ttest_rel(array_1, array_2, axis=1)
        # a.pvalue[0] = 0
        time_shift_list = [a.net_data.time_shift for a in network_output_list][1:]
        time_shift_list = time_shift_list[len(time_shift_list) // 2:]
        network_output_list = array_1 - array_2

        # for i in range(len(y_all[0])):
        #     y_i = [a[i] for a in y_all]
        #     c = get_c(i)
        #     ax.plot(time_shift_list,y_i,color=c)#label="cv "+str(i+1)+"/10",
    x = []
    for i,output in enumerate(network_output_list):
        x.append(np.average(output))
    ax.plot(time_shift_list, x, label="", color=color_list[0], marker="None",
                        linestyle=":")  # ,linestyle="None"
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



    # regular decoding
    dir = "C:/Users/NN/Desktop/Master/experiments/Experiments for thesis 2/Position decoding/"
    # dir = "C:/Users/NN/Desktop/Master/experiments/Experiments for thesis 2/naive test/"
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
    color_code_list_5 = ["slateblue","mediumslateblue","darkslateblue","midnightblue"]
    color_code_list_4 = ["green","limegreen","forestgreen","lightgreen"]

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
    fig, ((ax1,ax2),(ax3,ax4)) = plt.subplots(nrows=2,ncols=2,sharey='all')
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
    edit_axis(model_path_list_4,ax_label_list_4,color_list=color_code_list_4,ax=ax4,plot_error_bars=plot_error_bars,plot_type=plot_type)
    # ax1.tick_params(labelsize=fontsize)
    # ax2.tick_params(labelsize=fontsize)
    # ax3.tick_params(labelsize=fontsize)
    # ax4.tick_params(labelsize=fontsize)
    ax1.set_ylabel(axis_label_y,fontsize=fontsize)
    ax3.set_ylabel(axis_label_y,fontsize=fontsize)
    ax3.set_xlabel(axis_label_x,fontsize=fontsize)
    ax4.set_xlabel(axis_label_x,fontsize=fontsize)

    if plot_type == "r2":
        ax1.axhline(0)
        ax2.axhline(0)
        ax3.axhline(0)
        ax4.axhline(0)
    # else:
        # ax1.set_ylim(-1,1)
        # ax2.set_ylim(-1,1)
        # ax3.set_ylim(-1,1)
        # ax4.set_ylim(-1,1)

    plt.show()
    plt.savefig(save_path)
    plt.close()