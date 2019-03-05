import matplotlib
from matplotlib import pyplot as plt

from src.database_api_beta import  Filter, hann, Net_data

from src.preprocessing import lickwells_io
from src.plots import get_accuracy_for_comparison_2, get_corrected_std
from src.metrics import print_metric_details, get_metric_details
from random import seed
import numpy as np


if __name__ == '__main__':

# Removes a portion of the lick events and checks if performance changes

    model_path = "C:/Users/NN/Desktop/Master/experiments/Lickwell_prediction/"
    path_hc = "C:/Users/NN/Desktop/Master/experiments/Lickwell_prediction/MLP_PFC/output/"
    path_pfc = model_path + "MLP_PFC_tleave/output/"
    save_path=model_path
    barcolor = "darkviolet"
    add_trial_numbers=True
    # load fraction and std data
    lick_id_details, lick_id_details_k = get_metric_details(path_hc, 1)
    # return_sample_count_by_lick_id(lick_id_details_k_1)
    x_1, std_1, n_1 = get_accuracy_for_comparison_2(lick_id_details, lick_id_details_k)

    std_lower_1, std_upper_1 = get_corrected_std(x_1, std_1)
    lick_id_details, lick_id_details_k = get_metric_details(path_hc, -1)
    # return_sample_count_by_lick_id(lick_id_details_k_1)
    x_2, std_2, n_2 = get_accuracy_for_comparison_2(lick_id_details, lick_id_details_k)
    std_lower_2, std_upper_2 = get_corrected_std(x_2, std_2)

    shift = -1
    lick_id_details, lick_id_details_k = get_metric_details(path_pfc, 1)
    # return_sample_count_by_lick_id(lick_id_details_k_1)
    x_3, std_3, n_3 = get_accuracy_for_comparison_2(lick_id_details, lick_id_details_k)
    std_lower_3, std_upper_3 = get_corrected_std(x_3, std_3)
    lick_id_details, lick_id_details_k = get_metric_details(path_pfc, -1)
    # return_sample_count_by_lick_id(lick_id_details_k_1)
    x_4, std_4, n_4 = get_accuracy_for_comparison_2(lick_id_details, lick_id_details_k)
    std_lower_4, std_upper_4 = get_corrected_std(x_4, std_4)
    # plot bar charts

    width = 0.75
    fontsize = 18
    font = {'family': 'normal',
            'size': 18}
    matplotlib.rc('font', **font)
    matplotlib.rc('xtick', labelsize=fontsize - 3)

    ind = np.arange(3)  # the x locations for the groups
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2)
    ax1.set_ylim(0, 1.0)
    ax2.set_ylim(0, 1.0)
    ax3.set_ylim(0, 1.0)
    ax4.set_ylim(0, 1.0)

    error_kw = {'capsize': 5, 'capthick': 1, 'ecolor': 'black'}
    ax1.bar(ind, x_1, color="r", yerr=[std_lower_1, std_upper_1], error_kw=error_kw, align='center',label="first five seconds",edgecolor="black")
    ax1.set_xticks(ind)
    ax1.set_ylabel("fraction decoded", fontsize=fontsize)
    ax1.set_xticklabels(['next well', 'last well', 'current phase'], fontsize=fontsize)
    ax1.set_title("decoding next well",fontsize=fontsize)
    if add_trial_numbers is True:
        for i, j in zip(ind, x_1):
            if j < 0.2:
                offset = 0.1
            else:
                offset = -0.1
            ax1.annotate(int(n_1[i]), xy=(i - 0.1, j + offset))

    ax2.bar(ind, x_2, color="r", yerr=[std_lower_2, std_upper_2], error_kw=error_kw, align='center',edgecolor="black")
    ax2.set_xticks(ind)
    # ax2.set_ylabel("fraction decoded", fontsize=fontsize)
    ax2.set_xticklabels(['last well', 'next well', 'current phase'], fontsize=fontsize)
    ax2.set_title("decoding last well", fontsize=fontsize)

    if add_trial_numbers is True:
        for i, j in zip(ind, x_2):
            if j < 0.2:
                offset = 0.1
            else:
                offset = -0.1
            ax2.annotate(int(n_2[i]), xy=(i - 0.1, j + offset))
    ax3.bar(ind, x_3, color="darkred", yerr=[std_lower_3, std_upper_3], error_kw=error_kw, align='center',edgecolor="black",label="last five seconds")
    ax3.set_xticks(ind)
    ax3.set_ylabel("fraction decoded", fontsize=fontsize)

    ax3.set_xticklabels(['next well', 'last well', 'current phase'], fontsize=fontsize)
    if add_trial_numbers is True:
        for i, j in zip(ind, x_3):
            if j < 0.2:
                offset = 0.1
            else:
                offset = -0.1
            ax3.annotate(int(n_3[i]), xy=(i - 0.1, j + offset))
    ax4.bar(ind, x_4, color="darkred", yerr=[std_lower_4, std_upper_4], error_kw=error_kw, align='center',edgecolor="black")
    ax4.set_xticks(ind)
    ax4.set_xticklabels(['last well', 'next well', 'current phase'], fontsize=fontsize)
    if add_trial_numbers is True:
        for i, j in zip(ind, x_4):
            if j < 0.2:
                offset = 0.1
            else:
                offset = -0.1
            ax4.annotate(int(n_4[i]), xy=(i - 0.1, j + offset))

    # plt.tight_layout(pad=0.1, w_pad=0.5, h_pad=0)
    ax1.legend()
    ax2.legend()
    ax3.legend()
    ax4.legend()
    ax1.grid(True,axis="y")
    ax2.grid(True,axis="y")
    ax3.grid(True,axis="y")
    ax4.grid(True,axis="y")
    plt.show()
    plt.savefig(save_path)
    pass
