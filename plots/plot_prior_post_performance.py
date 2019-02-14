import matplotlib
from matplotlib import pyplot as plt

from src.database_api_beta import  Filter, hann, Net_data

from src.preprocessing import lickwells_io
from src.plots import plot_performance_by_licktime, get_accuracy_for_comparison_2, get_corrected_std
from src.network_functions import initiate_lickwell_network, run_lickwell_network
from src.metrics import print_metric_details, get_metric_details
from random import seed
import numpy as np


if __name__ == '__main__':

# Removes a portion of the lick events and checks if performance changes


    model_path = "C:/Users/NN/Desktop/Master/experiments/Lickwell_prediction/MLP_HC_prior_post/"
    path = model_path + "output/"
    save_path=model_path+"images/"
    barcolor = "darkviolet"
    add_trial_numbers=False
    # load fraction and std data

    shift = 1
    lick_id_details, lick_id_details_k = get_metric_details(path, shift, pathname_metadata="_prior")
    # return_sample_count_by_lick_id(lick_id_details_k_1)
    x_1, std_1, n_1 = get_accuracy_for_comparison_2(lick_id_details, lick_id_details_k)
    std_lower_1, std_upper_1 = get_corrected_std(x_1, std_1)
    lick_id_details, lick_id_details_k = get_metric_details(path, shift, pathname_metadata="_post")
    # return_sample_count_by_lick_id(lick_id_details_k_1)
    x_2, std_2, n_2 = get_accuracy_for_comparison_2(lick_id_details, lick_id_details_k)
    std_lower_2, std_upper_2 = get_corrected_std(x_2, std_2)

    shift = -1
    lick_id_details, lick_id_details_k = get_metric_details(path, shift, pathname_metadata="_prior")
    # return_sample_count_by_lick_id(lick_id_details_k_1)
    x_3, std_3, n_3 = get_accuracy_for_comparison_2(lick_id_details, lick_id_details_k)
    std_lower_3, std_upper_3 = get_corrected_std(x_3, std_3)
    lick_id_details, lick_id_details_k = get_metric_details(path, shift, pathname_metadata="_post")
    # return_sample_count_by_lick_id(lick_id_details_k_1)
    x_4, std_4, n_4 = get_accuracy_for_comparison_2(lick_id_details, lick_id_details_k)
    std_lower_4, std_upper_4 = get_corrected_std(x_4, std_4)
    # plot bar charts

    width = 0.75
    fontsize = 12
    font = {'family': 'normal',
            'size': 12}
    matplotlib.rc('font', **font)
    matplotlib.rc('xtick', labelsize=fontsize - 3)

    ind = np.arange(3)  # the x locations for the groups
    fig, ((ax1, ax3), (ax2, ax4)) = plt.subplots(2, 2)
    ax1.set_ylim(0, 1.0)
    ax2.set_ylim(0, 1.0)
    ax3.set_ylim(0, 1.0)
    ax4.set_ylim(0, 1.0)

    error_kw = {'capsize': 5, 'capthick': 1, 'ecolor': 'black'}
    ax1.bar(ind, x_1, color=barcolor, yerr=[std_lower_1, std_upper_1], error_kw=error_kw, align='center')
    ax1.set_xticks(ind)
    ax1.set_ylabel("fraction decoded correctly", fontsize=fontsize)
    ax1.set_xticklabels(['next lick is target', 'last lick is target', 'phase is target'], fontsize=fontsize)
    if add_trial_numbers is True:
        for i, j in zip(ind, x_1):
            if j < 0.2:
                offset = 0.1
            else:
                offset = -0.1
            ax1.annotate(int(n_1[i]), xy=(i - 0.1, j + offset))

    ax2.bar(ind, x_2, color=barcolor, yerr=[std_lower_2, std_upper_2], error_kw=error_kw, align='center')
    ax2.set_xticks(ind)
    ax2.set_ylabel("fraction decoded correctly", fontsize=fontsize)
    ax2.set_xticklabels(['next lick is target', 'last lick is target', 'phase is target'], fontsize=fontsize)
    if add_trial_numbers is True:
        for i, j in zip(ind, x_2):
            if j < 0.2:
                offset = 0.1
            else:
                offset = -0.1
            ax2.annotate(int(n_2[i]), xy=(i - 0.1, j + offset))
    ax3.bar(ind, x_3, color=barcolor, yerr=[std_lower_3, std_upper_3], error_kw=error_kw, align='center')
    ax3.set_xticks(ind)
    ax3.set_xticklabels(['next lick is target', 'last lick is target', 'phase is target'], fontsize=fontsize)
    if add_trial_numbers is True:
        for i, j in zip(ind, x_3):
            if j < 0.2:
                offset = 0.1
            else:
                offset = -0.1
            ax3.annotate(int(n_3[i]), xy=(i - 0.1, j + offset))
    ax4.bar(ind, x_4, color=barcolor, yerr=[std_lower_4, std_upper_4], error_kw=error_kw, align='center')
    ax4.set_xticks(ind)
    ax4.set_xticklabels(['next lick is target', 'last lick is target', 'phase is target'], fontsize=fontsize)
    if add_trial_numbers is True:
        for i, j in zip(ind, x_4):
            if j < 0.2:
                offset = 0.1
            else:
                offset = -0.1
            ax4.annotate(int(n_4[i]), xy=(i - 0.1, j + offset))

    # plt.tight_layout(pad=0.1, w_pad=0.5, h_pad=0)
    plt.show()
    plt.savefig(save_path)
    pass
