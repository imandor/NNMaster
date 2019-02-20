import matplotlib
from matplotlib import pyplot as plt

from src.database_api_beta import  Filter, hann, Net_data

from src.preprocessing import lickwells_io
from src.plots import get_accuracy_for_comparison_2, get_corrected_std,return_fraction_decoded_and_std
from src.metrics import print_metric_details, get_metric_details
from random import seed
import numpy as np




def get_accuracy(lick_id_details, lick_id_details_k,d=0):
    """local function, compares regular accuracy, accuracy when prior post switch is removed, and acc only prior post switch

    :param lick_id_details: lick_id_details_object
    :param lick_id_details_k: list of lick_id_detail objects, one for each k-cross-validation step in first parameter
    : param k_cross: cross validation factor, relevant for the updated standard deviation since the old format didn't need it
    :return: list of accuracies by filtered lickwell,list of bernoulli standard deviations, list of absolute count of samples for each list entry
    """
    # change licks_decoded to let guesses be valid that are d off
    if d!=0:
        for i,lick in enumerate(lick_id_details.licks):
            if lick.target == lick.prediction + d or lick.target == lick.prediction-d:
                lick_id_details.valid_licks[i] = 1
            else:
                lick_id_details.valid_licks[i] = 0

    phase_change_filter = []
    for i,e in enumerate(lick_id_details.licks_prior_to_switch):
        if e == 1 or lick_id_details.licks_after_switch[i] == 1:
            phase_change_filter.append(1)
        else:
            phase_change_filter.append(0)
    # fraction decoded in all licks
    fractions_decoded_all, std_all, n_all = return_fraction_decoded_and_std(lick_id_details=lick_id_details,
    lick_id_details_k=lick_id_details_k,parameter=lick_id_details.licks_decoded, filter=lick_id_details.valid_licks)

    # fraction decoded if only phase changes are included
    lick_id_details.filter = lick_id_details.valid_licks
    fractions_decoded_phase, std_phase, n_phase = return_fraction_decoded_and_std(lick_id_details=lick_id_details,
    lick_id_details_k=lick_id_details_k,parameter=lick_id_details.licks_decoded, filter=phase_change_filter)

    # fraction decoded if phase changes are not included
    phase_change_filter = np.ndarray.tolist(np.ones(len(phase_change_filter))-phase_change_filter)# phase_change_filter ^= 1 # invert filter

    fractions_decoded_nophase, std_nophase, n_nophase = return_fraction_decoded_and_std(lick_id_details=lick_id_details,
    lick_id_details_k=lick_id_details_k,parameter=lick_id_details.licks_decoded, filter=phase_change_filter)

    fractions_decoded= [fractions_decoded_all,fractions_decoded_phase,fractions_decoded_nophase]
    std = [std_all, std_phase, std_nophase]
    n = [n_all, n_phase, n_nophase]

    return fractions_decoded,std,n



if __name__ == '__main__':

# Removes a portion of the lick events and checks if performance changes


    model_path = "C:/Users/NN/Desktop/Master/experiments/Lickwell_prediction/MLP_HC/"
    save_path=model_path+"images/"
    barcolor = "darkviolet"
    add_trial_numbers=True
    # load fraction and std data
    path_hc = model_path + "output/"
    lick_id_details, lick_id_details_k = get_metric_details(path_hc, 1)
    x_1, std_1, n_1 = get_accuracy(lick_id_details, lick_id_details_k)
    x_2, std_2, n_2 = get_accuracy(lick_id_details, lick_id_details_k, d=1)
    x_3, std_3, n_3 = get_accuracy(lick_id_details, lick_id_details_k, d=2)
    path_pfc = "C:/Users/NN/Desktop/Master/experiments/Lickwell_prediction/MLP_PFC/output/"
    lick_id_details, lick_id_details_k = get_metric_details(path_hc, 1)
    x_4, std_4, n_4 = get_accuracy(lick_id_details, lick_id_details_k)
    x_5, std_5, n_5 = get_accuracy(lick_id_details, lick_id_details_k,d=1)
    x_6, std_6, n_6 = get_accuracy(lick_id_details, lick_id_details_k,d=2)

    std_lower_1, std_upper_1 = get_corrected_std(x_1, std_1)
    std_lower_2, std_upper_2 = get_corrected_std(x_2, std_2)
    std_lower_3, std_upper_3 = get_corrected_std(x_3, std_3)
    std_lower_4, std_upper_4 = get_corrected_std(x_4, std_4)
    std_lower_5, std_upper_5 = get_corrected_std(x_5, std_5)
    std_lower_6, std_upper_6 = get_corrected_std(x_6, std_6)

# plot results

    width = 0.75
    fontsize = 16
    font = {'family': 'normal',
            'size': 12}
    matplotlib.rc('font', **font)
    matplotlib.rc('xtick', labelsize=fontsize - 3)

    ind = np.arange(3)  # the x locations for the groups
    # fig, ((ax1, ax2,ax3), (ax4, ax5,ax6)) = plt.subplots(nrows=3, ncols=2)
    fig, ((ax1, ax2),(ax3, ax4), (ax5,ax6)) = plt.subplots(nrows=3, ncols=2)

    ax1.set_ylim(0, 1.0)
    ax2.set_ylim(0, 1.0)
    ax3.set_ylim(0, 1.0)
    ax4.set_ylim(0, 1.0)
    ax5.set_ylim(0, 1.0)
    ax6.set_ylim(0, 1.0)
    error_kw = {'capsize': 5, 'capthick': 1, 'ecolor': 'black'}
    ax1.bar(ind, x_1, color="b", yerr=[std_lower_1, std_upper_1], error_kw=error_kw, align='center',label="Hippocampus")
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

    ax2.bar(ind, x_2, color="r", yerr=[std_lower_2, std_upper_2], error_kw=error_kw, align='center',label="Prefrontal Cortex")
    ax2.set_xticks(ind)
    ax2.set_ylabel("fraction decoded", fontsize=fontsize)
    ax2.set_xticklabels(['next well', 'last well', 'current phase'], fontsize=fontsize)
    if add_trial_numbers is True:
        for i, j in zip(ind, x_2):
            if j < 0.2:
                offset = 0.1
            else:
                offset = -0.1
            ax2.annotate(int(n_2[i]), xy=(i - 0.1, j + offset))
    ax3.bar(ind, x_3, color="b", yerr=[std_lower_3, std_upper_3], error_kw=error_kw, align='center',label="Hippocampus")
    ax3.set_xticks(ind)
    ax3.set_xticklabels(['last well', 'next well', 'current phase'], fontsize=fontsize)
    ax3.set_title("decoding last well", fontsize=fontsize)
    if add_trial_numbers is True:
        for i, j in zip(ind, x_3):
            if j < 0.2:
                offset = 0.1
            else:
                offset = -0.1
            ax3.annotate(int(n_3[i]), xy=(i - 0.1, j + offset))

    ax4.bar(ind, x_4, color="r", yerr=[std_lower_4, std_upper_4], error_kw=error_kw, align='center',label="Prefrontal Cortex")
    ax4.set_xticks(ind)
    ax4.set_xticklabels(['last well', 'next well', 'current phase'], fontsize=fontsize)
    if add_trial_numbers is True:
        for i, j in zip(ind, x_4):
            if j < 0.2:
                offset = 0.1
            else:
                offset = -0.1
            ax4.annotate(int(n_4[i]), xy=(i - 0.1, j + offset))

    ax5.bar(ind, x_5, color="r", yerr=[std_lower_5, std_upper_5], error_kw=error_kw, align='center',label="Prefrontal Cortex")
    ax5.set_xticks(ind)
    ax5.set_xticklabels(['last well', 'next well', 'current phase'], fontsize=fontsize)
    if add_trial_numbers is True:
        for i, j in zip(ind, x_5):
            if j < 0.2:
                offset = 0.1
            else:
                offset = -0.1
            ax4.annotate(int(n_5[i]), xy=(i - 0.1, j + offset))

    ax6.bar(ind, x_6, color="r", yerr=[std_lower_4, std_upper_4], error_kw=error_kw, align='center',label="Prefrontal Cortex")
    ax6.set_xticks(ind)
    ax6.set_xticklabels(['last well', 'next well', 'current phase'], fontsize=fontsize)
    if add_trial_numbers is True:
        for i, j in zip(ind, x_6):
            if j < 0.2:
                offset = 0.1
            else:
                offset = -0.1
            ax4.annotate(int(n_6[i]), xy=(i - 0.1, j + offset))
    # plt.tight_layout(pad=0.1, w_pad=0.5, h_pad=0)
    ax1.legend()
    ax2.legend()
    ax3.legend()
    ax4.legend()
    ax5.legend()
    ax6.legend()
    plt.show()
    plt.savefig(save_path)
    pass
