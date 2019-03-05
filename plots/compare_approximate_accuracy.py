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
    parameter = np.zeros(len(lick_id_details.valid_licks))
    if d!=0:
        for i,lick in enumerate(lick_id_details.licks):
            if lick.lickwell == 1 and abs(lick.target - lick.prediction)<= d :
                parameter[lick.lick_id-1] = 1
            else:
                parameter[lick.lick_id-1] = 0
        parameter = np.ndarray.tolist(parameter)
    else:
        parameter = lick_id_details.licks_decoded
    fractions_decoded_all, std_all, n_all = return_fraction_decoded_and_std(lick_id_details=lick_id_details,
    lick_id_details_k=lick_id_details_k,parameter=parameter, filter=lick_id_details.valid_licks)

    # fraction decoded before a phase change
    lick_id_details.filter = lick_id_details.valid_licks
    fractions_decoded_phase, std_phase, n_phase = return_fraction_decoded_and_std(lick_id_details=lick_id_details,
    lick_id_details_k=lick_id_details_k,parameter=parameter, filter=lick_id_details.licks_prior_to_switch)

    # fraction decoded after phase change

    fractions_decoded_nophase, std_nophase, n_nophase = return_fraction_decoded_and_std(lick_id_details=lick_id_details,
    lick_id_details_k=lick_id_details_k,parameter=parameter, filter=lick_id_details.licks_after_switch)

    fractions_decoded= [fractions_decoded_all,fractions_decoded_phase,fractions_decoded_nophase]
    std = [std_all, std_phase, std_nophase]
    n = [n_all, n_phase, n_nophase]

    return fractions_decoded,std,n



if __name__ == '__main__':

# Removes a portion of the lick events and checks if performance changes


    model_path = "C:/Users/NN/Desktop/Master/experiments/Lickwell_prediction/MLP_HC/"
    save_path=model_path+"images/approximate_accuracy.png"
    barcolor = "darkviolet"
    add_trial_numbers = True
    shift = -1
    # load fraction and std data
    path_hc = model_path + "output/"
    lick_id_details, lick_id_details_k = get_metric_details(path_hc, shift)
    x_1, std_1, n_1 = get_accuracy(lick_id_details, lick_id_details_k)
    x_2, std_2, n_2 = get_accuracy(lick_id_details, lick_id_details_k, d=1)
    x_3, std_3, n_3 = get_accuracy(lick_id_details, lick_id_details_k, d=2)
    path_pfc = "C:/Users/NN/Desktop/Master/experiments/Lickwell_prediction/MLP_PFC/output/"
    lick_id_details, lick_id_details_k = get_metric_details(path_pfc, shift)
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

    width = 0.3
    fontsize = 16
    font = {'family': 'normal',
            'size': 12}
    matplotlib.rc('font', **font)
    matplotlib.rc('xtick', labelsize=fontsize - 3)

    ind = np.arange(3)  # the x locations for the groups
    # fig, ((ax1, ax2,ax3), (ax4, ax5,ax6)) = plt.subplots(nrows=3, ncols=2)
    fig, (ax1,ax2) = plt.subplots(nrows=1, ncols=2)

    ax1.set_ylim(0, 1.0)
    ax2.set_ylim(0, 1.0)

    error_kw = {'capsize': 5, 'capthick': 1, 'ecolor': 'black'}
    ax1.bar(ind, x_1, color="darkblue", error_kw=error_kw, align='center',label="d=0",edgecolor="black")
    ax1.bar(ind, x_2, color="slateblue",bottom=x_1, error_kw=error_kw, align='center', label="d=1",edgecolor="black")
    ax1.bar(ind, x_3, color="blue",bottom=x_2, error_kw=error_kw, align='center', label="d=2",edgecolor="black")

    ax1.set_xticks(ind)
    ax1.set_ylabel("fraction decoded", fontsize=fontsize)
    ax1.set_xticklabels(["all events","prior change", "after change"], fontsize=fontsize)
    ax1.set_title("hippocampus",fontsize=fontsize)
    if add_trial_numbers is True:
        for i, j in zip(ind, x_1):
            if j < 0.2:
                offset = 0.1
            else:
                offset = -0.1
            ax1.annotate(int(n_1[i]), xy=(i - 0.1, j + offset))

    ax2.set_xticks(ind)
    ax2.set_ylabel("fraction decoded", fontsize=fontsize)
    ax2.set_xticklabels(["all events","prior change", "after change"], fontsize=fontsize)
    ax2.set_title("prefrontal cortex", fontsize=fontsize)
    ax2.bar(ind, x_4, color="darkred", error_kw=error_kw, align='center',
            label="d=0",edgecolor="black")
    ax2.bar(ind, x_5, color="tomato", bottom=x_4, error_kw=error_kw, align='center',
            label="d=1",edgecolor="black")
    ax2.bar(ind, x_6, color="r", bottom=x_5, error_kw=error_kw, align='center', label="d=2",edgecolor="black")
    if add_trial_numbers is True:
        for i, j in zip(ind, x_4):
            if j < 0.2:
                offset = 0.1
            else:
                offset = -0.1
            ax2.annotate(int(n_4[i]), xy=(i - 0.1, j + offset))
    ax1.legend()
    ax2.legend()
    plt.show()
    plt.savefig(save_path)
    pass
