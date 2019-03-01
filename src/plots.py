import matplotlib
import numpy as np
from matplotlib import pyplot as plt

from src.metrics import get_metric_details
from src.settings import load_pickle


def plot_accuracy_inside_phase(path, shift, title, save_path, color="darkviolet"):
    # load accuracy data

    metrics = load_pickle(path + "metrics_timeshift=" + str(shift) + ".pkl")

    # plot chart
    sample_counter = np.zeros(1000)
    bin_values = []
    accuracy_sum = np.zeros(1000)
    position = 0
    current_phase = metrics[0].phase
    for i, lick in enumerate(metrics):
        sample_counter[position] += 1
        bin_values.append(position)
        accuracy_sum[position] += lick.fraction_decoded
        position += 1
        if lick.phase != current_phase:  # new phase
            current_phase = lick.phase
            position = 0

    # remove trailing zeros and normalize phase
    sample_counter = np.trim_zeros(sample_counter, 'b')
    accuracy_sum = np.trim_zeros(accuracy_sum, 'b')

    y = np.divide(accuracy_sum, sample_counter)
    fig, ax = plt.subplots()
    fontsize = 12
    x = np.arange(0, len(y))
    ax.plot(x, y, label='average', color=color, marker='.', linestyle="None")  # ,linestyle="None"
    ax.legend()
    ax.grid(c='k', ls='-', alpha=0.3)
    ax.set_xlabel("Number of visits of well 1 inside phase")
    ax.set_ylabel("Average fraction of samples decoded correctly")
    ax.set_title(title)
    ax_b = ax.twinx()
    ax_b.set_ylabel("Phases with number of visits")
    z = np.arange(0, 12)
    ax_b.hist(bin_values, bins=z, facecolor='g', alpha=0.2)
    # plt.show()
    plt.savefig(save_path)

    pass


def plot_performance_comparison(path_1, shift_1, path_2, shift_2, title_1, title_2, save_path, barcolor="darkviolet",
                                add_trial_numbers=False):
    # load fraction and std data

    lick_id_details_1, lick_id_details_k_1 = get_metric_details(path_1, shift_1)
    # return_sample_count_by_lick_id(lick_id_details_k_1)
    x_1, std_1, n_1 = get_accuracy_for_comparison(lick_id_details_1, lick_id_details_k_1)
    lick_id_details_2, lick_id_details_k_2 = get_metric_details(path_2, shift_2)
    x_2, std_2, n_2 = get_accuracy_for_comparison(lick_id_details_2, lick_id_details_k_2)
    std_lower_1, std_upper_1 = get_corrected_std(x_1, std_1)
    std_lower_2, std_upper_2 = get_corrected_std(x_2, std_2)

    # plot bar charts

    width = 0.75
    fontsize = 12
    font = {'family': 'normal',
            'size': 12}
    matplotlib.rc('font', **font)
    matplotlib.rc('xtick', labelsize=fontsize - 3)

    ind = np.arange(5)  # the x locations for the groups
    fig, (ax1, ax2) = plt.subplots(2)
    ax1.set_ylim(0,1.0)
    ax2.set_ylim(0,1.0)
    ax_b1 = ax1.twinx()
    ax_b2 = ax2.twinx()
    error_kw = {'capsize': 5, 'capthick': 1, 'ecolor': 'black'}
    ax1.bar(ind, x_1, color=barcolor, yerr=[std_lower_1, std_upper_1], error_kw=error_kw, align='center')
    ax1.set_xticks(ind)
    ax1.set_xticklabels(['all licks', 'target correct', 'target false', 'prior switch', 'after switch'])
    ax_b1.set_ylim(0, 1.0)
    ax1.set_title(title_1)
    ax1.set_ylabel("fraction decoded correctly", fontsize=fontsize)
    if add_trial_numbers is True:
        for i, j in zip(ind, x_1):
            if j < 0.2:
                offset = 0.1
            else:
                offset = -0.1
            ax1.annotate(int(n_1[i]), xy=(i - 0.1, j + offset))

    ax2.bar(ind, x_2, color=barcolor, yerr=[std_lower_2, std_upper_2], error_kw=error_kw, align='center')
    ax2.set_xticks(ind)
    ax2.set_xticklabels(['all licks', 'target correct', 'target false', 'prior switch', 'after switch'])
    ax_b2.set_ylim(0, 1.0)
    ax2.set_title(title_2)
    ax2.set_ylabel("fraction decoded correctly", fontsize=fontsize)
    if add_trial_numbers is True:
        for i, j in zip(ind, x_2):
            if j < 0.2:
                offset = 0.1
            else:
                offset = -0.1
            ax2.annotate(int(n_2[i]), xy=(i - 0.1, j + offset))

    plt.tight_layout(pad=0.1, w_pad=0.5, h_pad=0)
    plt.savefig(save_path)
    pass


def get_accuracy_for_comparison(lick_id_details, lick_id_details_k):
    """fra_list, std_list, n_list

    :param lick_id_details: lick_id_details_object
    :param lick_id_details_k: list of lick_id_detail objects, one for each k-cross-validation step in first parameter
    :return: list of accuracies by filtered lickwell,list of bernoulli standard deviations, list of absolute count of samples for each list entry
    """
    # fraction decoded in all licks
    fractions_decoded_all, std_all, n_all = return_fraction_decoded_and_std(lick_id_details=lick_id_details,
    lick_id_details_k=lick_id_details_k, parameter=lick_id_details.fraction_decoded, filter=lick_id_details.valid_licks)

    # fraction decoded if target lick is correct
    lick_id_details.filter = lick_id_details.target_lick_correct
    for i, li in enumerate(lick_id_details_k):
        lick_id_details_k[i].filter = li.target_lick_correct
    fractions_decoded_target_correct, std_target_correct, n_target_correct = return_fraction_decoded_and_std(
        lick_id_details=lick_id_details,lick_id_details_k=lick_id_details_k,parameter=lick_id_details.fraction_decoded,
        filter=lick_id_details.target_lick_correct)

    # fraction decoded if target lick is false
    lick_id_details.filter = lick_id_details.target_lick_false
    for i, li in enumerate(lick_id_details_k):
        lick_id_details_k[i].filter = li.target_lick_false
    fractions_decoded_target_false, std_target_false, n_target_false = return_fraction_decoded_and_std(
        lick_id_details=lick_id_details,lick_id_details_k=lick_id_details_k,parameter=lick_id_details.fraction_decoded,
        filter=lick_id_details.target_lick_false)

    # fraction decoded in licks prior to a switch
    lick_id_details.filter = lick_id_details.licks_prior_to_switch
    for i, li in enumerate(lick_id_details_k):
        lick_id_details_k[i].filter = li.licks_prior_to_switch
    fractions_decoded_licks_prior_to_switch, std_licks_prior_to_switch, n_licks_prior_to_switch = return_fraction_decoded_and_std(
        lick_id_details=lick_id_details,lick_id_details_k=lick_id_details_k,parameter=lick_id_details.fraction_decoded,
        filter=lick_id_details.licks_prior_to_switch)

    # fraction decoded in licks after a switch
    lick_id_details.filter = lick_id_details.licks_after_switch
    for i, li in enumerate(lick_id_details_k):
        lick_id_details_k[i].filter = li.licks_after_switch
    fractions_decoded_licks_after_switch, std_licks_after_switch, n_licks_after_switch = return_fraction_decoded_and_std(
        lick_id_details=lick_id_details,lick_id_details_k=lick_id_details_k,parameter=lick_id_details.fraction_decoded,
        filter=lick_id_details.licks_after_switch)

    fra_list = [fractions_decoded_all, fractions_decoded_target_correct, fractions_decoded_target_false,
                fractions_decoded_licks_prior_to_switch, fractions_decoded_licks_after_switch]
    std_list = [std_all, std_target_correct, std_target_false,
                std_licks_prior_to_switch, std_licks_after_switch]
    n_list = [n_all, n_target_correct, n_target_false,
              n_licks_prior_to_switch, n_licks_after_switch]
    return fra_list, std_list, n_list






def get_accuracy_for_comparison_2(lick_id_details, lick_id_details_k,k_cross=10):
    """second version on request of Professor, I wanted to keep the old one as well

    :param lick_id_details: lick_id_details_object
    :param lick_id_details_k: list of lick_id_detail objects, one for each k-cross-validation step in first parameter
    : param k_cross: cross validation factor, relevant for the updated standard deviation since the old format didn't need it
    :return: list of accuracies by filtered lickwell,list of bernoulli standard deviations, list of absolute count of samples for each list entry
    """
    # fraction decoded in all licks
    lick_id_details.filter = lick_id_details.valid_licks
    fractions_decoded_all, std_all, n_all = return_fraction_decoded_and_std(lick_id_details=lick_id_details,
    lick_id_details_k=lick_id_details_k,parameter=lick_id_details.fraction_decoded, filter=lick_id_details.valid_licks)

    # fraction decoded if target was last lick instead of next (or vice versa)
    for i, li in enumerate(lick_id_details_k):
        lick_id_details_k[i].filter = lick_id_details.filter
    fractions_decoded_last, std_last, n_last = return_fraction_decoded_and_std(
        lick_id_details=lick_id_details,lick_id_details_k=lick_id_details_k,
        parameter=lick_id_details.last_lick_decoded*lick_id_details.fraction_decoded, filter=lick_id_details.valid_licks)

    # fraction decoded if target lick is current phase
    for i, li in enumerate(lick_id_details_k):
        lick_id_details_k[i].filter = lick_id_details.filter
    fractions_decoded_phase, std_phase, n_phase = return_fraction_decoded_and_std(
        lick_id_details=lick_id_details,lick_id_details_k=lick_id_details_k,
        parameter=lick_id_details.current_phase_decoded*lick_id_details.fraction_decoded, filter=lick_id_details.valid_licks)



    fra_list = [fractions_decoded_all, fractions_decoded_last, fractions_decoded_phase]
    std_list = [std_all, std_last, std_phase]
    n_list = [n_all, n_last, n_phase]
    return fra_list, std_list, n_list


def get_corrected_std(bar_values, std_well):
    """
    :param bar_values: list of fractional values for bar charts
    :param std_well: list of corresponding standard deviations
    :return: two standard deviation lists corrected for lower and upper limits 0 and 1 of bar values (so error bars don't go below 0 and above 1
    """
    std_lower = []
    std_upper = []
    for i, std in enumerate(std_well):
        if std + bar_values[i] <= 1:
            std_upper.append(std)
        else:
            std_upper.append(1 - bar_values[i])
        if bar_values[i] - std >= 0:
            std_lower.append(std)
        else:
            std_lower.append(bar_values[i])
    return std_lower, std_upper


def return_fraction_decoded_and_std(lick_id_details,parameter,filter,lick_id_details_k,k_cross=10):
    """

    :param lick_id_details: lick_id_details object
    :param parameter: parameter which is handled as correct
    :param filter: filter data before passing to parameter
    :param lick_id_details_k: cross validated list of lick_id_details, currently deprecated
    :param k_cross: cross validation factor, currently deprecated
    :return:
    """
    confidence = 2.57
    filtered_details = []
    std_fractions = []
    n = 0
    # calculate number of samples in range and correct fraction of filtered licks

    for i, lick in enumerate(lick_id_details.licks):
        j = lick.lick_id-1
        if filter[j] == 1 and lick_id_details.valid_licks[j] == 1:
            filtered_details.append(parameter[j])
            # fractions.append(lick_id_details.fraction_decoded)
            n += lick.total_decoded
    # catch no correct decodings
    if filtered_details == []:
        fraction_decoded = 0
        std = 0
    else:
        fraction_decoded = np.average(filtered_details)
        # std = confidence * np.sqrt(fraction_decoded * (1 - fraction_decoded) / n)    # calculate bernoulli standard deviation
        k_fraction = []
        for lick_id_detail_k in lick_id_details_k:
            for i, lick in enumerate(lick_id_detail_k.licks):
                j = lick.lick_id - 1
                if filter[j] == 1 and lick_id_detail_k.valid_licks[j] == 1:
                    k_fraction.append(parameter[j])
                    # fractions.append(lick_id_details.fraction_decoded)
            if k_fraction!= []:
                std_fractions.append(np.average(k_fraction))
        std = np.std(std_fractions)
    return fraction_decoded, std, n



def metric_details_by_lickwell(path,timeshift):
    lick_id_details, lick_id_details_k = get_metric_details(path+"output/", timeshift)
    fractions_decoded_2, std_2, n_2 = return_fraction_decoded_and_std(lick_id_details=lick_id_details,
        lick_id_details_k=lick_id_details_k,parameter=lick_id_details.fraction_decoded,filter=lick_id_details.next_well_licked_2)
    fractions_decoded_3, std_3, n_3 = return_fraction_decoded_and_std(lick_id_details=lick_id_details,
        lick_id_details_k=lick_id_details_k, parameter=lick_id_details.fraction_decoded,filter=lick_id_details.next_well_licked_3)
    fractions_decoded_4, std_4, n_4 = return_fraction_decoded_and_std(lick_id_details=lick_id_details,
        lick_id_details_k=lick_id_details_k, parameter=lick_id_details.fraction_decoded,filter=lick_id_details.next_well_licked_4)
    fractions_decoded_5, std_5, n_5 = return_fraction_decoded_and_std(lick_id_details=lick_id_details,
        lick_id_details_k=lick_id_details_k, parameter=lick_id_details.fraction_decoded,filter=lick_id_details.next_well_licked_5)

    bar_values = [fractions_decoded_2, fractions_decoded_3, fractions_decoded_4, fractions_decoded_5]
    std_well = [std_2, std_3, std_4, std_5]
    n_well = [n_2,n_3,n_4,n_5]
    # so error bars show up correctly
    std_lower, std_upper = get_corrected_std(bar_values, std_well)
    return bar_values,std_lower,std_upper,n_well


def fraction_decoded_in_array(filter_func, array):
    return np.sum(filter_func * array) / np.sum(filter_func)


def plot_position_by_licktime(session,y,metadata,plotrange,title,save_path,color="darkviolet"):


    positions = []
    speeds = []
    std_lower_speed = []
    std_upper_speed = []
    std_lower = []
    std_upper = []
    lick_ids = []
    for lick in session.licks:
        if lick.lickwell == 1 and lick.lick_id != 1:
            lick_ids.append(lick.lick_id)
            timestart = lick.time
            timestop = lick.time + 5000
            slice = session[int(timestart):int(timestop)]
            positions.append(slice.position_x[0])
            # std_lower.append(np.min(slice.position_x))
            # std_upper.append(np.max(slice.position_x))
            std_lower.append(slice.position_x[2500])
            std_upper.append(slice.position_x[4999])
            speeds.append(slice.speed[2500])
            std_lower_speed.append(np.min(slice.speed))
            std_upper_speed.append(np.max(slice.speed))


    # plot
    fontsize = 12
    font = {'family': 'normal',
            'size': 12}
    matplotlib.rc('font', **font)
    matplotlib.rc('xtick', labelsize=fontsize - 3)
    ind = np.arange(len(positions))  # the x locations for the groups
    fig, ax = plt.subplots()
    # ax_b = ax.twinx()
    # ax.plot( , color=color, label="",linestyle="None",marker="X",)  # label="cv "+str(i+1)+"/10",
    error_kw = {'capsize': 5, 'capthick': 1, 'ecolor': 'black'}
    ax.plot(lick_ids,positions,color="g",linestyle="None",marker="X",label="During lick")
    ax.plot(lick_ids,std_lower,color="b",linestyle="None",marker=".",label="+2.5 seconds")
    ax.plot(lick_ids,std_upper,color="c",linestyle="None",marker=".",label="+5 seconds")
    ax.legend()
    # ax.set_xticklabels(corrected_ind)
    # ax1.set_xticklabels(['all licks', 'target correct', 'target false', 'prior switch', 'after switch'])
    ax.set_title(title)
    ax.set_ylabel("X axis position [cm]", fontsize=fontsize)
    ax.set_xlabel("Lick - id",fontsize=fontsize)
    error_kw_speed = {'capsize': 5, 'capthick': 1, 'ecolor': 'black'}

    # ax_b.errorbar(lick_ids,speeds, color="b", linestyle = "None",marker=".",yerr=[std_lower_speed, std_upper_speed])
    # ax_b.set_ylabel("Speed [cm/s]")
    plt.tight_layout(pad=0.1, w_pad=0.5, h_pad=0)
    # plt.show()
    plt.savefig(save_path)