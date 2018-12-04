import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from matplotlib.colors import BoundaryNorm
from src.settings import save_as_pickle, load_pickle
import glob
import numpy as np
from scipy.interpolate import interp1d
from matplotlib.patches import Patch


def load_trained_network(path):
    dict_files = glob.glob(path + "output/" + "*.pkl")
    if len(dict_files) == 0:
        raise OSError("Warning: network Directory is empty")
    r2_scores_valid_list = []
    r2_scores_train_list = []
    acc_scores_valid_list = []
    acc_scores_train_list = []
    avg_scores_valid_list = []
    avg_scores_train_list = []
    time_shift_list = []
    sorted_list = []
    for i, file_path in enumerate(dict_files):
        net_dict_i = load_pickle(file_path)
        sorted_list.append([file_path, net_dict_i["TIME_SHIFT"]])
    sorted_list = sorted(sorted_list, key=lambda x: x[1])
    dict_files = [i[0] for i in sorted_list]
    for file_path in dict_files:
        print("processing", file_path)
        net_dict_i = load_pickle(file_path)
        r2_scores_train_list.append(net_dict_i["r2_scores_train"])
        r2_scores_valid_list.append(net_dict_i["r2_scores_valid"])
        acc_scores_train_list.append(net_dict_i["acc_scores_train"])
        acc_scores_valid_list.append(net_dict_i["acc_scores_valid"])
        avg_scores_train_list.append(net_dict_i["avg_scores_train"])
        avg_scores_valid_list.append(net_dict_i["avg_scores_valid"])
        time_shift_list.append(net_dict_i["TIME_SHIFT"])
    # if len(dict_files) == 1:
    #     r2_scores_train_list = [r2_scores_train_list]
    #     r2_scores_valid_list = [r2_scores_valid_list]
    #     acc_scores_train_list = [acc_scores_train_list]
    #     acc_scores_valid_list = [acc_scores_valid_list]
    #     avg_scores_train_list = [avg_scores_train_list]
    #     avg_scores_valid_list = [avg_scores_valid_list]

    return r2_scores_valid_list, r2_scores_train_list, acc_scores_valid_list, acc_scores_train_list, avg_scores_valid_list, avg_scores_train_list, net_dict_i, time_shift_list

# PATH_2 = "G:/master_datafiles/trained_networks/MLP_HC_2018-11-11_1000_200_100_dmf/"
# PATH = "G:/master_datafiles/trained_networks/MLP_PFC_2018-11-06_1000_200_100_dmf/"

PATH = "G:/master_datafiles/trained_networks/MLP_PFC_2018-12-03_1000_200_200_dmf/" #"G:/master_datafiles/trained_networks/test_MLP_HC_2018-11-13_1000_200_100_dmf/"#
PATH_2 = "G:/master_datafiles/trained_networks/MLP_PFC_2018-12-03_1000_200_200_dmf/"#"G:/master_datafiles/trained_networks/test_MLP_HC_2018-11-13_1000_200_100_dmf/"
SINGLE_ACCURACY = False
SINGLE_AVERAGE = False
SINGLE_R2 = False
COMPARE_ACCURACY = False
COMPARE_DISTANCE = False
COMPARE_R2 = False
PAIRED_T_TEST = True
FILTER_NEURON_TEST = False
COMPARE_DISCRETE = False

r2_scores_valid_list, r2_scores_train_list, acc_scores_valid_list, acc_scores_train_list, avg_scores_valid_list, avg_scores_train_list, net_dict, time_shift_list = load_trained_network(
    PATH)

training_step_list = [net_dict["METRIC_ITER"]]
for i in range(0, len(r2_scores_valid_list[0]) - 1):
    training_step_list.append(training_step_list[-1] + net_dict["METRIC_ITER"])

# ----------------------------------------------------


r2_scores_valid = [x[-1] for x in r2_scores_valid_list] # TODO uncomment
# r2_scores_train = [x[-1] for x in r2_scores_train_list]
acc_scores_valid = list(map(list, zip(*[e[-1] for e in acc_scores_valid_list]))) # TODO uncomment
# acc_scores_train = list(map(list, zip(*[e[-1] for e in acc_scores_train_list])))
distance_scores_valid = [x[-1] for x in avg_scores_valid_list]  # takes the latest trained value for each time shift
# distance_scores_train = [x[-1] for x in avg_scores_train_list]
plt.ion()
#
#     # Get data for current amount of training steps
#
#     r2_scores_valid = [x[-i] for x in r2_scores_valid_list]
#     r2_scores_train = [x[-i] for x in r2_scores_train_list]
#     acc_scores_valid = list(map(list, zip(*[e[-i] for e in acc_scores_valid_list])))
#     acc_scores_train = list(map(list, zip(*[e[-i] for e in acc_scores_train_list])))
#     distance_scores_valid = [x[-i] for x in avg_scores_valid_list] # takes the latest trained value for each time shift
#     distance_scores_train = [x[-i] for x in avg_scores_train_list]
#
#     # acc_scores_valid_list =np.array(acc_scores_valid_list).T.tolist()
#     # acc_scores_train_list = np.array(acc_scores_train_list).T.tolist()
#     # distance_scores_valid_list = np.array(distance_scores_valid_list).T.tolist()
#     # distance_scores_valid_list = np.array(distance_scores_valid_list).T.tolist()


# cf = ax0.contourf(time_shift_list,distance_list,acc_scores_train, levels=levels, cmap=cmap)
# fig.colorbar(cf, ax=ax0)
# # ax0.grid(c='k', ls='-', alpha=0.3)
# ax0.set_title('Portion of training predictions in radius wrt time-shift')
# ax0.set_xlabel("time shift [ms]")
# ax0.set_ylabel("distance to actual position(cm)")
# # ax0.set_xticklabels(time_shift_list)

# ax0.set_xticks(time_shift_list)
if SINGLE_AVERAGE is True:
    distance_list = np.linspace(0, 20, 20)
    levels = MaxNLocator(nbins=20).tick_values(np.min(acc_scores_valid), np.max(acc_scores_valid))
    cmap = plt.get_cmap('inferno')
    norm = BoundaryNorm(levels, ncolors=cmap.N, clip=True)
    # fig, (ax0, ax1) = plt.subplots(nrows=2)
    fig, (ax1) = plt.subplots()
    levels = MaxNLocator(nbins=20).tick_values(np.min(acc_scores_valid), np.max(acc_scores_valid))
    cf = ax1.contourf(time_shift_list, distance_list, acc_scores_valid, levels=levels, cmap=cmap)
    cbar = plt.colorbar(cf, ax=ax1)
    cbar.set_label('fraction of instances in range', rotation=270, labelpad=20)
    # ax1.set_title('Portion of predictions inside given radius wrt time-shift')
    ax1.set_xlabel("time shift [ms]")
    ax1.set_ylabel("absolute position error [cm]")
    # ax1.set_xticks(time_shift_list)
    fig.tight_layout()
    plt.show()
    # plt.savefig(PATH + "images/acc_score" + "_epoch=" + str(training_step_list[-i]) + ".pdf")
    plt.close()
if SINGLE_ACCURACY is True:
    fig, ax = plt.subplots()
    # ax.plot(time_shift_list,distance_scores_train,label='Training set',color='r')
    ax.plot(time_shift_list, distance_scores_valid, label='validation set', color='r')
    ax.legend()
    ax.grid(c='k', ls='-', alpha=0.3)
    # ax.set_title(r'$\varnothing$distance of validation wrt time-shift')
    ax.set_xlabel("Time shift [ms]")
    ax.set_ylabel(r'$\varnothing$ absolute position error [cm]')
    fig.tight_layout()
    plt.show()
    # plt.savefig(PATH + "images/avg_dist" + "_epoch=" + str(training_step_list[-i]) + ".pdf")
    plt.close()
if SINGLE_R2 is True:
    # fig, (ax0, ax1) = plt.subplots(nrows=2,sharey=True)
    fig, ax1 = plt.subplots()
    # ax0.grid(c='k', ls='-', alpha=0.3)
    # ax0.plot(time_shift_list,r2_scores_train)
    # ax0.set_title('r2 of training wrt time-shift')
    # ax0.set_xlabel("Time shift (s)")
    # ax0.set_ylabel("r2 score")
    ax1.plot(time_shift_list, r2_scores_valid)
    ax1.grid(c='k', ls='-', alpha=0.3)
    # ax1.set_title('R2 of validation wrt time-shift')
    ax1.set_xlabel("Time shift [ms]")
    ax1.set_ylabel("R2 score")
    ax1.set_ylim([-1, 0.6])
    fig.tight_layout()
    plt.show()
    # plt.savefig(PATH + "images/r2_score" + "_epoch=" + str(training_step_list[-i]) + ".pdf")
    plt.close()
# ---------------------------------------------------------------

# Comparisons


r2_scores_valid_list_2, r2_scores_train_list_2, acc_scores_valid_list_2, acc_scores_train_list_2, avg_scores_valid_list_2, avg_scores_train_list_2, net_dict_2, time_shift_list_2 = load_trained_network(
    PATH_2)
acc_scores_valid = list(map(list, zip(*[e[-1] for e in acc_scores_valid_list]))) # TODO uncomment
acc_scores_valid_2 = list(map(list, zip(*[e[-1] for e in acc_scores_valid_list_2])))
r2_scores_valid = [x[-1][0] for x in r2_scores_valid_list]
r2_scores_valid_2 = [x[-1][0] for x in r2_scores_valid_list_2]
distance_scores_valid = [x[-1] for x in avg_scores_valid_list]  # takes the latest trained value for each time shift
distance_scores_valid_2 = [x[-1] for x in avg_scores_valid_list_2]  # takes the latest trained value for each time shift

acc_scores_middle = np.ndarray.tolist(np.array(acc_scores_valid) - np.array(acc_scores_valid_2))
r2_scores_middle = np.ndarray.tolist(np.array(r2_scores_valid) - np.array(r2_scores_valid_2))
distance_scores_middle = np.ndarray.tolist(np.array(distance_scores_valid) - np.array(distance_scores_valid_2))

# Compare accuracy plot
if COMPARE_ACCURACY is True:
    cmap = plt.get_cmap('PiYG')
    fig, ax1 = plt.subplots()
    distance_list = np.linspace(0, 20, 20)
    levels = MaxNLocator(nbins=20).tick_values(np.min(acc_scores_middle), np.max(acc_scores_middle))
    cf = ax1.contourf(time_shift_list, distance_list, acc_scores_middle, levels=levels, cmap=cmap)
    cbar = plt.colorbar(cf, ax=ax1)
    cbar.set_label(r'$\Delta$ fraction of instances in range', rotation=270, labelpad=20)
    custom_lines = [Patch(facecolor='mediumvioletred', edgecolor='b',
                          label='acc_1'),
                    Patch(facecolor='green', edgecolor='b',
                          label='acc_2')
                    ]
    ax1.legend(custom_lines, ['PFC underperformance', 'PFC overperformance'], bbox_to_anchor=(1, 1),
               bbox_transform=plt.gcf().transFigure)
    # ax1.set_title('Portion of predictions inside given radius wrt time-shift')
    ax1.set_xlabel("time shift [ms]")
    ax1.set_ylabel(r'$\Delta$ absolute position error [cm]')
    # ax1.set_xticks(time_shift_list)
    fig.tight_layout()
    plt.ion()
    # plt.savefig(PATH + "images/acc_score_middle" + "_epoch=" + str(training_step_list[-i]) + ".pdf")
    plt.show()
    plt.close()
# Compare distances plot

if COMPARE_DISTANCE is True:
    fig, ax = plt.subplots()
    # ax.plot(time_shift_list,distance_scores_train,label='Training set',color='r')
    ax.plot(time_shift_list, distance_scores_middle, color='k')
    ax.plot(time_shift_list, distance_scores_valid, color='darkgreen')
    ax.plot(time_shift_list, distance_scores_valid_2, color='maroon')
    # ax.legend()
    custom_lines = [Patch(facecolor='green', edgecolor='b',
                          label='d_1'),
                    Patch(facecolor='red', edgecolor='b',
                          label='d_2')
                    ]
    ax.legend(custom_lines, [r'$\varnothing$ absolute position error PFC', r'$\varnothing$ absolute position error HC'])
    ax.grid(c='k', ls='-', alpha=0.3)
    # ax.set_title(r'$\varnothing$ distance of validation wrt time-shift')
    ax.set_xlabel("time shift [ms]")
    ax.set_ylabel('absolute position error [cm]')
    f = interp1d(time_shift_list, distance_scores_middle)
    x = np.linspace(time_shift_list[0], time_shift_list[-1], 1000)
    ax.fill_between(x, 0, f(x), where=(np.array(f(x))) < 0, color='green')
    ax.fill_between(x, 0, f(x), where=(np.array(f(x))) > 0, color='red')
    fig.tight_layout()
    # plt.savefig(PATH + "images/avg_dist_middle" + "_epoch=" + str(training_step_list[-i]) + ".pdf")
    plt.close()
if COMPARE_R2 is True:
    # fig, (ax0, ax1) = plt.subplots(nrows=2,sharey=True)
    fig, ax1 = plt.subplots()
    # ax0.grid(c='k', ls='-', alpha=0.3)
    # ax0.plot(time_shift_list,r2_scores_train)
    # ax0.set_title('r2 of training wrt time-shift')
    # ax0.set_xlabel("time shift [ms]")
    # ax0.set_ylabel("r2 score")
    custom_lines = [Patch(facecolor='green', edgecolor='b',
                          label='R2_1'),
                    Patch(facecolor='red', edgecolor='b',
                          label='R2_2')
                    ]
    ax1.legend(custom_lines, ['R2 Prefrontal Cortex', 'R2 Hippocampus'])
    f = interp1d(time_shift_list, r2_scores_middle)
    x = np.linspace(time_shift_list[0], time_shift_list[-1], 1000)
    ax1.fill_between(x, 0, f(x), where=(np.array(f(x))) < 0, color='maroon')
    ax1.fill_between(x, 0, f(x), where=(np.array(f(x))) > 0, color='darkgreen')
    ax1.plot(time_shift_list, r2_scores_middle)
    ax1.plot(time_shift_list, r2_scores_valid, color="g")
    ax1.plot(time_shift_list, r2_scores_valid_2, color="r")
    ax1.plot(time_shift_list, r2_scores_middle)
    ax1.grid(c='k', ls='-', alpha=0.3)
    # ax1.set_title('R2 of validation wrt time-shift')
    ax1.set_xlabel("time shift [ms]")
    ax1.set_ylabel('R2')
    ax1.set_ylim([-1, 0.6])
    fig.tight_layout()
    plt.savefig(PATH + "images/r2_score_middle" + "_epoch=" + str(training_step_list[-i]) + ".pdf")
    plt.close()
# paired t test

if PAIRED_T_TEST is True:
    test_samples = (len(acc_scores_valid[0]) - 1) // 2
    positive_range = range(test_samples , len(acc_scores_valid[0]))
    negative_range = range(test_samples , -1, -1)
    time_shift_list_t_test = time_shift_list[test_samples:]
    positive_acc_score = [[a[i] for i in positive_range] for a in acc_scores_valid]
    negative_acc_score = [[a[i] for i in negative_range] for a in acc_scores_valid]
    t_score_list_acc = np.ndarray.tolist(np.array(positive_acc_score) - np.array(negative_acc_score))
    positive_avg_score = distance_scores_valid[test_samples:]
    negative_avg_score = distance_scores_valid[test_samples::-1]
    t_score_list_avg = np.ndarray.tolist(np.array(positive_avg_score) - np.array(negative_avg_score))
    cmap = plt.get_cmap('PRGn')
    fig, ax1 = plt.subplots()
    distance_list = np.linspace(0, 20, 20)
    level_range = max(np.abs(np.min(t_score_list_acc)),np.abs(np.max(t_score_list_acc)))
    levels = MaxNLocator(nbins=20).tick_values(-level_range, level_range)
    cf = ax1.contourf(time_shift_list_t_test, distance_list, t_score_list_acc, levels=levels, cmap=cmap)
    cbar = plt.colorbar(cf, ax=ax1)
    cbar.set_label(r'$\Delta$ fraction of instances in range', rotation=270, labelpad=20)
    custom_lines = [Patch(facecolor='green', edgecolor='b',
                          label='acc_1'),
                    Patch(facecolor='mediumvioletred', edgecolor='b',
                          label='acc_2')
                    ]
    ax1.legend(custom_lines, ['positive time shift overperforms', 'negative time shift overperforms'], bbox_to_anchor=(1, 1),
               bbox_transform=plt.gcf().transFigure)
    # ax1.set_title('Portion of predictions inside given radius wrt time-shift')
    ax1.set_xlabel("absolute time shift [ms]")
    ax1.set_ylabel(r'$\Delta$ absolute position error [cm]')
    # ax1.set_xticks(time_shift_list)
    fig.tight_layout()
    plt.savefig(PATH + "images/t-test_acc" + "_epoch=" + str(training_step_list[-i]) + ".pdf")
    fig, ax = plt.subplots()
    # ax.plot(time_shift_list,distance_scores_train,label='Training set',color='r')

    # t test distance compare
    ax.plot(time_shift_list_t_test, t_score_list_avg, color='k')
    ax.plot(time_shift_list_t_test, positive_avg_score, color='darkgreen')
    ax.plot(time_shift_list_t_test, negative_avg_score, color='maroon')
    ax.axhline(y=0)
    # ax.legend()
    custom_lines = [Patch(facecolor='green', edgecolor='b',
                          label='d_1'),
                    Patch(facecolor='red', edgecolor='b',
                          label='d_2')
                    ]
    ax.legend(custom_lines, ['positive time shifts', 'negative time shifts'])
    ax.grid(c='k', ls='-', alpha=0.3)
    # ax.set_title(r'$\varnothing$ distance of validation wrt time-shift')
    ax.set_xlabel("absolute time shift [ms]")
    ax.set_ylabel('absolute position error [cm]')
    f = interp1d(time_shift_list_t_test, t_score_list_avg)
    x = np.linspace(time_shift_list_t_test[0], time_shift_list_t_test[-1], 1000)
    ax.fill_between(x, 0, f(x), where=(np.array(f(x))) < 0, color='green')
    ax.fill_between(x, 0, f(x), where=(np.array(f(x))) > 0, color='red')
    fig.tight_layout()
    plt.show()
    plt.savefig(PATH + "images/t_score_avg" + "_epoch=" + str(training_step_list[-i]) + ".pdf")


if FILTER_NEURON_TEST is True:
    PATH_100 = "G:/master_datafiles/trained_networks/MLP_HC_2018-11-08_1000_200_100_neuron_filter=100/"
    # # PATH_90 = "G:/master_datafiles/trained_networks/MLP_HC_2018-11-08_1000_200_100_neuron_filter=90/"
    PATH_80 = "G:/master_datafiles/trained_networks/MLP_HC_2018-11-08_1000_200_100_neuron_filter=80/"
    # # PATH_70 = "G:/master_datafiles/trained_networks/MLP_HC_2018-11-08_1000_200_100_neuron_filter=70/"
    PATH_60 = "G:/master_datafiles/trained_networks/MLP_HC_2018-11-08_1000_200_100_neuron_filter=60/"
    # # PATH_50 = "G:/master_datafiles/trained_networks/MLP_HC_2018-11-08_1000_200_100_neuron_filter=50/"
    PATH_40 = "G:/master_datafiles/trained_networks/MLP_HC_2018-11-08_1000_200_100_neuron_filter=40/"
    # # PATH_30 = "G:/master_datafiles/trained_networks/MLP_HC_2018-11-08_1000_200_100_neuron_filter=30/"
    PATH_20 = "G:/master_datafiles/trained_networks/MLP_HC_2018-11-08_1000_200_100_neuron_filter=20/"
    # PATH_10 = "G:/master_datafiles/trained_networks/MLP_HC_2018-11-08_1000_200_100_neuron_filter=10/"
    # PATH_100 = "G:/master_datafiles/trained_networks/MLP_PFC_2018-11-08_1000_200_100_neuron_filter=100/"
    # # PATH_90 = "G:/master_datafiles/trained_networks/MLP_PFC_2018-11-08_1000_200_100_neuron_filter=90/"
    # PATH_80 = "G:/master_datafiles/trained_networks/MLP_PFC_2018-11-08_1000_200_100_neuron_filter=80/"
    # # PATH_70 = "G:/master_datafiles/trained_networks/MLP_PFC_2018-11-08_1000_200_100_neuron_filter=70/"
    # PATH_60 = "G:/master_datafiles/trained_networks/MLP_PFC_2018-11-08_1000_200_100_neuron_filter=60/"
    # # PATH_50 = "G:/master_datafiles/trained_networks/MLP_PFC_2018-11-08_1000_200_100_neuron_filter=50/"
    # PATH_40 = "G:/master_datafiles/trained_networks/MLP_PFC_2018-11-08_1000_200_100_neuron_filter=40/"
    # # PATH_30 = "G:/master_datafiles/trained_networks/MLP_PFC_2018-11-08_1000_200_100_neuron_filter=30/"
    # PATH_20 = "G:/master_datafiles/trained_networks/MLP_PFC_2018-11-08_1000_200_100_neuron_filter=20/"
    # # PATH_10 = "G:/master_datafiles/trained_networks/MLP_PFC_2018-11-08_1000_200_100_neuron_filter=10/"
    r2_scores_valid_list_100, r2_scores_train_list_100, acc_scores_valid_list_100, acc_scores_train_list_100, avg_scores_valid_list_100, avg_scores_train_list_100, net_dict_100, time_shift_list_100 = load_trained_network(
        PATH_100)
    # r2_scores_valid_list_90, r2_scores_train_list_90, acc_scores_valid_list_90, acc_scores_train_list_90, avg_scores_valid_list_90, avg_scores_train_list_90, net_dict_90, time_shift_list_90 = load_trained_network(
    #     PATH_90)
    r2_scores_valid_list_80, r2_scores_train_list_80, acc_scores_valid_list_80, acc_scores_train_list_80, avg_scores_valid_list_80, avg_scores_train_list_80, net_dict_80, time_shift_list_80 = load_trained_network(
        PATH_80)
    # r2_scores_valid_list_70, r2_scores_train_list_70, acc_scores_valid_list_70, acc_scores_train_list_70, avg_scores_valid_list_70, avg_scores_train_list_70, net_dict_70, time_shift_list_70 = load_trained_network(
    #     PATH_70)
    r2_scores_valid_list_60, r2_scores_train_list_60, acc_scores_valid_list_60, acc_scores_train_list_60, avg_scores_valid_list_60, avg_scores_train_list_60, net_dict_60, time_shift_list_60 = load_trained_network(
        PATH_60)
    # r2_scores_valid_list_50, r2_scores_train_list_50, acc_scores_valid_list_50, acc_scores_train_list_50, avg_scores_valid_list_50, avg_scores_train_list_50, net_dict_50, time_shift_list_50 = load_trained_network(
    #     PATH_50 )
    r2_scores_valid_list_40, r2_scores_train_list_40, acc_scores_valid_list_40, acc_scores_train_list_40, avg_scores_valid_list_40, avg_scores_train_list_40, net_dict_40, time_shift_list_40 = load_trained_network(
        PATH_40 )
    # r2_scores_valid_list_30, r2_scores_train_list_30, acc_scores_valid_list_30, acc_scores_train_list_30, avg_scores_valid_list_30, avg_scores_train_list_30, net_dict_30, time_shift_list_30 = load_trained_network(
    #     PATH_30)
    r2_scores_valid_list_20, r2_scores_train_list_20, acc_scores_valid_list_20, acc_scores_train_list_20, avg_scores_valid_list_20, avg_scores_train_list_20, net_dict_20, time_shift_list_20 = load_trained_network(
        PATH_20)
    # r2_scores_valid_list_10, r2_scores_train_list_10, acc_scores_valid_list_10, acc_scores_train_list_10, avg_scores_valid_list_10, avg_scores_train_list_10, net_dict_10, time_shift_list_10 = load_trained_network(
    #     PATH_10)

    r2_scores_100 = r2_scores_valid_list_100[0]
    distance_scores_100 = avg_scores_valid_list_100[0]
    # r2_scores_90 = r2_scores_valid_list_90[0]
    # distance_scores_90 = avg_scores_valid_list_90[0]
    r2_scores_80 = r2_scores_valid_list_80[0]
    distance_scores_80 = avg_scores_valid_list_80[0]
    # r2_scores_70 = r2_scores_valid_list_70[0]
    # distance_scores_70 = avg_scores_valid_list_70[0]
    r2_scores_60 = r2_scores_valid_list_60[0]
    distance_scores_60 = avg_scores_valid_list_60[0]
    # r2_scores_50 = r2_scores_valid_list_50[0]
    # distance_scores_50 = avg_scores_valid_list_50[0]
    r2_scores_40 = r2_scores_valid_list_40[0]
    distance_scores_40 = avg_scores_valid_list_40[0]
    # r2_scores_30 = r2_scores_valid_list_30[0]
    # distance_scores_30 = avg_scores_valid_list_30[0]
    r2_scores_20 = r2_scores_valid_list_20[0]
    distance_scores_20 = avg_scores_valid_list_20[0]
    # r2_scores_10 = r2_scores_valid_list_10[0]
    # distance_scores_10 = avg_scores_valid_list_10[0]
    acc_scores_100 = [a[19] for a in acc_scores_valid_list_100[0]]
    # acc_scores_90 = [a[19] for a in acc_scores_valid_list_90[0]]
    acc_scores_80 = [a[19] for a in acc_scores_valid_list_80[0]]
    # acc_scores_70 = [a[19] for a in acc_scores_valid_list_70[0]]
    acc_scores_60 = [a[19] for a in acc_scores_valid_list_60[0]]
    # acc_scores_50 = [a[19] for a in acc_scores_valid_list_50[0]]
    acc_scores_40 = [a[19] for a in acc_scores_valid_list_40[0]]
    # acc_scores_30 = [a[19] for a in acc_scores_valid_list_30[0]]
    acc_scores_20 = [a[19] for a in acc_scores_valid_list_20[0]]
    # acc_scores_10 = [a[19] for a in acc_scores_valid_list_10[0]]

    #     # Get data for current amount of training steps
    #
    no_neurons_list = [56, 51, 45, 40, 34, 28, 23, 17, 12, 6]
    # no_neurons_list = [147, 133, 118, 103, 89, 74, 59, 45, 30, 15]

    training_step_list_filter = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18,
                                 19, 20]
    fig, ax = plt.subplots()
    # ax.plot(training_step_list_filter, distance_scores_100, color='b',label="56 (1.0)")
    # # # ax.plot(training_step_list_filter, distance_scores_90, color='g',label="51 (0.9)")
    # ax.plot(training_step_list_filter, distance_scores_80, color='r',label="45 (0.8)")
    # # # ax.plot(training_step_list_filter, distance_scores_70, color='c',label="40 (0.7)")
    # ax.plot(training_step_list_filter, distance_scores_60, color='m',label="34 (0.6)")
    # # # ax.plot(training_step_list_filter, distance_scores_50, color='y',label="28 (0.5)")
    # ax.plot(training_step_list_filter, distance_scores_40, color='k',label="23 (0.4)")
    # # # ax.plot(training_step_list_filter, distance_scores_30, color='w',label="17 (0.3)")
    # ax.plot(training_step_list_filter, distance_scores_20, color='maroon',label="12 (0.2)")
    # # ax.plot(training_step_list_filter, distance_scores_10, color='darkgreen',label=" 6 (0.1)")


    ax.plot(training_step_list_filter, distance_scores_100, color='b',label="147 (1.0)")
    # ax.plot(training_step_list_filter, distance_scores_90, color='g',label="133 (0.9)")
    ax.plot(training_step_list_filter, distance_scores_80, color='r',label="118 (0.8)")
    # # ax.plot(training_step_list_filter, distance_scores_70, color='c',label="103 (0.7)")
    ax.plot(training_step_list_filter, distance_scores_60, color='m',label=" 89 (0.6)")
    # # ax.plot(training_step_list_filter, distance_scores_50, color='y',label=" 74 (0.5)")
    ax.plot(training_step_list_filter, distance_scores_40, color='k',label=" 59 (0.4)")
    # # ax.plot(training_step_list_filter, distance_scores_30, color='w',label=" 45 (0.3)")
    ax.plot(training_step_list_filter, distance_scores_20, color='maroon',label=" 30 (0.2)")
    # ax.plot(training_step_list_filter, distance_scores_10, color='darkgreen',label=" 15 (0.1)")
    ax.legend()
    ax.grid(c='k', ls='-', alpha=0.3)
    # ax.set_title(r'$\varnothing$ distance of validation wrt time-shift')
    ax.set_xlabel("epoch")
    ax.set_ylabel('absolute position error [cm]')
    f = interp1d(time_shift_list, distance_scores_middle)
    x = np.linspace(time_shift_list[0], time_shift_list[-1], 1000)
    ax.fill_between(x, 0, f(x), where=(np.array(f(x))) < 0, color='green')
    ax.fill_between(x, 0, f(x), where=(np.array(f(x))) > 0, color='red')
    fig.tight_layout()
    plt.savefig(PATH + "images/avg_dist_middle" + "_epoch=" + str(training_step_list[-i]) + ".pdf")
    plt.close()
    fig, ax = plt.subplots()
    # ax.plot(time_shift_list,distance_scores_train,label='Training set',color='r')
    ax.plot(training_step_list_filter, acc_scores_100, color='b',label="56 (1.0)")
    # # ax.plot(training_step_list_filter, acc_scores_90, color='g',label="51 (0.9)")
    ax.plot(training_step_list_filter, acc_scores_80, color='r',label="45 (0.8)")
    # # ax.plot(training_step_list_filter, acc_scores_70, color='c',label="40 (0.7)")
    ax.plot(training_step_list_filter, acc_scores_60, color='m',label="34 (0.6)")
    # # ax.plot(training_step_list_filter, acc_scores_50, color='y',label="28 (0.5)")
    ax.plot(training_step_list_filter, acc_scores_40, color='k',label="23 (0.4)")
    # # ax.plot(training_step_list_filter, acc_scores_30, color='w',label="17 (0.3)")
    ax.plot(training_step_list_filter, acc_scores_20, color='maroon',label="12 (0.2)")
    # ax.plot(training_step_list_filter, acc_scores_10, color='darkgreen',label=" 6 (0.1)")
    # ax.plot(training_step_list_filter, acc_scores_100, color='b',label="147 (1.0)")
    # # ax.plot(training_step_list_filter, acc_scores_90, color='g',label="133 (0.9)")
    # ax.plot(training_step_list_filter, acc_scores_80, color='r',label="118 (0.8)")
    # # ax.plot(training_step_list_filter, acc_scores_70, color='c',label="103 (0.7)")
    # ax.plot(training_step_list_filter, acc_scores_60, color='m',label=" 89 (0.6)")
    # # ax.plot(training_step_list_filter, acc_scores_50, color='y',label=" 74 (0.5)")
    # ax.plot(training_step_list_filter, acc_scores_40, color='k',label=" 59 (0.4)")
    # # ax.plot(training_step_list_filter, acc_scores_30, color='w',label=" 45 (0.3)")
    # ax.plot(training_step_list_filter, acc_scores_20, color='maroon',label=" 30 (0.2)")
    # # ax.plot(training_step_list_filter, acc_scores_10, color='darkgreen',label=" 15 (0.1)")

    ax.legend()
    custom_lines = [Patch(facecolor='green', edgecolor='b',
                          label='d_1'),
                    Patch(facecolor='red', edgecolor='b',
                          label='d_2')
                    ]
    # ax.legend(custom_lines,
    #           [r'$\varnothing$ absolute position error PFC', r'$\varnothing$ absolute position error HC'])
    ax.grid(c='k', ls='-', alpha=0.3)
    ax.set_title(r'$\varnothing$ distance of validation wrt time-shift')
    ax.set_xlabel("epoch")
    ax.set_ylabel('fraction of instances in range')
    # f = interp1d(time_shift_list, distance_scores_middle)
    # x = np.linspace(time_shift_list[0], time_shift_list[-1], 1000)
    # ax.fill_between(x, 0, f(x), where=(np.array(f(x))) < 0, color='green')
    # ax.fill_between(x, 0, f(x), where=(np.array(f(x))) > 0, color='red')
    fig.tight_layout()
    plt.savefig(PATH + "images/avg_dist_middle" + "_epoch=" + str(training_step_list[-i]) + ".pdf")


if COMPARE_DISCRETE is True:
    # Get data for current amount of training steps


    acc_scores_valid_list_ts =np.array(acc_scores_valid_list).T.tolist()
    acc_scores_valid_list_ts_2 =np.array(acc_scores_valid_list_2).T.tolist()
    acc_scores_ts_middle = np.ndarray.tolist(np.array(acc_scores_valid_list_ts) - np.array(acc_scores_valid_list_ts_2))
    fig, ax = plt.subplots()
    # ax.plot(time_shift_list,distance_scores_train,label='Training set',color='r')
    ax.plot(time_shift_list, acc_scores_ts_middle, color='k')
    ax.plot(time_shift_list, acc_scores_valid_list_ts, color='darkgreen')
    ax.plot(time_shift_list, acc_scores_valid_list_ts_2, color='maroon')
    # ax.legend()
    custom_lines = [Patch(facecolor='green', edgecolor='b',
                          label='d_1'),
                    Patch(facecolor='red', edgecolor='b',
                          label='d_2')
                    ]
    ax.legend(custom_lines, [r'$\varnothing$ absolute position error PFC', r'$\varnothing$ absolute position error HC'])
    ax.grid(c='k', ls='-', alpha=0.3)
    # ax.set_title(r'$\varnothing$ distance of validation wrt time-shift')
    ax.set_xlabel("time shift [ms]")
    ax.set_ylabel('absolute position error [cm]')
    f = interp1d(time_shift_list, distance_scores_middle)
    x = np.linspace(time_shift_list[0], time_shift_list[-1], 1000)
    ax.fill_between(x, 0, f(x), where=(np.array(f(x))) < 0, color='green')
    ax.fill_between(x, 0, f(x), where=(np.array(f(x))) > 0, color='red')
    fig.tight_layout()
    # plt.savefig(PATH + "images/avg_dist_middle" + "_epoch=" + str(training_step_list[-i]) + ".pdf")