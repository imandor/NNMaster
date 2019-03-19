import matplotlib
from matplotlib import pyplot as plt

from src.database_api_beta import  Filter, hann, Net_data

from src.preprocessing import lickwells_io
from src.plots import plot_position_by_licktime, get_accuracy_for_comparison,return_fraction_decoded_and_std
from src.network_functions import initiate_lickwell_network, run_lickwell_network
from src.metrics import print_metric_details, get_metric_details
from random import seed
import numpy as np
from matplotlib import rc



if __name__ == '__main__':

# Removes a portion of the lick events and checks if performance changes

    model_path_list = [
       "C:/Users/NN/Desktop/Master/experiments/Lickwell_prediction/MLP_HC_timetest/",
        "C:/Users/NN/Desktop/Master/experiments/Lickwell_prediction/MLP_PFC_timetest/"

    ]
    save_path = "C:/Users/NN/Desktop/Master/experiments/Lickwell_prediction/time_test_comparison.png"
    plotrange = range(0, 139)
    y_range = range(-5000,10000)
    offset = 1000  # every point is supposed to show +-1000 ms around marker, but does actually 0 to 1000 ms. Offset shifts the axis labels


    # fraction_list = []
    # for k in plotrange:
    #     lick_id_details, lick_id_details_k = get_metric_details(model_path_list[0] + "output/", 1, pathname_metadata="_" + str(k))
    #     k_fraction = []
    #     for i,detail in enumerate(lick_id_details_k):
    #         counter = 0
    #         for j,fra in enumerate(detail.fraction_decoded):
    #             if detail.valid_licks[j] == 1:
    #                 counter += fra
    #         k_fraction.append(counter/np.sum(detail.valid_licks))
    #     fraction_list.append(k_fraction)
    # fraction_list = np.transpose(fraction_list)
    # for k in fraction_list:
    #     print(np.average(k))
    # total_fraction = np.average(fraction_list,axis=0)
    # fig,ax = plt.subplots(1)
    # for i,ie in enumerate(fraction_list[5:6]):
    #     if i == 0:
    #         c = "b"
    #     if i == 1:
    #         c = "g"
    #     if i == 2:
    #         c = "r"
    #     if i == 3:
    #         c = "c"
    #     if i == 4:
    #         c = "m"
    #     if i == 5:
    #         c = "y"
    #     if i == 6:
    #         c = "k"
    #     if i == 7:
    #         c = "g"
    #     if i == 8:
    #         c = "sandybrown"
    #     if i == 9:
    #         c = "goldenrod"
    #     ax.plot(ie,color=c,marker="X")
    #     y = []
    # # ax.plot(total_fraction,color="r")
    # plt.show()



    y_list = []
    n_list = []
    for shift in [1,-1]: # TODO
        for path in model_path_list:
            y_n = []
            n_n = []
            for i in plotrange:
                lick_id_details, lick_id_details_k = get_metric_details(path+"output/", shift,pathname_metadata="_"+str(i))
                y,std,n = return_fraction_decoded_and_std(lick_id_details=lick_id_details,
                                                                                        lick_id_details_k=lick_id_details_k,
                                                                                        parameter=lick_id_details.fraction_decoded,
                                                                                        filter=lick_id_details.valid_licks)
                y_n.append(y)
                n_n.append(np.sum(n))
            y_list.append(y_n)
            n_list.append(n_n)#
    y_list_1 = y_list[0]
    n_list_1 = n_list[0]
    y_list_2 = y_list[1]
    n_list_2 = n_list[1]
    y_list_3 = y_list[2]
    n_list_3 = n_list[2]
    y_list_4 = y_list[3]
    n_list_4 = n_list[3]

# plot chart
    fontsize = 24
    rc('font', **{'family': 'serif', 'serif': ['Palatino']})
    rc('text', usetex=True)
    rc('xtick', labelsize=fontsize)
    rc('ytick', labelsize=fontsize)
    rc('axes', labelsize=fontsize)

    corrected_ind = np.arange(y_range.start+offset,y_range.stop+offset,(y_range.stop-y_range.start)/(plotrange.stop-plotrange.start))  # the x locations for the groups
    # corrected_ind = ind * (5.0 / 39.0)
    fig, ((ax1,ax2),(ax3,ax4)) = plt.subplots(2,2)
    ax1.plot(corrected_ind, y_list_1, color="violet", label="", linestyle="None", marker="P",markersize=5)  # label="cv "+str(i+1)+"/10",
    ax1.set_ylabel("next well accuracy", fontsize=fontsize)
    # ax1.set_xlabel("Time since start of lick [ms]",fontsize=fontsize)
    ax1.set_title("hippocampus",fontsize=fontsize)
    ax2.plot(corrected_ind, y_list_2, color="violet", label="", linestyle="None", marker="P",markersize=5)  # label="cv "+str(i+1)+"/10",
    # ax2.set_ylabel("next well accuracy", fontsize=fontsize)
    # ax2.set_xlabel("Time since start of lick [ms]", fontsize=fontsize)
    ax2.set_title("prefrontal cortex",fontsize=fontsize)
    ax3.plot(corrected_ind, y_list_3, color="violet", label="", linestyle="None", marker="P",markersize=5)  # label="cv "+str(i+1)+"/10",
    ax3.set_ylabel("last well accuracy", fontsize=fontsize)
    ax3.set_xlabel("time since start of lick [ms]", fontsize=fontsize)
    ax4.plot(corrected_ind, y_list_4, color="violet", label="", linestyle="None", marker="P",markersize=5)  # label="cv "+str(i+1)+"/10",
    # ax4.set_ylabel("last well accuracy", fontsize=fontsize)
    ax4.set_xlabel("time since start of lick [ms]", fontsize=fontsize)
    ax1.set_ylim(0.25, 0.6)
    ax2.set_ylim(0.25, 0.6)
    ax3.set_ylim(0.25, 0.6)
    ax4.set_ylim(0.25, 0.6)

    # determine average during first five seconds after lick:
    ax1.axhline(np.average(y_list_1[50:100]))
    ax2.axhline(np.average(y_list_2[50:100]))
    ax3.axhline(np.average(y_list_3[50:100]))
    ax4.axhline(np.average(y_list_4[50:100]))

    # plt.tight_layout(pad=0.1, w_pad=0.5, h_pad=0)
    plt.show()
    plt.savefig(save_path)