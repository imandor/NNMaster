import matplotlib
from matplotlib import pyplot as plt

from src.database_api_beta import  Filter, hann, Net_data

from src.preprocessing import lickwells_io
from src.plots import plot_position_by_licktime, get_accuracy_for_comparison
from src.network_functions import initiate_lickwell_network, run_lickwell_network
from src.metrics import print_metric_details, get_metric_details
from random import seed
import numpy as np




if __name__ == '__main__':

# Removes a portion of the lick events and checks if performance changes

    model_path_list = [
       # "C:/Users/NN/Desktop/Master/experiments/Lickwell_prediction/MLP_HC_timetest/",
       #  "C:/Users/NN/Desktop/Master/experiments/Lickwell_prediction/MLP_PFC_timetest/"
"C:/Users/NN/Desktop/Master/experiments/Lickwell_prediction/MLP_HC_xkcd2/",
  # "C:/Users/NN/Desktop/Master/experiments/Lickwell_prediction/MLP_HC_xkcd2/"
    ]
    save_path = "C:/Users/NN/Desktop/Master/experiments/Lickwell_prediction/time_test_comparison.png"
    plotrange = range(0, 140)
    y_range = range(-5000,10000)
    offset = 1500  # every point is supposed to show +-500 ms around marker, but does actually 0 to 1000 ms. Offset shifts the axis labels

# for model_path in model_path_list:
        # plot_performance_by_licktime(path=model_path + "output/", shift=shift,save_path=model_path+"images/by_licktime_"+shiftpath+".png",
        #                              add_trial_numbers=True,title="Fraction decoded correctly by time into lick-event",
        #                              plotrange=plotrange)


    y_list = []
    n_list = []
    for shift in [1]: # TODO
        for path in model_path_list:
            y_n = []
            n_n = []
            for i in plotrange:
                lick_id_details, lick_id_details_k = get_metric_details(path+"output/", shift,pathname_metadata="_"+str(i))
                y, std, n = get_accuracy_for_comparison(lick_id_details, lick_id_details_k)
                y_n.append(np.average(y))
                n_n.append(np.sum(n))
            y_list.append(y_n)
            n_list.append(n_n)#
    y_list_1 = y_list[0]
    n_list_1 = n_list[0]
    # y_list_2 = y_list[1]
    # n_list_2 = n_list[1]
    # y_list_3 = y_list[2]
    # n_list_3 = n_list[2]
    # y_list_4 = y_list[3]
    # n_list_4 = n_list[3]

# plot chart
    fontsize = 16
    font = {'family': 'normal',
            'size': 12}
    matplotlib.rc('font', **font)
    matplotlib.rc('xtick', labelsize=fontsize - 3)
    corrected_ind = np.arange(y_range.start+offset,y_range.stop+offset,(y_range.stop-y_range.start)/(plotrange.stop-plotrange.start))  # the x locations for the groups
    # corrected_ind = ind * (5.0 / 39.0)
    fig, ((ax1,ax2),(ax3,ax4)) = plt.subplots(2,2)
    ax1.plot(corrected_ind, y_list_1, color="violet", label="", linestyle="None", marker="X")  # label="cv "+str(i+1)+"/10",
    ax1.set_ylabel("next well accuracy", fontsize=fontsize)
    # ax1.set_xlabel("Time since start of lick [s]",fontsize=fontsize)
    ax1.set_title("Hippocampus")
    # ax2.plot(corrected_ind, y_list_2, color="violet", label="", linestyle="None", marker="X")  # label="cv "+str(i+1)+"/10",
    # ax2.set_ylabel("fraction decoded correctly", fontsize=fontsize)
    # # ax2.set_xlabel("Time since start of lick [s]", fontsize=fontsize)
    # ax2.set_title("Prefrontal Cortex")
    # ax3.plot(corrected_ind, y_list_3, color="violet", label="", linestyle="None", marker="X")  # label="cv "+str(i+1)+"/10",
    # ax3.set_ylabel("last well accuracy", fontsize=fontsize)
    # ax3.set_xlabel("Time since start of lick [s]", fontsize=fontsize)
    # ax4.plot(corrected_ind, y_list_4, color="violet", label="", linestyle="None", marker="X")  # label="cv "+str(i+1)+"/10",
    # ax4.set_ylabel("fraction decoded correctly", fontsize=fontsize)
    ax4.set_xlabel("Time since start of lick [s]", fontsize=fontsize)
    ax1.set_ylim(0, 1)
    ax2.set_ylim(0, 1)
    ax3.set_ylim(0, 1)
    ax4.set_ylim(0, 1)
    plt.tight_layout(pad=0.1, w_pad=0.5, h_pad=0)
    plt.show()
    plt.savefig(save_path)