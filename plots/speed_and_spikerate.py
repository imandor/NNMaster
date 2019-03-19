from src.database_api_beta import  Filter, hann, Net_data
from matplotlib import pyplot as plt
import numpy as np
import pickle
from src.preprocessing import lickwells_io
from src.network_functions import initiate_lickwell_network, run_lickwell_network
from src.metrics import print_metric_details
from random import seed# plots histogram of times the rat spent inside a given range from well 1
from matplotlib.ticker import MaxNLocator
from src.settings import save_as_pickle, load_pickle, save_net_dict
from itertools import dropwhile,takewhile
from matplotlib import rc



def return_bar_values(nd,lickstart,lickstop,resolution):
    # local function, determines lists of times of entry and exit for plot
    session = initiate_lickwell_network(nd)  # Initialize session
    speed_list = []
    spike_rate_list = []
    for lick in session.licks:
        if lick.lickwell == 1:
            speeds = []
            spike_rates = []
            timeslice = session[int(lick.time+lickstart):int(lick.time+lickstop)]

            # determine what percentage of licks are in range for histogram
            for i in range(0,len(timeslice.position_x),resolution):
                interval = timeslice[i:i+resolution]
                speed = np.average(interval.speed)
                speeds.append(speed)
                spikes = [a for li in interval.spikes for a in li] # flatten all spikes in time range
                spike_rates.append(len(spikes)/nd.n_neurons)
            spike_rate_list.append(spike_rates)
            speed_list.append(speeds)
    speed_list = np.average(speed_list,axis=0)
    spike_rate_list = np.average(spike_rate_list,axis=0)
    return speed_list,spike_rate_list


if __name__ == '__main__':
    # Data set 2 Hippocampus

    MODEL_PATH = "G:/master_datafiles/trained_networks/MLP_HC_2019-02-07_phase/"
    RAW_DATA_PATH = "G:/master_datafiles/raw_data/HC/"
    FILTERED_DATA_PATH = "session_hc_lw.pkl"
    nd = Net_data(
        # Program execution settings
        epochs=20,
        evaluate_training=False,
        slice_size=100,
        stride=100,
        y_step=100,
        win_size=100,
        search_radius=100,
        k_cross_validation=10,
        session_filter=Filter(func=hann, search_radius=100, step_size=100),
        valid_ratio=0.1,
        testing_ratio=0,
        time_shift_steps=1,
        early_stopping=False,
        model_path=MODEL_PATH,
        raw_data_path=RAW_DATA_PATH,
        filtered_data_path=FILTERED_DATA_PATH,
        metric="discrete",
        shuffle_data=True,
        shuffle_factor=1,
        lw_classifications=4,
        lw_normalize=True,
        lw_differentiate_false_licks=False,
        num_wells=5,
        initial_timeshift=-1,
        from_raw_data=False,
        dropout=0.2
    )

    # print_metric_details(MODEL_PATH,nd.initial_timeshift)

    lickstart= -10000
    lickstop = 10000
    searchradius = 40
    xtickresolution = 2000
    # ytickresolution = 10
    fontsize = 24
    resolution = 500
    image_title_list = ["pfc_phase","hc_phase","pfc","hc"]
    path = nd.model_path+ "output/"
    save_path = nd.model_path + "images/lick_duration"+".png"
    time_ind = np.arange(lickstart, lickstop, resolution)
    rc('font',**{'family':'serif','serif':['Palatino']})
    rc('text', usetex=True)
    rc('xtick', labelsize=fontsize)
    rc('ytick', labelsize=fontsize)
    rc('axes', labelsize=fontsize)

    # speed_list_hc, spike_rate_list_hc = return_bar_values(nd, lickstart, lickstop, resolution)
    # nd.filtered_data_path = "session_pfc_lw.pkl"
    # speed_list_pfc, spike_rate_list_pfc = return_bar_values(nd, lickstart, lickstop, resolution)
    #
    # with open("speed_list_hc", 'wb') as f:
    #     pickle.dump(speed_list_hc, f)
    # with open("spike_rate_list_hc", 'wb') as f:
    #     pickle.dump(spike_rate_list_hc, f)
    #
    # with open("speed_list_pfc", 'wb') as f:
    #     pickle.dump(speed_list_pfc, f)
    # with open("spike_rate_list_pfc", 'wb') as f:
    #     pickle.dump(spike_rate_list_pfc, f)

    speed_list_hc = load_pickle("speed_list_hc")
    speed_list_pfc = load_pickle("speed_list_pfc")
    spike_rate_list_hc = load_pickle("spike_rate_list_hc")
    spike_rate_list_pfc = load_pickle("spike_rate_list_pfc")


    # plot results
    fig, ((ax1,ax3),(ax2,ax4)) = plt.subplots(2,2)
    ax1.plot(time_ind, speed_list_hc, color="b",label="Hippocampus")  # label="cv "+str(i+1)+"/10",
    ax1.set_ylabel("speed [cm/s]", fontsize=fontsize)
    ax1.xaxis.set_major_locator(plt.MaxNLocator(3))
    ax1.yaxis.set_major_locator(plt.MaxNLocator(3))
    ax1.legend(fontsize=fontsize)
    ax2.plot(time_ind, spike_rate_list_hc, color="b")  # label="cv "+str(i+1)+"/10",
    ax2.set_ylabel("spikes/s", fontsize=fontsize)
    ax2.set_xlabel("time", fontsize=fontsize)
    ax2.xaxis.set_major_locator(plt.MaxNLocator(3))
    ax2.yaxis.set_major_locator(plt.MaxNLocator(3))

    ax3.plot(time_ind, speed_list_pfc, color="r",label="Prefrontal Cortex")  # label="cv "+str(i+1)+"/10",
    # ax3.set_ylabel("speed [cm/s]", fontsize=fontsize)
    ax3.xaxis.set_major_locator(plt.MaxNLocator(3))
    ax3.yaxis.set_major_locator(plt.MaxNLocator(3))
    ax3.legend(fontsize=fontsize)
    ax4.plot(time_ind, spike_rate_list_pfc, color="r")  # label="cv "+str(i+1)+"/10",
    # ax4.set_ylabel("spikes/s", fontsize=fontsize)
    ax4.set_xlabel("time", fontsize=fontsize)
    ax4.xaxis.set_major_locator(plt.MaxNLocator(3))
    ax4.yaxis.set_major_locator(plt.MaxNLocator(3))
    # ax1.tick_params(axis='both', which='major', labelsize=fontsize)
    # ax2.tick_params(axis='both', which='major', labelsize=fontsize)
    # ax3.tick_params(axis='both', which='major', labelsize=fontsize)
    # ax4.tick_params(axis='both', which='major', labelsize=fontsize)


    plt.show()
    plt.savefig(save_path)

    pass


