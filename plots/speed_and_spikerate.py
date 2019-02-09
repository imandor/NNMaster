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





def timehistogram_bar_values(nd,lickstart,lickstop,searchradius,interval):
    # Determines lists of times of entry and exit for plot
    entry_list = []
    exit_list = []
    lickids = []
    session = initiate_lickwell_network(nd)  # Initialize session
    for lick in session.licks:
        if lick.lickwell == 1:
            timeslice = session[int(lick.time+lickstart):int(lick.time+lickstop)]
            enter_time = None
            exit_time = None
            for time, position in enumerate(timeslice.position_x):
                is_before_lick = (time<-lickstart) # determines if current time is before lick event
                if not is_before_lick and enter_time is None: # sets time when rat enters vicinity of well 1
                    enter_time = 0
                    print("Warning: there seems to be a problem with the position of the rat during a lick at well 1")
                if (position<searchradius or time==lickstart) and enter_time is None: # determine when rat enters area
                    enter_time = time
                if position>searchradius and exit_time is None:
                    if is_before_lick: # rat leaves area before lick
                        enter_time = None
                    else: # set exit time
                        exit_time=time
                if exit_time is None and time==lickstop-lickstart-1:
                    print("Warning: the rat does not leave the vicinity inside the given time range")
                    exit_time = lickstop-lickstart
                if enter_time is not None and exit_time is not None:
                    entry_list.append(enter_time+lickstart)
                    exit_list.append(exit_time+lickstart)
                    lickids.append(lick.lick_id)
                    break
    # determine what percentage of licks are in range for histogram
    counter_list = []
    ind_labels = []
    for i in range(-5000,15000,interval):
        counter = 0
        for j in range(0,len(lickids)):
            if i>=entry_list[j] and i<exit_list[j]:
                counter += 1
        counter_list.append(counter/len(lickids))
        ind_labels.append(i)

    return lickids,entry_list,exit_list,counter_list,ind_labels


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

    lickstart= -5000
    lickstop = 30000
    searchradius = 40
    xtickresolution = 2500
    # ytickresolution = 10
    fontsize = 16
    interval = 500
    image_title_list = ["pfc_phase","hc_phase","pfc","hc"]
    path = nd.model_path+ "output/"
    save_path = nd.model_path + "images/lick_duration"+".png"


    lickids,entry_list,exit_list,counter_list,ind_labels = timehistogram_bar_values(nd,lickstart,lickstop,searchradius,interval)
    # plot results
    ind = np.arange(0,len(ind_labels))
    fig, (ax1,ax2) = plt.subplots(2)
    ax1.bar(ind_labels, counter_list, color='lightblue', align='center',width=500)
    ax1.set_xticklabels(ind_labels)
    ax1.set_ylabel("fraction of occurring events", fontsize=fontsize)
    # ax1.set_xlabel("time", fontsize=fontsize)
    ax1.axhline(0.95)
    ax1.set_xticks(np.arange(min(ind_labels), max(ind_labels) + 1, xtickresolution))
    ax1.set_xticklabels(np.arange(min(ind_labels), max(ind_labels) + 1, xtickresolution),fontsize=fontsize)
    # ax1.set_yticklabels(np.arange(0, len(lickids) + 1, ytickresolution),fontsize=fontsize)
    ax1.legend(["HC","0.95"])

    nd.model_path = "G:/master_datafiles/trained_networks/MLP_PFC_2019-02-07_phase/"
    nd.raw_data_path = "G:/master_datafiles/raw_data/PFC/"
    nd.filtered_data_path = "session_pfc_lw.pkl"


    lickids,entry_list,exit_list,counter_list,ind_labels = timehistogram_bar_values(nd,lickstart,lickstop,searchradius,interval)
    ax2.bar(ind_labels, counter_list, color='lightblue', align='center',width=interval)
    ax2.set_xticklabels(ind_labels)
    ax2.set_ylabel("fraction of occurring events", fontsize=fontsize)
    ax2.set_xlabel("time", fontsize=fontsize)
    ax2.axhline(0.95)
    ax2.set_xticks(np.arange(min(ind_labels), max(ind_labels) + 1, xtickresolution))
    ax2.set_xticklabels(np.arange(min(ind_labels), max(ind_labels) + 1, xtickresolution),fontsize=fontsize)
    # ax2.set_yticklabels(np.arange(0, len(lickids) + 1, ytickresolution),fontsize=fontsize)
    ax2.legend(["PFC","0.95"])
    plt.show()
    plt.savefig(save_path)

    pass


