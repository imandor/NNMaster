from src.database_api_beta import  Filter, hann, Net_data

from src.preprocessing import lickwells_io
from src.network_functions import initiate_lickwell_network, run_lickwell_network
from src.metrics import print_metric_details
from random import seed
from experiments.tleave import determine_t_leave
import numpy as np
from plots.time_histogram import timehistogram_bar_values
import matplotlib.pyplot as plt

if __name__ == '__main__':
    # Determines time rat leaves area and trains for that + 5 preceding seconds


    # Data set 1 Prefrontal Cortex

    MODEL_PATH = "G:/master_datafiles/trained_networks/MLP_PFC_tleave/"
    RAW_DATA_PATH = "G:/master_datafiles/raw_data/PFC/"
    FILTERED_DATA_PATH = "session_pfc_lw.pkl"

    # Data set 2 Hippocampus

    # MODEL_PATH = "G:/master_datafiles/trained_networks/MLP_HC_tleave/"
    # RAW_DATA_PATH = "G:/master_datafiles/raw_data/HC/"
    # FILTERED_DATA_PATH = "session_hc_lw.pkl"

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
        initial_timeshift=1,
        from_raw_data=False,
        dropout=0.65,
        number_of_bins=10,

    )

    search_radius = 40
    fontsize = 18
    lickstart= -5000
    lickstop = 20000
    interval = 500
    xtickresolution = 2500
    save_path = nd.model_path + "images/lick_duration"+".png"
    # print_metric_details(MODEL_PATH,nd.initial_timeshift)
    session = initiate_lickwell_network(nd)  # Initialize session
    nd.start_time_by_lick_id =  determine_t_leave(session,search_radius)
    lick_ids = [x[0] for x in nd.start_time_by_lick_id]
    t_leave = [x[1] for x in nd.start_time_by_lick_id]
    lickids,entry_list,exit_list,counter_list,ind_labels = timehistogram_bar_values(nd,lickstart,lickstop,search_radius,interval)
    ind = np.arange(0,len(ind_labels))
    fig, (ax1,ax2) = plt.subplots(2)
    ax1.bar(ind_labels, counter_list, color='magenta', align='center',width=500)
    ax1.set_xticklabels(ind_labels)
    ax1.set_ylabel("fraction of occurring events", fontsize=fontsize)
    # ax1.set_xlabel("time", fontsize=fontsize)
    ax1.axhline(0.95)
    ax1.set_xticks(np.arange(min(ind_labels), max(ind_labels) + 1, xtickresolution))
    ax1.set_xticklabels(np.arange(min(ind_labels), max(ind_labels) + 1, xtickresolution),fontsize=fontsize)
    # ax1.set_yticklabels(np.arange(0, len(lickids) + 1, ytickresolution),fontsize=fontsize)
    ax1.legend(["HC","0.95"])
    ax2.bar(ind_labels, counter_list, color='magenta', align='center',width=interval)
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
