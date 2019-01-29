from src.database_api_beta import  Filter, hann, Net_data

from src.preprocessing import lickwells_io
from src.plots import plot_performance_by_licktime,plot_position_by_licktime
from src.network_functions import initiate_lickwell_network, run_lickwell_network
from src.metrics import print_metric_details
from random import seed
import numpy as np
if __name__ == '__main__':

# Removes a portion of the lick events and checks if performance changes

    # Data set 1 Prefrontal Cortex

    # MODEL_PATH = "G:/master_datafiles/trained_networks/MLP_PFC_2019-01-29_lickwell_licktimetest/"
    # RAW_DATA_PATH = "G:/master_datafiles/raw_data/PFC/"
    # FILTERED_DATA_PATH = "session_pfc_lw.pkl"

    # Data set 2 Hippocampus

    MODEL_PATH = "G:/master_datafiles/trained_networks/MLP_HC_2019-01-29_lickwell_licktimetest_filter/"
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
        lw_classifications=5,
        lw_normalize=True,
        lw_differentiate_false_licks=False,
        num_wells=5,
        initial_timeshift=1,
        from_raw_data=False
    )

    seed(0)
    plotrange = range(0,28)
    # plot_performance_by_licktime(path=MODEL_PATH + "output/", shift=1,save_path=MODEL_PATH+"images/by_licktime_next.png",
    #                              add_trial_numbers=True,title="Fraction decoded correctly by time into lick-event",
    #                              plotrange=plotrange)
    session = initiate_lickwell_network(nd)  # Initialize session
    X, y, metadata,nd,session = lickwells_io(session, nd, excluded_wells=[1], shift=nd.initial_timeshift,
                                  normalize=nd.lw_normalize,
                                  differentiate_false_licks=nd.lw_differentiate_false_licks)
    # for i, lick in session.licks: # only uncomment during phase test
    #     if lick.target!=1:
    #         lick.target = lick.phase


    # plot_position_by_licktime(session,y,metadata,plotrange,title="Current position and at +-2.5 seconds during lick at well 1",save_path="asd")

    for z in plotrange:
        offset = z
        lower_border = 0 + offset
        upper_border = 11 + offset

        X_star = []
        y_star = []
        metadata_star = []
        licks = []
        for i,y_i in enumerate(y):
            j = i%39
            if j>=lower_border and j <upper_border:
                X_star.append(X[i])
                y_star.append(y_i)
                metadata_star.append(metadata[i])
        for lick in session.licks:
            if lick.lickwell == 1 and lick.lick_id!=1:
                timestart = lick.time +(lower_border+offset)*(5000/39)
                timestop = lick.time +(upper_border+offset-1)*(5000/39)
        run_lickwell_network(nd, session, X_star, y_star, metadata_star,pathname_metadata="_"+str(z))

