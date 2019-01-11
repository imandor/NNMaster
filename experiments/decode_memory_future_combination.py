from src.database_api_beta import  Net_data
import numpy as np

from src.network_functions import run_network_process, initiate_network, run_network


# Combination data set
if __name__ == '__main__':

    RAW_DATA_PATH = "G:/master_datafiles/raw_data/31-07-2018_hc"

    # Hippocampus

    # MODEL_PATH = "G:/master_datafiles/trained_networks/MLP_CHC_2018-11-30_1000_200_100_lickwell/"
    # FILTERED_DATA_PATH = "slice_CHC.pkl"
    # filter_tetrodes=range(13,1000)

    # Prefrontal Cortex

    # MODEL_PATH = "G:/master_datafiles/trained_networks/MLP_CHC_2018-11-30_1000_200_100_lickwell/"
    # FILTERED_DATA_PATH = "slice_CPFC.pkl"
    # filter_tetrodes=range(0,13)

    # Combination

    MODEL_PATH = "G:/master_datafiles/trained_networks/no_shuffle_test_3_different_dataset/"

    FILTERED_DATA_PATH = "slice_C.pkl"
    filter_tetrodes=None
    nd = Net_data(
        initial_timeshift=-30000,
        epochs=20,
        time_shift_iter=500,
        time_shift_steps=1,
        early_stopping=False,
        model_path=MODEL_PATH,
        raw_data_path=RAW_DATA_PATH,
        k_cross_validation = 10
    )
    session = initiate_network(nd,load_raw_data=False)
    # x and y are switched for this session and are restored to their original order to make them identical to the other sessions
    copy_pos_x = session.position_x
    session.position_x = session.position_y
    session.position_y = copy_pos_x
    run_network(nd, session)
    # Create save file directories