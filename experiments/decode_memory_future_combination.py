from src.database_api_beta import  Net_data
import numpy as np

from src.network_functions import run_network_process, initiate_network, run_network


# Combination data set
if __name__ == '__main__':

    RAW_DATA_PATH = "G:/master_datafiles/raw_data/31-07-2018_hc"

    # Hippocampus

    MODEL_PATH = "G:/master_datafiles/trained_networks/MLP_CHC_2018-11-30_1000_200_100_lickwell/"
    FILTERED_DATA_PATH = "slice_CHC.pkl"
    filter_tetrodes=range(13,1000)

    # Prefrontal Cortex

    # MODEL_PATH = "G:/master_datafiles/trained_networks/MLP_CHC_2018-11-30_1000_200_100_lickwell/"
    # FILTERED_DATA_PATH = "slice_CPFC.pkl"
    # filter_tetrodes=range(0,13)

    # Combination

    # MODEL_PATH = "G:/master_datafiles/trained_networks/MLP_CHC_2018-11-30_1000_200_100_lickwell/"
    # FILTERED_DATA_PATH = "slice_C.pkl"
    # filter_tetrodes=range(0,13)
    nd = Net_data(
        INITIAL_TIMESHIFT=-30000,
        EPOCHS=20,
        TIME_SHIFT_ITER=500,
        TIME_SHIFT_STEPS=40,
        EARLY_STOPPING=False,
        MODEL_PATH=MODEL_PATH,
        RAW_DATA_PATH=RAW_DATA_PATH,
        K_CROSS_VALIDATION = 1
    )
    session = initiate_network(nd,load_raw_data=False)
    # x and y are switched for this session and are restored to their original order to make them identical to the other sessions
    copy_pos_x = session.position_x
    session.position_x = session.position_y
    session.position_y = copy_pos_x
    run_network(nd, session)
    # Create save file directories
