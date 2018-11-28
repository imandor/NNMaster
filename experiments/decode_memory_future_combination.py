from src.database_api_beta import  Net_data
import numpy as np

from src.network_functions import run_network_process, initiate_network, run_network


# Combination data set
if __name__ == '__main__':

    # Hippocampus

    # MODEL_PATH = "G:/master_datafiles/trained_networks/MLP_CHC_2018-11-28_1000_200_100_lickwell/"
    # RAW_DATA_PATH = "G:/master_datafiles/raw_data/31-07-2018_hc"
    # FILTERED_DATA_PATH = "slice_CHC_200.pkl"

    # Prefrontal Cortex

    MODEL_PATH = "G:/master_datafiles/trained_networks/MLP_CHC_2018-11-28_1000_200_100_lickwell/"
    RAW_DATA_PATH = "G:/master_datafiles/raw_data/31-07-2018_pfc"
    FILTERED_DATA_PATH = "slice_CPFC_200.pkl"

    nd = Net_data(
        INITIAL_TIMESHIFT=0,
        EPOCHS=30,
        TIME_SHIFT_STEPS=1,
        EARLY_STOPPING=True,
        MODEL_PATH=MODEL_PATH,
        RAW_DATA_PATH=RAW_DATA_PATH

    )
    session = initiate_network(nd)
    # x and y are switched for this session and are restored to their original order to make them identical to the other sessions
    copy_pos_x = session.position_x
    session.position_x = session.position_y
    session.position_y = copy_pos_x
    run_network(nd, session)
    # Create save file directories
