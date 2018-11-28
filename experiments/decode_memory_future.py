from src.database_api_beta import  Net_data
import numpy as np

from src.network_functions import run_network_process, initiate_network, run_network

if __name__ == '__main__':

    # prefrontal cortex

    # MODEL_PATH = "G:/master_datafiles/trained_networks/MLP_PFC_2018-11-06_1000_200_100_dmf/"
    # RAW_DATA_PATH = "G:/master_datafiles/raw_data/2018-04-09_14-39-52/"
    # FILTERED_DATA_PATH = "G:/master_datafiles/filtered_data/neocortex_hann_win_size_20.pkl"

    # hippocampus

    MODEL_PATH = "G:/master_datafiles/trained_networks/MLP_HC_2018-11-13_1000_200_100_dmf/"
    RAW_DATA_PATH = "G:/master_datafiles/raw_data/2018-05-16_17-13-37/"
    FILTERED_DATA_PATH = "G:/master_datafiles/filtered_data/hippocampus_hann_win_size_25_09-5_7.pkl"

    nd = Net_data(
        INITIAL_TIMESHIFT=0,
        EPOCHS=30,
        TIME_SHIFT_STEPS=1,
        EARLY_STOPPING=True,
        MODEL_PATH=MODEL_PATH,
        RAW_DATA_PATH=RAW_DATA_PATH


    )
    session = initiate_network(nd)


    run_network(nd, session)
    # Create save file directories
