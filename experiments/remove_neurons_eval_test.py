from src.database_api_beta import  Net_data
import numpy as np
from src.metrics import print_metric_details
from src.network_functions import run_network_process, initiate_network, run_network

if __name__ == '__main__':

    # prefrontal cortex

    MODEL_PATH = "G:/master_datafiles/trained_networks/remove_neurons_pfc/"
    RAW_DATA_PATH = "G:/master_datafiles/raw_data/2018-04-09_14-39-52/"
    FILTERED_DATA_PATH = "session_pfc"

    # hippocampus

    # MODEL_PATH = "G:/master_datafiles/trained_networks/remove_neurons_hc/"
    # RAW_DATA_PATH = "G:/master_datafiles/raw_data/2018-05-16_17-13-37/"
    # FILTERED_DATA_PATH = "session_hc"

    nd = Net_data(
        initial_timeshift=-5000,
        time_shift_iter=500,
        time_shift_steps=21,
        early_stopping=False,
        model_path=MODEL_PATH,
        raw_data_path=RAW_DATA_PATH,
        k_cross_validation = 1,
        from_raw_data=False,
        epochs = 10,
        neurons_kept_factor=0.8
    )
    session = initiate_network(nd)

    run_network(nd, session)
    # Create save file directories
