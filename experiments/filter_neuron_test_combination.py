from src.database_api_beta import  Net_data
import numpy as np
from src.metrics import print_metric_details
from src.network_functions import run_network_process, initiate_network, run_network

if __name__ == '__main__':

    RAW_DATA_PATH = "G:/master_datafiles/raw_data/C"

    # Hippocampus

    MODEL_PATH = "G:/master_datafiles/trained_networks/filter_neurons_chc_80/"
    FILTERED_DATA_PATH = "session_CHC.pkl"
    filter_tetrodes=range(13,1000)

    # Prefrontal Cortex

    # MODEL_PATH = "G:/master_datafiles/trained_networks/filter_neurons_cpfc_100/"
    # FILTERED_DATA_PATH = "session_CPFC.pkl"
    # filter_tetrodes=range(0,13)

    # Combination

    # MODEL_PATH = "G:/master_datafiles/trained_networks/filter_neurons_c_100/"
    # FILTERED_DATA_PATH = "slice_C.pkl"
    # filter_tetrodes=None

    nd = Net_data(
        initial_timeshift=-5000,
        filter_tetrodes=filter_tetrodes,
        time_shift_iter=500,
        time_shift_steps=21,
        early_stopping=False,
        model_path=MODEL_PATH,
        raw_data_path=RAW_DATA_PATH,
        k_cross_validation = 1,
        naive_test=False,
        from_raw_data=True,
        epochs = 10,
        neurons_kept_factor=0.8
    )
    session = initiate_network(nd)
    copy_pos_x = session.position_x
    session.position_x = session.position_y
    session.position_y = copy_pos_x
    run_network(nd, session)