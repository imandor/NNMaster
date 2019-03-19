from src.database_api_beta import  Net_data
import numpy as np
from src.metrics import print_metric_details
from src.network_functions import run_network_process, initiate_network, run_network
# Filters neurons for different settings
if __name__ == '__main__':
    combination_data_set  = False
    filter_tetrodes = None

    # prefrontal cortex
    # MODEL_PATH = "G:/master_datafiles/trained_networks/pfc"
    # RAW_DATA_PATH = "G:/master_datafiles/raw_data/PFC/"
    # FILTERED_DATA_PATH = "session_pfc"

    # hippocampus
    #
    # MODEL_PATH = "G:/master_datafiles/trained_networks/hc"
    # RAW_DATA_PATH = "G:/master_datafiles/raw_data/HC/"
    # FILTERED_DATA_PATH = "session_hc"

    # Combination data set

    RAW_DATA_PATH = "G:/master_datafiles/raw_data/C"
    combination_data_set = True # for some reason, the combination data set has switched x and y axis, which needs to be manually switched back
    # only Hippocampus neurons

    # MODEL_PATH = "G:/master_datafiles/trained_networks/chc"
    # FILTERED_DATA_PATH = "session_CHC.pkl"
    # filter_tetrodes=range(13,28)

    # only Prefrontal Cortex neurons

    MODEL_PATH = "G:/master_datafiles/trained_networks/cpfc"
    FILTERED_DATA_PATH = "session_CPFC.pkl"
    filter_tetrodes=range(0,13)


    neuron_filter_list = [56,44,33,22,11,1]


    for neuron_filter in neuron_filter_list:
        model_path = MODEL_PATH + str(neuron_filter) + "/"
        nd = Net_data(
            initial_timeshift=0,
            time_shift_iter=500,
            time_shift_steps=1,
            keep_neurons=neuron_filter,
            early_stopping=False,
            model_path=model_path,
            raw_data_path=RAW_DATA_PATH,
            filtered_data_path=FILTERED_DATA_PATH,
            k_cross_validation=10,
            valid_ratio=0.1,
            naive_test=False,
            from_raw_data=False,
            epochs=30,
            dropout=0.65,
            behavior_component_filter=None,
            filter_tetrodes=filter_tetrodes,
            shuffle_data=True,
            shuffle_factor=10,
            batch_size=50,
            switch_x_y=combination_data_set
        )
        session = initiate_network(nd)
        run_network(nd, session)
        # Create save file directories