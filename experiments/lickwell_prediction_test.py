from src.database_api_beta import Slice, Filter, hann, Net_data

from src.preprocessing import lickwells_io
from src.network_functions import run_network_process, initiate_lickwell_network, run_lickwell_network
from src.metrics import print_metric_details
if __name__ == '__main__':



    # Data set 3 raw data path
    RAW_DATA_PATH = "G:/master_datafiles/raw_data/31-07-2018_hc"
    filter_tetrodes=None
    # Data set 1 Prefrontal Cortex

    # MODEL_PATH = "G:/master_datafiles/trained_networks/MLP_PFC_2018-12-17_lickwell/"
    # RAW_DATA_PATH = "G:/master_datafiles/raw_data/2018-04-09_14-39-52/"
    # FILTERED_DATA_PATH = "session_pfc_lw.pkl"

    # Data set 2 Hippocampus

    MODEL_PATH = "G:/master_datafiles/trained_networks/MLP_HC_2018-12-29_lickwell/"
    RAW_DATA_PATH = "G:/master_datafiles/raw_data/2018-05-16_17-13-37/"
    FILTERED_DATA_PATH = "session_hc_lw.pkl"



    # Data set 3 Hippocampus
    # MODEL_PATH = "G:/master_datafiles/trained_networks/MLP_CHC_2018-12-11_lickwell/"
    # FILTERED_DATA_PATH = "session_chc_lw.pkl"
    # filter_tetrodes = range(13, 1000)

    # data set 3 Prefrontal Cortex

    # MODEL_PATH = "G:/master_datafiles/trained_networks/MLP_CHC_2018-11-30_1000_200_100_lickwell/"
    # FILTERED_DATA_PATH = "slice_CPFC.pkl"
    # filter_tetrodes=range(0,13)

    # data set 3 Combination

    # MODEL_PATH = "G:/master_datafiles/trained_networks/MLP_CHC_2018-11-30_1000_200_100_lickwell/"
    # FILTERED_DATA_PATH = "slice_C.pkl"
    # filter_tetrodes=range(0,13)


    nd = Net_data(

        # Program execution settings
        epochs=10,
        evaluate_training=False,
        filter_tetrodes=filter_tetrodes,
        slice_size=200,
        stride=200,
        y_step=200,
        win_size=200,
        search_radius=200,
        k_cross_validation=10,
        session_filter=Filter(func=hann, search_radius=200, step_size=200),
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
    )
    session = initiate_lickwell_network(nd,load_raw_data=False)  # Initialize session
    X, y, metadata,nd = lickwells_io(session, nd, excluded_wells=[1], shift=nd.initial_timeshift,
                                  normalize=nd.lw_normalize,
                                  differentiate_false_licks=nd.lw_differentiate_false_licks)
    path = "G:/master_datafiles/trained_networks/MLP_HC_2018-12-29_lickwell/output/network_output_timeshift=1.pkl"
    print_metric_details(path,nd)
    # run_lickwell_network(nd, session, X, y, metadata)

    print("fin")
