from src.database_api_beta import Slice, Filter, hann, Net_data

from src.preprocessing import lickwells_io
from src.network_functions import run_network_process, initiate_lickwell_network, run_lickwell_network
from src.metrics import print_metric_details,print_lickwell_metrics
if __name__ == '__main__':



    # Data set 1 Prefrontal Cortex

    MODEL_PATH = "G:/master_datafiles/trained_networks/MLP_PFC_2019-01-08_lickwell_testversion/"
    RAW_DATA_PATH = "G:/master_datafiles/raw_data/2018-04-09_14-39-52/"
    FILTERED_DATA_PATH = "session_pfc_lw.pkl"

    # Data set 2 Hippocampus

    # MODEL_PATH = "G:/master_datafiles/trained_networks/MLP_HC_2018-12-29_lickwell/"
    # RAW_DATA_PATH = "G:/master_datafiles/raw_data/2018-05-16_17-13-37/"
    # FILTERED_DATA_PATH = "session_hc_lw.pkl"



    nd = Net_data(

        # Program execution settings
        epochs=10,
        evaluate_training=False,
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
        initial_timeshift=-1,
    )

    path = MODEL_PATH + "output/"
    print_metric_details(path,nd.initial_timeshift)
    session = initiate_lickwell_network(nd)  # Initialize session
    X, y, metadata,nd = lickwells_io(session, nd, excluded_wells=[1], shift=nd.initial_timeshift,
                                  normalize=nd.lw_normalize,
                                  differentiate_false_licks=nd.lw_differentiate_false_licks)

    run_lickwell_network(nd, session, X, y, metadata)

    print("fin")
