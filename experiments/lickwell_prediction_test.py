from src.database_api_beta import  Filter, hann, Net_data

from src.preprocessing import lickwells_io
from src.network_functions import initiate_lickwell_network, run_lickwell_network
from src.metrics import print_metric_details
from random import seed
if __name__ == '__main__':



    # Data set 1 Prefrontal Cortex

    # MODEL_PATH = "G:/master_datafiles/trained_networks/MLP_PFC_2019-02-20_dropout_20/"
    # RAW_DATA_PATH = "G:/master_datafiles/raw_data/PFC/"
    # FILTERED_DATA_PATH = "session_pfc_lw.pkl"

    # Data set 2 Hippocampus

    MODEL_PATH = "G:/master_datafiles/trained_networks/MLP_HC_2019-02-07_phase/"
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
        lw_classifications=4,
        lw_normalize=True,
        lw_differentiate_false_licks=False,
        num_wells=5,
        initial_timeshift=-1,
        from_raw_data=False,
        dropout=0.2,
        number_of_bins=11
    )
    MODEL_PATH = "C:/Users/NN/Desktop/Master/experiments/Lickwell_prediction/MLP_HC/"
    print_metric_details(MODEL_PATH,nd.initial_timeshift)
    session = initiate_lickwell_network(nd)  # Initialize session
    X, y,nd,session,_,_ = lickwells_io(session, nd, excluded_wells=[1], shift=nd.initial_timeshift,
                                  normalize=nd.lw_normalize,
                                  differentiate_false_licks=nd.lw_differentiate_false_licks,target_is_phase=True,
                                   lickstart=0,lickstop=5000)
    run_lickwell_network(nd, session, X, y)
    print("fin")
