from src.database_api_beta import  Filter, hann, Net_data

from src.preprocessing import lickwells_io
from src.network_functions import initiate_lickwell_network, run_lickwell_network
from src.metrics import print_metric_details
from random import seed
if __name__ == '__main__':



    # Data set 1 Prefrontal Cortex

    # MODEL_PATH = "G:/master_datafiles/trained_networks/MLP_PFC_2019-01-29_lickwell_filter/"
    # RAW_DATA_PATH = "G:/master_datafiles/raw_data/PFC/"
    # FILTERED_DATA_PATH = "session_pfc_lw.pkl"

    # Data set 2 Hippocampus

    MODEL_PATH = "G:/master_datafiles/trained_networks/MLP_HC_2019-01-30_lickwell"
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
        lw_classifications=5,
        lw_normalize=True,
        lw_differentiate_false_licks=False,
        num_wells=5,
        initial_timeshift=1,
        from_raw_data=False
    )

    # print_metric_details(MODEL_PATH,nd.initial_timeshift) # Uncomment to read output without having to rerun the program
    session = initiate_lickwell_network(nd)  # Initialize session
    X, y, metadata,nd,session = lickwells_io(session, nd, excluded_wells=[1], shift=nd.initial_timeshift,
                                  normalize=nd.lw_normalize,
                                  differentiate_false_licks=nd.lw_differentiate_false_licks,phase_test=False)
    for i,lick in enumerate(session.licks):
        if lick.lickwell==1:
            print(lick.lick_id,session.licks[i+nd.initial_timeshift].lickwell,lick.target)
    run_lickwell_network(nd, session, X, y, metadata)
    print("fin")
