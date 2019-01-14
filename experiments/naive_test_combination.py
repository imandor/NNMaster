from src.database_api_beta import Net_data
from src.network_functions import initiate_network, run_network

if __name__ == '__main__':

    RAW_DATA_PATH = "G:/master_datafiles/raw_data/31-07-2018_hc"

    # Hippocampus

    # MODEL_PATH = "G:/master_datafiles/trained_networks/MLP_CHC_2018-11-30_1000_200_100_lickwell/"
    # FILTERED_DATA_PATH = "slice_CHC.pkl"
    # filter_tetrodes=range(13,1000)

    # Prefrontal Cortex

    # MODEL_PATH = "G:/master_datafiles/trained_networks/MLP_CHC_2018-11-30_1000_200_100_lickwell/"
    # FILTERED_DATA_PATH = "slice_CPFC.pkl"
    # filter_tetrodes=range(0,13)

    # Combination

    MODEL_PATH = "G:/master_datafiles/trained_networks/no_shuffle_test_3_different_dataset/"

    nd = Net_data(
        initial_timeshift=0,
        time_shift_iter=-500,
        time_shift_steps=11,
        early_stopping=False,
        model_path=MODEL_PATH,
        raw_data_path=RAW_DATA_PATH,
        k_cross_validation = 1,
        naive_test=True,
        session_from_raw=False,
        epochs = 10


    )

    session = initiate_network(nd) # x and y are switched for this session and are restored to their original order to make them identical to the other sessions
    copy_pos_x = session.position_x
    session.position_x = session.position_y
    session.position_y = copy_pos_x
    run_network(nd, session)

