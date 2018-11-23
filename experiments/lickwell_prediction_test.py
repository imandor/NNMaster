from src.database_api_beta import Slice, Filter, hann, Net_data

from src.preprocessing import lickwells_io,get_all_valid_licks
from src.network_functions import run_network_process, initiate_lickwell_network, run_lickwell_network

if __name__ == '__main__':
    # prefrontal cortex

    # MODEL_PATH = "G:/master_datafiles/trained_networks/MLP_PFC_2018-11-13_1000_200_100_lickwell_normalized/"
    # RAW_DATA_PATH = "G:/master_datafiles/raw_data/2018-04-09_14-39-52/"
    # FILTERED_DATA_PATH = "slice_PFC_200.pkl"

    # hippocampus

    MODEL_PATH = "G:/master_datafiles/trained_networks/MLP_HC_2018-11-22_1000_200_100_lickwell/"
    RAW_DATA_PATH = "G:/master_datafiles/raw_data/2018-05-16_17-13-37/"
    FILTERED_DATA_PATH = "slice_HC_200.pkl"

    nd = Net_data(

        # Program execution settings
        EPOCHS=10,
        evaluate_training=False,
        SLICE_SIZE=200,
        STRIDE=200,
        Y_STEP=200,
        WIN_SIZE=200,
        SEARCH_RADIUS=200,
        K_CROSS_VALIDATION=10,
        session_filter=Filter(func=hann, search_radius=200, step_size=200),
        VALID_RATIO=0.1,
        testing_ratio = 0,
        TIME_SHIFT_STEPS=1,
        EARLY_STOPPING=False,
        MODEL_PATH=MODEL_PATH,
        RAW_DATA_PATH=RAW_DATA_PATH,
        FILTERED_DATA_PATH = FILTERED_DATA_PATH,
        metric="discrete",
        SHUFFLE_DATA=True,
        SHUFFLE_FACTOR=1,
        lw_classifications=5,
        lw_normalize=True,
        lw_differentiate_false_licks=False,
        num_wells = 5,
        INITIAL_TIMESHIFT=1,
    )

    session = initiate_lickwell_network(nd) # Initialize session
    valid_licks = get_all_valid_licks(session,start_well = 1,change_is_valid=True)
    X, y = lickwells_io(session, nd, excluded_wells=[1], shift=nd.INITIAL_TIMESHIFT, normalize=nd.lw_normalize,
                        differentiate_false_licks=nd.lw_differentiate_false_licks)
    run_lickwell_network(nd, session,X,y)

    print("fin")
