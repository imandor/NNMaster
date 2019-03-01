from src.database_api_beta import  Filter, hann, Net_data

from src.preprocessing import lickwells_io
from src.network_functions import initiate_lickwell_network, run_lickwell_network
from src.metrics import print_metric_details
from random import seed

def determine_t_leave(session,max_pos):
    time_by_lick_id = []
    for lick in session.licks:
        if lick.lickwell == 1:
            for i,pos in enumerate(session.position_x[int(lick.time):-1]):
                if pos>max_pos:
                    time_by_lick_id.append((lick.lick_id,i))
                    break
    return time_by_lick_id

if __name__ == '__main__':
    # Determines time rat leaves area and trains for that + 5 preceding seconds


    # Data set 1 Prefrontal Cortex

    MODEL_PATH = "G:/master_datafiles/trained_networks/MLP_PFC_tleave/"
    RAW_DATA_PATH = "G:/master_datafiles/raw_data/PFC/"
    FILTERED_DATA_PATH = "session_pfc_lw.pkl"

    # Data set 2 Hippocampus

    # MODEL_PATH = "G:/master_datafiles/trained_networks/MLP_HC_tleave/"
    # RAW_DATA_PATH = "G:/master_datafiles/raw_data/HC/"
    # FILTERED_DATA_PATH = "session_hc_lw.pkl"

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
        dropout=0.65,
        number_of_bins=10,

    )

    search_radius = 40
    # print_metric_details(MODEL_PATH,nd.initial_timeshift)
    session = initiate_lickwell_network(nd)  # Initialize session
    nd.start_time_by_lick_id =  determine_t_leave(session,search_radius)

    X, y,nd,session, = lickwells_io(session, nd, excluded_wells=[1], shift=nd.initial_timeshift,target_is_phase=False,
                                   lickstart=-5000,lickstop=0)
    run_lickwell_network(nd, session, X, y)

    print("fin")
