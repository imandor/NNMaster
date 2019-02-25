from src.database_api_beta import  Filter, hann, Net_data

from src.preprocessing import lickwells_io,shuffle_list_key
from src.network_functions import initiate_lickwell_network, run_lickwell_network
from random import seed
if __name__ == '__main__':

# Removes a portion of the lick events and checks if performance changes

    # Data set 1 Prefrontal Cortex

    MODEL_PATH = "G:/master_datafiles/trained_networks/MLP_PFC_2019-02-23_licktime/"
    RAW_DATA_PATH = "G:/master_datafiles/raw_data/PFC/"
    FILTERED_DATA_PATH = "session_pfc_lw.pkl"

    # Data set 2 Hippocampus

    # MODEL_PATH = "G:/master_datafiles/trained_networks/MLP_HC_2019-02-22_licktime/"
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
        k_cross_validation=1,
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
        initial_timeshift=1,
        from_raw_data=False,
        dropout=0.65,
    )
    lickstart = -5000
    lickstop = 10000
    seed(0)
    plotrange = range(0,int((lickstop-lickstart)/nd.win_size)-nd.number_of_bins) # number of possible partitions of lick event slices

    session = initiate_lickwell_network(nd)  # Initialize session

    X, y, nd, session = lickwells_io(session, nd, excluded_wells=[1], shift=nd.initial_timeshift,
                                  normalize=nd.lw_normalize,
                                  differentiate_false_licks=nd.lw_differentiate_false_licks,target_is_phase=True,
                                   lickstart=lickstart,lickstop=lickstop)
    # for i, lick in session.licks: # only uncomment during phase test
    #     if lick.target!=1:
    #         lick.target = lick.phase


    # plot_position_by_licktime(session,y,metadata,plotrange,title="Current position and at +-2.5 seconds during lick at well 1",save_path="asd")

    for z in plotrange:
        offset = z
        lower_border = z*nd.win_size # borders in ms of start of sliced event
        upper_border = (z+20)*nd.win_size

        X_star = []
        y_star = []
        for i,x in enumerate(X):
            X_star.append(x[lower_border:upper_border])
            # if samplestarts[i]>=lower_border and samplestarts[i]<upper_border:
            #     X_star.append(X[i])
            #     y_star.append(y_i)
        print("Border is now",str(lickstart + lower_border),str(lickstart + upper_border))


        run_lickwell_network(nd, session, X_star, y,pathname_metadata="_"+str(z))

