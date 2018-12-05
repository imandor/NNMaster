from src.database_api_beta import Slice, Filter, hann, Net_data
import os
import errno
import datetime

from src.network_functions import run_network_process,initiate_network,run_network

if __name__ == '__main__':

    # prefrontal cortex

    MODEL_PATH = "G:/master_datafiles/trained_networks/MLP_PFC_2018-11-11_1000_200_100_naive/"
    RAW_DATA_PATH = "G:/master_datafiles/raw_data/2018-04-09_14-39-52/"
    FILTERED_DATA_PATH = "G:/master_datafiles/filtered_data/neocortex_hann_win_size_20.pkl"

    # hippocampus

    # MODEL_PATH = "G:/master_datafiles/trained_networks/MLP_HC_2018-11-08_1000_200_100_naive/"
    # RAW_DATA_PATH = "G:/master_datafiles/raw_data/2018-05-16_17-13-37/"
    # FILTERED_DATA_PATH = "G:/master_datafiles/filtered_data/hippocampus_hann_win_size_25_09-5_7.pkl"
    NEURONS_KEPT_FACTOR = 1

    WIN_SIZE = 20
    SEARCH_RADIUS = WIN_SIZE * 2
    session_filter = Filter(func=hann, search_radius=SEARCH_RADIUS, step_size=WIN_SIZE)

    nd = Net_data(

        # Program execution settings

        EPOCHS=30,
        SEARCH_RADIUS=SEARCH_RADIUS,
        WIN_SIZE=WIN_SIZE,
        INITIAL_TIMESHIFT=0,
        TIME_SHIFT_ITER=-200,
        TIME_SHIFT_STEPS=20,
        METRIC_ITER=1,  # after how many epochs network is validated <---
        SHUFFLE_DATA=True,  # whether to randomly shuffle the data in big slices
        SHUFFLE_FACTOR=500,
        EARLY_STOPPING=False,
        NAIVE_TEST=True,
        K_CROSS_VALIDATION=1,
        TRAIN_MODEL=True,

        # Input data parameters

        SLICE_SIZE=1000,
        Y_SLICE_SIZE=200,
        STRIDE=100,
        BATCH_SIZE=50,
        VALID_RATIO=0.1,
        X_MAX=240,
        Y_MAX=190,
        X_MIN=0,
        Y_MIN=100,
        X_STEP=3,
        y_step=3,
        LOAD_MODEL= False,
        session_filter=session_filter,
        MODEL_PATH=MODEL_PATH,
        r2_scores_train=[],
        r2_scores_valid=[],
        acc_scores_train=[],
        acc_scores_valid=[],
        avg_scores_train=[],
        avg_scores_valid=[],
        RAW_DATA_PATH=RAW_DATA_PATH,
    )
    session = initiate_network(nd)
    run_network(nd,session)
    # Create save file directories
