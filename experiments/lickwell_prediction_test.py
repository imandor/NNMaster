from src.database_api_beta import Slice, Filter, hann, Net_data
import os
import errno
import datetime
from src.preprocessing import lickwells_io
from src.network_functions import run_network_process,initiate_network,run_network

if __name__ == '__main__':

    # prefrontal cortex

    # MODEL_PATH = "G:/master_datafiles/trained_networks/MLP_PFC_2018-11-06_1000_200_100_dmf/"
    # RAW_DATA_PATH = "G:/master_datafiles/raw_data/2018-04-09_14-39-52/"
    # FILTERED_DATA_PATH = "G:/master_datafiles/filtered_data/neocortex_hann_win_size_20.pkl"

    # hippocampus

    MODEL_PATH = "G:/master_datafiles/trained_networks/MLP_HC_2018-11-13_1000_200_100_dmf/"
    RAW_DATA_PATH = "G:/master_datafiles/raw_data/2018-05-16_17-13-37/"
    FILTERED_DATA_PATH = "G:/master_datafiles/filtered_data/hippocampus_hann_win_size_25_09-5_7.pkl"
    NEURONS_KEPT_FACTOR = 1

    WIN_SIZE = 20
    SEARCH_RADIUS = WIN_SIZE * 2
    session_filter = Filter(func=hann, search_radius=SEARCH_RADIUS, step_size=WIN_SIZE)

    nd = Net_data(

        # Program execution settings
        EPOCHS = 100,
        TIME_SHIFT_STEPS = 1,
        EARLY_STOPPING = False,
    )
X, y, session = initiate_network(nd)

X, y = lickwells_io(session,X, nd, allowed_distance=30,filter=1)
run_lickwell_network(X, y, nd,session)
print("fin")
