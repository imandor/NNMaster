from src.database_api_beta import  Net_data
import numpy as np
from src.preprocessing import lickwells_io
from src.network_functions import  initiate_lickwell_network
import seaborn as sn
import pandas as pd
import matplotlib.pyplot as plt


if __name__ == '__main__':

    RAW_DATA_PATH = "G:/master_datafiles/raw_data/C"
    MODEL_PATH = "G:/master_datafiles/trained_networks/no_shuffle_test_3_different_dataset/"
    FILTERED_DATA_PATH = "slice_C.pkl"
    filter_tetrodes=None

    nd = Net_data(
        initial_timeshift=0,
        epochs=1,
        time_shift_iter=500,
        time_shift_steps=1,
        early_stopping=False,
        model_path=MODEL_PATH,
        raw_data_path=RAW_DATA_PATH,
        filtered_data_path=FILTERED_DATA_PATH,
        k_cross_validation = 10,
        from_raw_data=False
    )

    session = initiate_lickwell_network(nd)  # Initialize session

    # x and y are switched for this session and are restored to their original order to make them identical to the other sessions
    copy_pos_x = session.position_x
    session.position_x = session.position_y
    session.position_y = copy_pos_x
    for lick in session.licks:
        print(lick.lickwell)

    array = np.zeros((10,10))
    for i,lick in enumerate(session.licks):
        x = lick.lickwell - 1
        if lick.target is not None:
            y = lick.target - 1
            array[x][y] += 1
    df_cm = pd.DataFrame(array, index = [i for i in [1,2,3,4,5,6,7,8,9,10]],
                      columns = [i for i in [1,2,3,4,5,6,7,8,9,10]])
    plt.figure(figsize = (10,7))
    sn.heatmap(df_cm, annot=True)
    plt.show()
    pass