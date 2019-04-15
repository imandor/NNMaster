# Calculates chance level ape


from src.database_api_beta import  Net_data
import numpy as np
from src.metrics import print_metric_details
from src.network_functions import run_network_process, initiate_network, run_network
from src.model_data import c_dmf,chc_dmf,cpfc_dmf,hc_dmf,pfc_dmf
from src.preprocessing import time_shift_positions
if __name__ == '__main__':

    model_data = hc_dmf
    model_data.model_path="G:/master_datafiles/trained_networks/speedtest/"
    nd = Net_data(
        initial_timeshift=0,
        time_shift_iter=500,
        time_shift_steps=1,
        early_stopping=False,
        model_path=model_data.model_path,
        raw_data_path=model_data.raw_data_path,
        filtered_data_path=model_data.filtered_data_path,
        k_cross_validation = 10,
        valid_ratio=0.1,
        naive_test=False,
        from_raw_data=False,
        epochs = 30,
        dropout=0.65,
        # behavior_component_filter="at lickwell",
        # behavior_component_filter="not at lickwell",
        # behavior_component_filter="correct trials",
        # behavior_component_filter="incorrect trials",

        filter_tetrodes=model_data.filter_tetrodes,
        shuffle_data=True,
        shuffle_factor=10,
        batch_size=50,
        switch_x_y=model_data.switch_x_y
    )
    session = initiate_network(nd)
    X, y = time_shift_positions(session, 0, nd)
    y_list = []
    for y_i in y:
        y_list.append(np.argmax(y_i)*nd.y_step)
    y_avg = np.average(y_list)
    avg_distance = 0
    for i,y_i in enumerate(y_list):
        avg_distance+= abs(y_avg-y_i)
    avg_distance = avg_distance/len(y)
    print(avg_distance)
