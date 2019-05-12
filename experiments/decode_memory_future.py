from src.database_api_beta import  Net_data
import numpy as np
from src.metrics import print_metric_details
from src.network_functions import run_network_process, initiate_network, run_network
from src.model_data import c_dmf,chc_dmf,cpfc_dmf,hc_dmf,pfc_dmf
if __name__ == '__main__':

    model_data = pfc_dmf
    # model_data = chc_dmf

    model_data.model_path="G:/master_datafiles/trained_networks/pfc/"
    nd = Net_data(
        initial_timeshift=-10000,
        time_shift_iter=500,
        time_shift_steps=41,
        early_stopping=False,
        model_path=model_data.model_path,
        raw_data_path=model_data.raw_data_path,
        filtered_data_path=model_data.filtered_data_path,
        k_cross_validation = 10,
        valid_ratio=0.1,
        naive_test=False,
        from_raw_data=False,
        epochs = 15,
        dropout=0.65,
        # behavior_component_filter="rest",
        # behavior_component_filter="not at lickwell",
        # behavior_component_filter="correct trials",
        # behavior_component_filter="incorrect trials",
        # behavior_component_filter="move",

        filter_tetrodes=model_data.filter_tetrodes,
        shuffle_data=True,
        shuffle_factor=10,
        batch_size=50,
        switch_x_y=model_data.switch_x_y
    )
    session = initiate_network(nd)
    licks_timestamp = [lick.time for lick in session.licks]
    # session.plot(ax_filtered_spikes=session.filtered_spikes, ax_raw_spikes=session.spikes, ax_licks=licks_timestamp)
    # speedlist = [] # Speed test
    # maxspeed = 100 # np.max(session.speed) # max speed is not identical across data sets, not directly comparable
    # for i, speed in enumerate(session.speed):
    #     if speed<0:
    #         speedlist.append(0)
    #     else:
    #         speedbin = int(79*speed/maxspeed)
    #         if speedbin>=80:
    #             speedbin = 79
    #         speedlist.append(speedbin)
    # session.position_x = speedlist
    run_network(nd, session)

