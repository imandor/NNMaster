from src.database_api_beta import  Filter, hann, Net_data

from src.preprocessing import lickwells_io
from src.network_functions import initiate_lickwell_network, run_lickwell_network
from src.metrics import print_metric_details
from random import seed
from src.model_data import hc_lw,pfc_lw
import numpy as np
if __name__ == '__main__':

    # finds preferred spike rate for each neuron
    model_data = pfc_lw
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
        model_path=model_data.model_path,
        raw_data_path=model_data.raw_data_path,
        filtered_data_path=model_data.filtered_data_path,
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
        number_of_bins=10,
    )
    # print_metric_details("C:/Users/NN/Desktop/Master/experiments/Experiments for thesis 2/well decoding/hc/",1)
    # print_metric_details("C:/Users/NN/Desktop/Master/experiments/Experiments for thesis 2/well decoding/pfc/",1)
    # print_metric_details("C:/Users/NN/Desktop/Master/experiments/Experiments for thesis 2/well decoding/hc/",-1)
    # print_metric_details("C:/Users/NN/Desktop/Master/experiments/Experiments for thesis 2/well decoding/pfc/",-1)

    session = initiate_lickwell_network(nd)  # Initialize session
    X, y,nd,session = lickwells_io(session, nd, excluded_wells=[1], shift=nd.initial_timeshift, target_is_phase=True,
                                   lickstart=0,lickstop=5000)


    target_well_list = []
    firing_rate_list = []
    for i, lick in enumerate(session.licks):
        target_well_list.append(lick.target)
        firing_rates = np.zeros(len(X[i].spikes))
        for j,neuron_spikes in enumerate(X[i].spikes):
            firing_rates[j] = len(neuron_spikes)
        firing_rate_list.append(firing_rates)
    pass
