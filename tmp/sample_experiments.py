from src.database_api_beta import Slice, Filter, hann, Net_data
from src.conf import mlp, mlp_discrete
class Experiment():

    def __init__(self, study, nd):
        self.nd = nd
        self.study = study





lickwell_experiment_pfc_future = Experiment(
    study="lickwell",
nd = Net_data(
    network_shape=mlp_discrete,
    # Program execution settings
    epochs=10,
    evaluate_training=False,
    slice_size=200,
    stride=200,
    y_step=200,
    win_size=200,
    search_radius=200,
    k_cross_validation=10,
    session_filter=Filter(func=hann, search_radius=200, step_size=200),
    valid_ratio=0.1,
    testing_ratio=0,
    time_shift_steps=1,
    early_stopping=False,
    model_path="G:/master_datafiles/trained_networks/MLP_HC_2019-01-08_lickwell_testversion/",
    raw_data_path="G:/master_datafiles/raw_data/2018-05-16_17-13-37/",
    filtered_data_path="session_hc_lw.pkl",
    metric="discrete",
    shuffle_data=True,
    shuffle_factor=1,
    lw_classifications=5,
    lw_normalize=True,
    lw_differentiate_false_licks=False,
    num_wells=5,
    initial_timeshift=1,
))
lickwell_experiment_pfc_memory = Experiment(
study="lickwell",
nd = Net_data(
    network_shape=mlp_discrete,
    # Program execution settings
    epochs=10,
    evaluate_training=False,
    slice_size=200,
    stride=200,
    y_step=200,
    win_size=200,
    search_radius=200,
    k_cross_validation=10,
    session_filter=Filter(func=hann, search_radius=200, step_size=200),
    valid_ratio=0.1,
    testing_ratio=0,
    time_shift_steps=1,
    early_stopping=False,
    model_path="G:/master_datafiles/trained_networks/MLP_HC_2019-01-08_lickwell_testversion/",
    raw_data_path="G:/master_datafiles/raw_data/2018-05-16_17-13-37/",
    filtered_data_path="session_hc_lw.pkl",
    metric="discrete",
    shuffle_data=True,
    shuffle_factor=1,
    lw_classifications=5,
    lw_normalize=True,
    lw_differentiate_false_licks=False,
    num_wells=5,
    initial_timeshift=-1,
))
lickwell_experiment_hc_future = Experiment(
study="lickwell",
    nd =  Net_data(
    network_shape=mlp_discrete,
        # Program execution settings
    epochs=10,
    evaluate_training=False,
    slice_size=200,
    stride=200,
    y_step=200,
    win_size=200,
    search_radius=200,
    k_cross_validation=10,
    session_filter=Filter(func=hann, search_radius=200, step_size=200),
    valid_ratio=0.1,
    testing_ratio=0,
    time_shift_steps=1,
    early_stopping=False,
    model_path="G:/master_datafiles/trained_networks/MLP_HC_2018-12-29_lickwell/",
    raw_data_path="G:/master_datafiles/raw_data/2018-05-16_17-13-37/",
    filtered_data_path= "session_hc_lw.pkl",
    metric="discrete",
    shuffle_data=True,
    shuffle_factor=1,
    lw_classifications=5,
    lw_normalize=True,
    lw_differentiate_false_licks=False,
    num_wells=5,
    initial_timeshift=1,
))

lickwell_experiment_hc_memory = Experiment(
study="lickwell",
    nd =  Net_data(
    network_shape=mlp_discrete,
    epochs=10,
    evaluate_training=False,
    slice_size=200,
    stride=200,
    y_step=200,
    win_size=200,
    search_radius=200,
    k_cross_validation=10,
    session_filter=Filter(func=hann, search_radius=200, step_size=200),
    valid_ratio=0.1,
    testing_ratio=0,
    time_shift_steps=1,
    early_stopping=False,
    model_path="G:/master_datafiles/trained_networks/MLP_HC_2018-12-29_lickwell/",
    raw_data_path="G:/master_datafiles/raw_data/2018-05-16_17-13-37/",
    filtered_data_path= "session_hc_lw.pkl",
    metric="discrete",
    shuffle_data=True,
    shuffle_factor=1,
    lw_classifications=5,
    lw_normalize=True,
    lw_differentiate_false_licks=False,
    num_wells=5,
    initial_timeshift=-1,
))
