from src.database_api_beta import Slice, Filter, hann, Net_data
class Experiment():

    def __init__(self, study, nd):
        self.nd = nd
        self.study = study





lickwell_experiment_pfc_future = Experiment(
    study="lickwell",
nd = Net_data(
    network_shape=11,
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
    model_path="G:/master_datafiles/trained_networks/MLP_PFC_lickwell_example/",
    raw_data_path="G:/master_datafiles/raw_data/PFC/",
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
    network_shape=11,
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
    model_path="G:/master_datafiles/trained_networks/MLP_PFC_lickwell_example/",
    raw_data_path="G:/master_datafiles/raw_data/PFC/",
    filtered_data_path="session_pfc_lw.pkl",
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
    network_shape=11,
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
    model_path="G:/master_datafiles/trained_networks/MLP_HC_lickwell_example/",
    raw_data_path="G:/master_datafiles/raw_data/HC/",
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
    network_shape=11,
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
    model_path="G:/master_datafiles/trained_networks/MLP_HC_lickwell_example/",
    raw_data_path="G:/master_datafiles/raw_data/HC/",
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


position_decoding_hc = Experiment(
study="position",

nd = Net_data(
    initial_timeshift=-30000,
    epochs=10,
    time_shift_iter=500,
    time_shift_steps=121,
    early_stopping=False,
    model_path="G:/master_datafiles/trained_networks/DMF_HC_example/",
    raw_data_path="G:/master_datafiles/raw_data/HC/",
    filtered_data_path="session_hc",
    k_cross_validation=10

)
)

position_decoding_pfc = Experiment(
study="position",

nd = Net_data(
    initial_timeshift=-30000,
    epochs=10,
    time_shift_iter=500,
    time_shift_steps=121,
    early_stopping=False,
    model_path="G:/master_datafiles/trained_networks/DMF_PFC_example/",
    raw_data_path="G:/master_datafiles/raw_data/PFC/",
    filtered_data_path="session_pf",
    k_cross_validation=10
)
)

naive_test_hc = Experiment(
    study= "position",
    nd = Net_data(
        initial_timeshift=0,
        time_shift_iter=500,
        time_shift_steps=11,
        early_stopping=False,
        model_path="G:/master_datafiles/trained_networks/Naive_HC_example/",
        raw_data_path="G:/master_datafiles/raw_data/HC/",
        filtered_data_path="session_hc",
        k_cross_validation = 1,
        naive_test=True,
        from_raw_data=False,
        epochs = 10
    )
)

naive_test_pfc = Experiment(
    study= "position",
    nd = Net_data(
        initial_timeshift=0,
        time_shift_iter=500,
        time_shift_steps=11,
        early_stopping=False,
        model_path="G:/master_datafiles/trained_networks/Naive_PFC_example/",
        raw_data_path="G:/master_datafiles/raw_data/PFC/",
        filtered_data_path="session_pfc",
        k_cross_validation = 1,
        naive_test=True,
        from_raw_data=False,
        epochs = 10
    )
)