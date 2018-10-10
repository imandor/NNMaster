"""Saves and loads parameters used in all networks """
import pickle
# import dill as pickle
import json
import os
import errno

""" paths and filenames"""
# path_to_session = "C:/Users/NN/Desktop/Master/sample_data/2018-04-09_14-39-52/"
# foster_path = path_to_session + "2018-04-09_14-39-53_fostertask.dat"
# all_channels_path = path_to_session + "all_channels.events"
# spiketracker_path = path_to_session + "probe1/session1/spike_tracker_data.mat"
# figure_path = "data/figures/"
# unit_test_path = "data/pickle/unit_testing/"

path_to_session = "C:/Users/NN/PycharmProjects/NNMaster/data/2018-05-16_17-13-37/"
foster_path = path_to_session + "2018-05-16_17-13-37_fostertask.dat"
all_channels_path = path_to_session + "all_channels.events"
spiketracker_path = path_to_session + "probe1/session1/spike_tracker_data.mat"
figure_path = "data/figures/"
unit_test_path = "data/pickle/unit_testing/"
lickwell_list = [1, 2, 3, 4, 5]

config = dict(
    setup=dict(
        lickwell_list=lickwell_list

    ),
    paths=dict(
        path_to_session=path_to_session,
        foster_path=foster_path,
        all_channels_path=all_channels_path,
        spiketracker_path=spiketracker_path,
        figure_path=figure_path,
        data_slice=unit_test_path + "data_slice.pkl",
        smaller_data_slice=unit_test_path + "smaller_data_slice.pkl",
        phases=unit_test_path + "phases.pkl",
        trial=unit_test_path + "trial.pkl",
        list_of_trials=unit_test_path + "list_of_trials.pkl",
        sub_list_of_trials=unit_test_path + "sub_list_of_trials.pkl",
        nth_trial_list=unit_test_path + "nth_trial_list.pkl",
        all_trials_in_second_phase=unit_test_path + "all_trials_in_second_phase.pkl"

    ),
    image_labels=dict(
        trial_spikes_title="Spikes in trials by time",
        trial_spikes_y1_left="trial id",
        trial_spikes_y1_right="wells",
        trial_spikes_x1="time (ms)",
        position_spikes_title="Spikes in trials by position",
        position_y1_left="trial id",
        position_y1_right="wells",
        position_x1="position (cm)",
        filtered_spikes_title="Number of spikes in trial {} by time",
        filtered_spikes_y1_left="spikes over all neurons",
        filtered_spikes_y1_right="",
        filtered_spikes_x1="time ({} ms bins)"
    )

)


def save_as_pickle(file_name, data):
    """ saves data in a pickle file"""
    if not os.path.exists(os.path.dirname(file_name)):
        try:
            os.makedirs(os.path.dirname(file_name))
        except OSError as exc:  # Guard against race condition
            if exc.errno != errno.EEXIST:
                raise
    with open(file_name, 'wb') as f:
        pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
        return "saved!"


def save_net_dict(path, net_dict):
    file = open(path + "/net_dict.txt", "w")
    file.write("RAW_DATA_PATH: " + net_dict["RAW_DATA_PATH"] + "\n")
    file.write("STRIDE: " + str(net_dict["STRIDE"]) + "\n")
    file.write("Y_SLICE_SIZE: " + str(net_dict["Y_SLICE_SIZE"]) + "\n")
    file.write("network_type: " + str(net_dict["network_type"]) + "\n")
    file.write("EPOCHS: " + str(net_dict["EPOCHS"]) + "\n")
    file.write("session_filter: " + str(net_dict["session_filter"]) + "\n")
    file.write("TIME_SHIFT_STEPS: " + str(net_dict["TIME_SHIFT_STEPS"]) + "\n")
    file.write("SHUFFLE_DATA: " + str(net_dict["SHUFFLE_DATA"]) + "\n")
    file.write("SHUFFLE_FACTOR: " + str(net_dict["SHUFFLE_FACTOR"]) + "\n")
    file.write("TIME_SHIFT_ITER: " + str(net_dict["TIME_SHIFT_ITER"]) + "\n")
    file.write("MODEL_PATH: " + str(net_dict["MODEL_PATH"]) + "\n")
    file.write("learning_rate: " + str(net_dict["learning_rate"]) + "\n")
    file.write("LOAD_MODEL: " + str(net_dict["LOAD_MODEL"]) + "\n")
    file.write("INITIAL_TIMESHIFT: " + str(net_dict["INITIAL_TIMESHIFT"]) + "\n")
    file.write("METRIC_ITER: " + str(net_dict["METRIC_ITER"]) + "\n")
    file.write("BATCH_SIZE: " + str(net_dict["BATCH_SIZE"]) + "\n")
    file.write("SLICE_SIZE: " + str(net_dict["SLICE_SIZE"]) + "\n")
    file.write("SEARCH_RADIUS: " + str(net_dict["SEARCH_RADIUS"]) + "\n")
    file.write("EARLY_STOPPING: " + str(net_dict["EARLY_STOPPING"]) + "\n")
    file.close()
    pass


def load_pickle(file_name):
    """loads data from pickle file"""
    with open(file_name, 'rb') as f:
        data = pickle.load(f)
        return data

# save_as_pickle('config.pkl', config)
# config = load_pickle('config.pkl')
