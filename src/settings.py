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
lickwell_list = [1,2,3,4,5]

config = dict(
    setup = dict(
        lickwell_list=lickwell_list

    ),
    paths = dict(
    path_to_session = path_to_session,
    foster_path = foster_path,
    all_channels_path = all_channels_path,
    spiketracker_path = spiketracker_path,
    figure_path = figure_path,
    data_slice = unit_test_path +"data_slice.pkl",
    smaller_data_slice= unit_test_path +"smaller_data_slice.pkl",
    phases= unit_test_path +"phases.pkl",
    trial= unit_test_path +"trial.pkl",
    list_of_trials= unit_test_path +"list_of_trials.pkl",
    sub_list_of_trials= unit_test_path +"sub_list_of_trials.pkl",
    nth_trial_list= unit_test_path +"nth_trial_list.pkl",
    all_trials_in_second_phase= unit_test_path +"all_trials_in_second_phase.pkl"

),
    image_labels = dict(
        trial_spikes_title = "Spikes in trials by time",
        trial_spikes_y1_left = "trial id",
        trial_spikes_y1_right = "wells",
        trial_spikes_x1 = "time (ms)",
        position_spikes_title="Spikes in trials by position",
        position_y1_left="trial id",
        position_y1_right="wells",
        position_x1="position (cm)",
        filtered_spikes_title = "Number of spikes in trial {} by time",
        filtered_spikes_y1_left = "spikes over all neurons",
        filtered_spikes_y1_right = "",
        filtered_spikes_x1 = "time ({} ms bins)"
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
        pickle.dump(data, f,protocol=pickle.HIGHEST_PROTOCOL)
        return "saved!"


def load_pickle(file_name):
    """loads data from pickle file"""
    with open(file_name,'rb') as f:
        data = pickle.load(f)
        return data

# save_as_pickle('config.pkl', config)
# config = load_pickle('config.pkl')