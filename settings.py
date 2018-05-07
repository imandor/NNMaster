"""Saves and loads parameters used in all networks """
import pickle

""" paths and filenames"""
path_to_session = "C:/Users/NN/Desktop/Master/sample_data/2018-04-09_14-39-52/"
foster_path = path_to_session + "2018-04-09_14-39-53_fostertask.dat"
all_channels_path = path_to_session + "all_channels.events"
spiketracker_path = path_to_session + "probe1/session1/spike_tracker_data.mat"
config = dict(
    paths = dict(
    path_to_session = path_to_session,
    foster_path = foster_path,
    all_channels_path = all_channels_path,
    spiketracker_path = spiketracker_path,
    )


)


def save_as_pickle(file_name, data):
    """ saves data in a pickle file"""
    with open(file_name, 'wb') as f:
        pickle.dump(data, f)
        return "saved!"


def load_pickle(file_name):
    """loads data from pickle file"""
    with open(file_name,'rb') as f:
        data = pickle.load(f)
        return data


save_as_pickle('config.pkl', config)
config = load_pickle('config.pkl')