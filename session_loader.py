"""contains functions to load session data from source files"""
import numpy as np
import tensorflow as tf
import OpenEphys
import scipy
from settings import config


def load_datafile(file_path):
    """ loads data from .dat file with numpy"""
    try:
        data_file = open(file_path, "r")
        file_data = np.fromfile(data_file, dtype=np.uint16)
        return file_data
    except IOError:
        print("Error: File under " + file_path + " not found")
        return 0


def find_min_time(mat): #test
    """ finds minimum value in ordered np list of lists. The default np min and flatten functions don't different
    list sizes"""
    minimum = mat[0][0]
    for i in range(0, mat.size):
        current_value = mat[i][0]
        if current_value < minimum:
            minimum = current_value
    return minimum


def find_max_time(mat):
    """ finds maximum value in ordered np list of lists. The default np min and flatten functions don't different
    list sizes"""
    maximum = mat[0][-1]
    for i in range(0, mat.size):
        current_value = mat[i][-1]
        if current_value > maximum:
            maximum = current_value
    return maximum


def make_zero_nxm_matrix(n, m):
    """ creates a list of size n x m and fills it with zeros"""
    mat = [0] * m
    for i in range(m):
        mat[i] = [0] * n
    return mat


def clean_spikes(spikes):
    """spike data from the source file is formatted as [[[1,2],[3,4]]] which is unfavorable
    and replaced by [[1,2],[3,4]] here"""
    return_array = np.empty(spikes.size, dtype=object)
    for i in range(0, spikes.size):
        return_array[i] = spikes[0][i].squeeze()
    return return_array


def make_dense_np_matrix(mat, minimum_value=None, maximum_value=None):
    """  accepts a list of lists of float and retuns a binary dense tensorflow matrix with 1 on y-axis where float would
     be in axis. Should be interpreted as neuron x time"""

    if minimum_value == None:
        minimum_value = find_min_time(mat)
    if maximum_value == None:
        maximum_value = find_max_time(mat)

    dense_matrix = make_zero_nxm_matrix(1 + int(maximum_value) - int(minimum_value), len(mat))
    for i in range(0, len(mat)):  # for each neuron
        for j in range(0, len(mat[i]) - 1):  # for all saved times
            k = int(mat[i][j])  # saved time as index
            dense_matrix[i][k] = 1
    dense_matrix = np.array(dense_matrix)
    return dense_matrix


def make_session():
    """ extract all relevant information for session and returns Session dict. File paths are set under settings.py"""

    print("loading session...")
    foster_path = config["paths"]["foster_path"]
    all_channels_path = config["paths"]["all_channels_path"]
    spiketracker_path = config["paths"]["spiketracker_path"]

    # extract spiketracker data
    spiketracker_data = scipy.io.loadmat(spiketracker_path)
    # param: head_s, pixelcm, speed_s, spike_templates, spike_times, waveform,x_s, y_s
    spikes = spiketracker_data["spike_times"]
    position_x = spiketracker_data["x_s"].squeeze()
    position_y = spiketracker_data["y_s"].squeeze()
    speed = spiketracker_data["speed_s"].squeeze()

    # load foster data
    foster_data = load_datafile(foster_path)  # contains tripels with data about TODO TODO and location
    foster_data = np.reshape(foster_data, [foster_data.size // 3,
                                           3])  # 0: rewarded, 1: lickwell, 2: duration TODO what do values stand for?
    initial_detection = [x == 2 for x in
                         foster_data[:, 0]]  # TODO "Exclude detection events", i am not sure what this is used for

    # load data from all_channels.event file with ephys
    all_channels_dict = OpenEphys.load(
        all_channels_path)  # param: eventType, sampleNum, header, timestamps, recordingNumber, eventId, nodeId, channel
    timestamps = all_channels_dict["timestamps"]
    eventch_ttl = all_channels_dict["channel"]  # channel number
    eventtype_ttl = all_channels_dict["eventType"]  # 3 for TTL events TODO other numbers?
    eventid_ttl = all_channels_dict["eventId"]  # 1: for on, 0 for off
    recording_num = all_channels_dict["recordingNumber"]
    sample_rate = float(all_channels_dict["header"]["sampleRate"])

    # timestamp for foster data
    foster_timestamp = [(x
                         - timestamps[0]) / sample_rate * 1000 for ind, x in enumerate(timestamps) if (
                                (eventtype_ttl[ind] == 3) and (eventch_ttl[ind] == 2) and (eventid_ttl[ind] == 1) and (
                                recording_num[ind] == 0))]

    if len(initial_detection) > len(foster_timestamp):
        foster_data = foster_data[:, 0:foster_timestamp.size]
        initial_detection = initial_detection[1:foster_timestamp.size]
    foster_data = [x for ind, x in enumerate(foster_data) if initial_detection[ind] != 1]
    detected_events = [ind for ind, x in enumerate(initial_detection) if
                       x == False]  # index positions of non detection events
    initial_detection_timestamp = [foster_timestamp[ind - 1] for ind in
                                   detected_events[1:]]  # TODO why is this index shifted?
    initial_detection_timestamp = [
                                      0] + initial_detection_timestamp  # %the first lick is at homewell for which the initial lick timing is not measured
    foster_timestamp = [x for ind, x in enumerate(foster_timestamp) if initial_detection[ind] != 1]
    rewarded = [item[0] for item in foster_data]
    durations = [item[1] for item in foster_data]
    lickwells = [item[2] for item in foster_data]

    trial_timestamp = np.array([[x, initial_detection_timestamp[ind]] for ind, x in enumerate(lickwells) if
                                rewarded[ind] == 1])  # TODO was originally rewarded_licks, is this ok?

    licks = np.array([initial_detection_timestamp, foster_timestamp, lickwells,
                      rewarded])
    spikes = clean_spikes(spikes)
    # spikes = None, filter = None, filtered_spikes = None, metadata = None, enriched_metadata = None
    spikes_dense = make_dense_np_matrix(spikes)
    #spikes = spikes.astype(TimePoint)
    # session = Session(spikes=spikes, licks=licks, spikes_dense=spikes_dense, position_x=position_x,
    #                   position_y=position_y, speed=speed, trial_timestamp=trial_timestamp)
    return_dict = dict(
        spikes=spikes, licks=licks, spikes_dense=spikes_dense, position_x=position_x,
                           position_y=position_y, speed=speed, trial_timestamp=trial_timestamp)

    print("finished loading session")
    return return_dict
