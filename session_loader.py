"""contains functions to load session data from source files"""
import numpy as np
import OpenEphys
import scipy
from settings import config
from database_api import Session


def load_datafile(file_path):
    """ loads data from .dat file with numpy"""
    try:
        data_file = open(file_path, "r")
        file_data = np.fromfile(data_file, dtype=np.uint16)
        return file_data
    except IOError:
        print("Error: File under " + file_path + " not found")
        return 0


def make_session():
    """ extract all relevant information for session and returns Session object. File paths are set under settings.py"""

    # load path definitions
    print("loading session")
    foster_path = config["paths"]["foster_path"]
    all_channels_path = config["paths"]["all_channels_path"]
    spiketracker_path = config["paths"]["spiketracker_path"]

    # extract spiketracker data
    spiketracker_data = scipy.io.loadmat(spiketracker_path)
    # param: head_s, pixelcm, speed_s, spike_templates, spike_times, waveform,x_s, y_s
    spike_times = spiketracker_data["spike_times"]
    x_s = spiketracker_data["x_s"]
    y_s = spiketracker_data["y_s"]
    speed_s = spiketracker_data["speed_s"]

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

    rewarded_lick = [x for ind, x in enumerate(lickwells) if rewarded[ind] == 1]

    data_licks = [initial_detection_timestamp, foster_timestamp, lickwells,
                  rewarded]  # TODO: code is correct until here
    # spikes = None, filter = None, filtered_spikes = None, metadata = None, enriched_metadata = None
    session = Session(spikes=spike_times)
    print("finished loading session")
    return session