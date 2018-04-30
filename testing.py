
import numpy as np
import OpenEphys
class FilteredSlice:
    # has no __init__, must be constructed from a session
    def time_slice(self, start, stop):
        pass

    def to_frames(self, frame_size, frame_stride):
        pass

class Slice(FilteredSlice):
    # has no __init__, must be constructed from a session
    @property
    def trials(self):
        pass

    def convolve(self, window):
        # returns a FilteredSlice
        pass




class Session(Slice):
    def __init__(self, path):
        self.spikes = None
        self.filter = None
        self.filtered_spikes = None
        self.metadata = None
        self.enriched_metadata = None
        pass


def loadDataFile(file_path):
    # loads data from .dat file with numpy
    try:
        data_file = open(file_path, "r")
        file_data = np.fromfile(data_file, dtype=np.uint16)
        return file_data
    except IOError:
        print("Error: File under " + file_path + " not found")
        return 0




path_to_session = "C:/Users/NN/Desktop/Master/sample_data/2018-04-09_14-39-52/"
foster_path = path_to_session + "2018-04-09_14-39-53_fostertask.dat"
all_channels_path = path_to_session + "all_channels.events"

#load foster data
foster_data = loadDataFile(foster_path)#contains tripels with data about TODO TODO and location
foster_data = np.reshape(foster_data, [foster_data.size // 3, 3]) #TODO what do the 0 1 2 in the first part of the tripels stand for?
initial_detection = [x == 2 for x in foster_data[:, 0]] #TODO "Exclude detection events", i am not sure what this is used for

#load data from all_channels.event file with ephys
all_channels_dict = OpenEphys.load(all_channels_path)#dict contains eventType, sampleNum, header, timestamps, recordingNumber, eventId, nodeId, channel
timestamps = all_channels_dict["timestamps"]
eventCh_TTL = all_channels_dict["channel"] #channel number
eventType_TTL = all_channels_dict["eventType"]#3 for TTL events TODO other numbers?
eventID_TTL = all_channels_dict["eventId"]#1: for on, 0 for off
recording_num = all_channels_dict["recordingNumber"]
samplerate = all_channels_dict["header"]["sampleRate"]
print(all_channels_dict)
print(timestamps)
print(eventType_TTL)