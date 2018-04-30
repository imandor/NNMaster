import numpy as np


start = TimePoint(ms=0)
stop = TimePoint(ms=1000)
# or equivalently
start = TimePoint(s=0)
stop = TimePoint(s=1)

session = Session(path_to_session_1)
session += Session(path_to_session_2)  # concatenates session1 and session2
time_slice = session.time_slice(start, stop)
trials = session.trials  # list of Slices corresponding to the trials

filtered_time_slice = time_slice.convolve(window)
frame_size = TimePoint(ms=0)#test2
frame_stride = TimePoint(ms=150)
filtered_frames = filtered_time_slice.to_frames(frame_size, frame_stride)  # a list of FilteredSlice
frame1 = filtered_frames[0]
filtered_spikes = frame1.data  # a numpy array
metadata = frame1.data  # a dictionary


class TimePoint:
    def __init__(self, ms=None, s=None, min=None):
        self._ms = 0
        if ms is not None:
            self._ms += ms
        if s is not None:
            self._ms += 1e3 * s
        if min is not None:
            self._ms += 6e4 * min

    @property
    def ms(self):
        return self._ms

    @property
    def s(self):
        return self._ms / 1e3

    @property
    def min(self):
        return self._ms / 6e4


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
        pass


start = TimePoint(ms=0)
stop = TimePoint(ms=1000)
# or equivalently
start = TimePoint(s=0)
stop = TimePoint(s=1)

time_slice = session.time_slice(start, stop)
trials = session.trials  # list of Slices corresponding to the trials

filtered_time_slice = time_slice.convolve(window)
frame_size = TimePoint(ms=0)
frame_stride = TimePoint(ms=150)
filtered_frames = filtered_time_slice.to_frames(frame_size, frame_stride)  # a list of FilteredSlice
frame1 = filtered_frames[0]
filtered_spikes = frame1.data  # a numpy array
metadata = frame1.data  # a dictionary




my_data_slice = Slice("../data/path_to_folder/")
start = TimePoint(ms=0)
end = TimePoint(ms=150)
my_new_data_slice = my_data_slice.time_slice(start, end)
my_data_slice.is_convolved  ---> returns False
my_data_slice.set_filter(np_filter)
my_data_slice.is_convolved  ---> returns True

# returns a list of numy arrays with the convolved data
my_frames = my_data_slice.to_frames(TimePoint(ms=100), TimePoint(ms=10))


my_trial = my_data_slice.get_trials()


class Trial:
    def __init__(self, path):
        self.spikes = ???
        self.filter = ???
        self.filtered_spikes = ???
        self.metadata = ???
        self.enriched_metadata = ???
        pass

    @property
    def is_convolved(self):
        pass

    def set_filter(self, filter):
        pass

    def write(self, path):
        pass

    def time_slice(self, start, stop):
        pass

    def to_frames(self, frame_size, frame_stride):
        pass

    def plot(self, ax1, ax2, args1, args2):
        # plot spikes in ax1
        self.plot_spikes(ax1, args1, args2)
        # plot metadata in ax2
        self.plot_metadata(ax1, args1, args2)
        pass

    def plot_spikes(self, ax, args1, args2, filtered=True):
        # if filtered = False, plot raw spikes else plot filtered spikes
        pass

    def plot_metadata(self, ax, args1, args2):
        pass


class Slice(Trial):
    def get_trials(self, start_position=None, end_position=None, max_length=None, min_length=None):
        # returns a list Trial object corresponding to the trials contained in that slice
        pass
