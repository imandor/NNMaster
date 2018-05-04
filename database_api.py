import numpy as np




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
    def __init__(self, spikes=None, spikes_dense=None, licks=None, licks_dense=None, position_x=None, position_y=None,
                 speed=None, filter=None, filtered_spikes=None, metadata=None, enriched_metadata=None):
        self.spikes = spikes
        self.spikes_dense = spikes_dense
        self.licks = licks
        self.licks_dense = licks_dense
        self.position_x = position_x
        self.position_y = position_y
        self.speed = speed
        self.filter = filter
        self.filtered_spikes = filtered_spikes
        self.metadata = metadata
        self.enriched_metadata = enriched_metadata
        pass

class Trial:
    def __init__(self, spikes = None, filter = None, filtered_spikes = None, metadata = None, enriched_metadata = None):
        self.spikes = spikes
        self.filter = filter
        self.filtered_spikes = filtered_spikes
        self.metadata = metadata
        self.enriched_metadata = enriched_metadata
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




