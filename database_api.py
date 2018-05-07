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


def splice_nparray_at_start_stop(li, start, stop):
    """splices a numpy array at the entry higher than start and the entry lower than stop value"""
    return np.array([x for ind, x in enumerate(li) if stop.ms > li[ind] >= start.ms])


class FilteredSlice:
    # has no __init__, must be constructed from a session
    def time_slice(self, start, stop):
        self.spikes = np.array([splice_nparray_at_start_stop(x, start, stop) for x in self.spikes])
        # TODO all entries filter are not ready
        self.spikes_dense = np.array([x for ind, x in enumerate(self.spikes_dense) if stop.ms > ind >= start.ms])
        self.licks = np.array([splice_nparray_at_start_stop(x, start, stop) for x in self.licks])
        self.position_x = np.array([x for ind, x in enumerate(self.position_x) if stop.ms > ind >= start.ms])
        self.position_y = np.array([x for ind, x in enumerate(self.position_y) if stop.ms > ind >= start.ms])
        self.speed = np.array([x for ind, x in enumerate(self.speed) if stop.ms > ind >= start.ms])
        self.trial_timestamp = np.array([splice_nparray_at_start_stop(x, start, stop) for x in self.trial_timestamp])
        self.filter = filter
        self.filtered_spikes = self.filtered_spikes
        self.metadata = self.metadata
        self.enriched_metadata = self.enriched_metadata
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
    def __init__(self, spikes=None, spikes_dense=None, licks=None, position_x=None, position_y=None,
                 speed=None, trial_timestamp=None, filter=None, filtered_spikes=None, metadata=None,
                 enriched_metadata=None):
        self.spikes = spikes
        self.spikes_dense = spikes_dense
        self.licks = licks
        self.position_x = position_x
        self.position_y = position_y
        self.speed = speed
        self.trial_timestamp = trial_timestamp
        self.filter = filter
        self.filtered_spikes = filtered_spikes
        self.metadata = metadata
        self.enriched_metadata = enriched_metadata
        pass


class Trial:
    """ a slice of a session between two trials"""

    def __init__(self, spikes=None, spikes_dense=None, licks=None, position_x=None, position_y=None,
                 speed=None, trial_timestamp=None, filter=None, filtered_spikes=None, metadata=None,
                 enriched_metadata=None):
        self.spikes = spikes
        self.spikes_dense = spikes_dense
        self.licks = licks
        self.position_x = position_x
        self.position_y = position_y
        self.speed = speed
        self.trial_timestamp = trial_timestamp
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
    #     def get_trials(self, start_position=None, end_position=None, max_length=None, min_length=None):
    #         # returns a list Trial object corresponding to the trials contained in that slice
    #         pass
    def get_trials(self, start_position=None, end_position=None, max_length=None, min_length=None):
        # returns a list Trial object corresponding to the trials contained in that slice
        for ind in range(1, self.trial_timestamp.length):  # the first intended lick is always at well 1
            return_array = np.empty(1, dtype=object)
            last_well = self.trial_timestamp[ind - 1][0]
            last_time = self.trial_timestamp[ind - 1][1]
            current_well = self.trial_timestamp[ind][0]
            current_time = self.trial_timestamp[ind][1]
            trial_duration = current_time - last_time
            if start_position == last_well or start_position is None:
                if end_position == current_well or end_position is None:
                    if max_length < trial_duration or max_length is None:
                        if min_length > trial_duration or min_length is None:
                            return_array.append(Trial())
        pass

    # def get_trials_by_time(self, start_position=None, end_position=None, max_length=None, min_length=None):
    #     # returns a list Trial object corresponding to the trials contained in that slice
    #     pass
    # def get_trials_by_id(self, start_position=None, end_position=None, max_length=None, min_length=None):
    #     # returns a list Trial object corresponding to the trials contained in that slice
    #     pass
