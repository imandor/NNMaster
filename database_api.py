import numpy as np
from session_loader import make_session
from settings import save_as_pickle, load_pickle


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


class Session:
    def __init__(self, session_file=None, save_session_as=None, metadata=None,
                 enriched_metadata=None):
        """ Contains all data in one session. A session object can be stored or loaded from a pickle file.
        If no session_file is specified, Session is generated by reading the source files specified in settings.py"""
        if session_file is None:
            session_dict = make_session()
        else:
            session_dict = load_pickle(session_file)
        if save_session_as is not None:
            save_as_pickle(session_file)
        self.spikes = session_dict["spikes"]
        self.spikes_dense = session_dict["spikes_dense"]
        self.licks = session_dict["licks"]
        self.position_x = session_dict["position_x"]
        self.position_y = session_dict["position_y"]
        self.speed = session_dict["speed"]
        self.trial_timestamp = session_dict["trial_timestamp"]
        self.filter = session_dict["filter"]
        self.filtered_spikes = session_dict["filtered_spikes"]
        self.metadata = metadata
        self.enriched_metadata = enriched_metadata
        pass


class Slice(Session):
    def __init__(self, spikes=None, spikes_dense=None, licks=None, position_x=None, position_y=None, speed=None,
                 trial_timestamp=None, filtered_spikes=None):
        """ Represents a time range of collected data. """
        self.spikes = spikes
        self.spikes_dense = spikes_dense
        self.licks = licks
        self.position_x = position_x
        self.position_y = position_y
        self.speed = speed
        self.trial_timestamp = trial_timestamp
        self.filter = filter
        self.filtered_spikes = filtered_spikes
        pass

    @property
    def trials(self):
        pass

    def convolve(self, window):
        # returns a FilteredSlice
        pass


class FilteredSlice(Slice):
    # has no __init__, must be constructed from a session
    def time_slice(self, start, stop):
        spikes = np.array([splice_nparray_at_start_stop(x, start, stop) for x in self.spikes])
        # TODO all entries filter are not ready
        spikes_dense = np.array([x for ind, x in enumerate(self.spikes_dense) if stop.ms > ind >= start.ms])
        licks = np.array([splice_nparray_at_start_stop(x, start, stop) for x in self.licks])
        position_x = np.array([x for ind, x in enumerate(self.position_x) if stop.ms > ind >= start.ms])
        position_y = np.array([x for ind, x in enumerate(self.position_y) if stop.ms > ind >= start.ms])
        speed = np.array([x for ind, x in enumerate(self.speed) if stop.ms > ind >= start.ms])
        trial_timestamp = np.array([splice_nparray_at_start_stop(x, start, stop) for x in self.trial_timestamp])
        filter = filter
        filtered_spikes = self.filtered_spikes
        metadata = self.metadata
        enriched_metadata = self.enriched_metadata
        return FilteredSlice(spikes=spikes, spikes_dense=spikes_dense, licks=licks, position_x=position_x,
                             position_y=position_y, speed=speed,
                             trial_timestamp=trial_timestamp, filter=filter, filtered_spikes=filtered_spikes,
                             metadata=metadata,
                             enriched_metadata=enriched_metadata)
        pass

    def to_frames(self, frame_size, frame_stride):
        pass


class Trial:
    # has no __init__, must be constructed from a session

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

    def get_trial_by_id(self, trial_id):
        start = TimePoint(ms=self.trial_timestamp[trial_id][1])
        stop = TimePoint(ms=self.trial_timestamp[trial_id + 1][1])
        return Trial(Slice.time_slice())

    def get_trial_by_time(self, trial_time):
        start = TimePoint(ms=np.argmax(self.trial_timestamp[..., 1] > trial_time))
        trial_id = np.where(self.trial_timestamp[..., 1] == start.ms)
        stop = TimePoint(ms=self.trial_timestamp[trial_id + 1][1])
        return Trial(Slice.time_slice())

    def get_trials(self, start_position=None, end_position=None, max_length=None, min_length=None):
        # returns a list of Trial objects corresponding to the trials contained in that slice
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
                            return_array.append(self.get_trial_by_time(self, current_time))
        pass
