import numpy as np
from session_loader import make_session
from settings import save_as_pickle, load_pickle
import matplotlib.pyplot as plt


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
    return np.asarray([x for ind, x in enumerate(li) if stop.ms > li[ind] >= start.ms])


def splice_nparray_at_start_stop2(li, start, stop):
    """splices a numpy array at the entry higher than start and the entry lower than stop value"""
    return np.asarray([x for ind, x in enumerate(li) if stop.ms > li[ind][1] >= start.ms])





class Slice():
    """ contains data and metadata pertaining one time slice of the session"""

    def __init__(self, spikes=None, licks=None, position_x=None, position_y=None, speed=None,
                 trial_timestamp=None, filtered_spikes=None, slice_metadata=None, enriched_metadata=None):
        """ Represents a time range of collected data. """
        self.spikes = spikes
        # self.spikes_dense = spikes_dense
        self.licks = licks
        self.position_x = position_x
        self.position_y = position_y
        self.speed = speed
        self.trial_timestamp = trial_timestamp
        self.slice_metadata = slice_metadata
        self.enriched_metadata = enriched_metadata
        pass

    @property
    def filter(self):
        pass

    def convolve(self, window):
        # returns a FilteredSlice
        pass

    def get_trial_by_id(self, trial_id):
        start = TimePoint(ms=self.trial_timestamp[trial_id][1])
        stop = TimePoint(ms=self.trial_timestamp[trial_id + 1][1])
        start_well = self.trial_timestamp[trial_id - 1][0]
        stop_well = self.trial_timestamp[trial_id - 1][0]
        return self.make_slice(start, stop).make_trial(trial_id=trial_id, start_time=start, stop_time=stop,
                                                       start_well=start_well, stop_well=stop_well, trial_metadata=None)

    def get_trial_by_time(self, trial_time):
        start = TimePoint(ms=np.argmax(self.trial_timestamp[..., 1] > trial_time.ms))
        trial_id = np.where(self.trial_timestamp[..., 1] >= start.ms)
        trial_id = trial_id[0][0]
        return self.get_trial_by_id(trial_id)

    def get_all_trials_by_time(self, start=None, stop=None, max_length=None, min_length=None):
        """ returns a list of Trial objects corresponding to the trials contained in that slice """
        for ind in range(1, len(self.trial_timestamp)):  # the first intended lick is always at well 1
            return_array = np.empty(0, dtype=object)
            last_well = self.trial_timestamp[ind - 1][0]
            last_time = TimePoint(ms=self.trial_timestamp[ind - 1][1])
            current_well = self.trial_timestamp[ind][0]
            current_time = TimePoint(ms=self.trial_timestamp[ind][1])
            trial_duration = current_time.ms - last_time.ms
            trial_id = self.trial_timestamp[1]
            if start.ms <= last_time.ms or start is None:
                if stop.ms >= current_time.ms or stop is None:
                    if max_length is None or max_length.ms < trial_duration:
                        if min_length is None or min_length.ms > trial_duration:
                            return_array = np.concatenate((return_array, [
                                self.make_trial(trial_id=trial_id, start_time=last_time, stop_time=current_time,
                                                start_well=last_well, stop_well=current_well, trial_metadata=None)]),
                                                          axis=0)
        return return_array

    def make_trial(self, trial_id, start_time, stop_time, start_well, stop_well, trial_metadata=None):
        return Trial(trial_id=trial_id, start_time=start_time, stop_time=stop_time, start_well=start_well,
                     stop_well=stop_well,
                     spikes=self.spikes, licks=self.licks, position_x=self.position_x, position_y=self.position_y,
                     speed=self.speed,
                     filtered_spikes=self.filtered_spikes, trial_metadata=trial_metadata,
                     enriched_metadata=self.enriched_metadata, filter=self.filter)




class FilteredSlice(Slice):
    """ has no __init__, must be constructed from a slice"""

    @property
    def filter(self):
        return self.filter

    @property
    def filtered_spikes(self):
        return self.filter

    def bin_spikes(self, bin_size):
        pass

    def convolve(self, window):
        pass

    def to_frames(self, frame_size, frame_stride):
        pass


class Trial:
    def __init__(self, trial_id, start_time, stop_time, start_well, stop_well, spikes=None, licks=None, position_x=None,
                 position_y=None, speed=None,
                 filtered_spikes=None, trial_metadata=None, enriched_metadata=None, filter=None):
        self.trial_id = trial_id
        self.spikes = spikes
        self.start_time = start_time
        self.stop_time = stop_time
        self.start_well = start_well
        self.stop_well = stop_well
        # self.spikes_dense = spikes_dense
        self.licks = licks
        self.position_x = position_x
        self.position_y = position_y
        self.speed = speed
        self.filter = filter
        self.filtered_spikes = filtered_spikes
        self.trial_metadata = trial_metadata
        self.enriched_metadata = enriched_metadata

    @property
    def is_convolved(self):
        pass

    def set_filter(self, filter):
        pass

    def write(self, path):
        pass

    def to_frames(self, frame_size, frame_stride):
        pass

    def plot(self, ax1, ax2, args1, args2):
        # plot spikes in ax1
        self.plot_spikes(ax1, args1, args2)
        # plot metadata in ax2
        self.plot_metadata(ax1, args1, args2)
        pass

    def plot_spikes(self, filtered=False):
        # if filtered = False, plot raw spikes else plot filtered spikes
        if filtered is True:
            x = self.filtered_spikes
        else:
            x = self.spikes
        plt.plot(x, x, label='linear')
        plt.legend()
        plt.show()
        pass

    def plot_metadata(self, ax, args1, args2):
        pass


class Slice(Trial):
    def __init__(self, session_file=None, save_session_as=None):
        """ Contains all data in one session. A session object can be stored or loaded from a pickle file.
        If no session_file is specified, Session is generated by reading the source files specified in settings.py"""
        if session_file is None:
            session_dict = make_session()
        else:
            session_dict = load_pickle(session_file)
        if save_session_as is not None:
            save_as_pickle(save_session_as, session_dict)
        self.spikes = session_dict["spikes"]
        # self.spikes_dense = session_dict["spikes_dense"]
        self.licks = session_dict["licks"]
        self.position_x = session_dict["position_x"]
        self.position_y = session_dict["position_y"]
        self.speed = session_dict["speed"]
        self.trial_timestamp = session_dict["trial_timestamp"]
        self.filtered_spikes = None
        self._is_convolved = False
        self._filter = None
        pass

    def set_filter(self, filter):
        self._filter = filter
        self._convolve()

    def _convolve(self):
        # do some stuff
        self._is_convolved = True

    @classmethod
    def from_data(cls, spikes, licks, position_x, position_y, speed,
                  trial_timestamp, filter=None):
        self.spikes = spikes
        self.licks = licks
        self.position_x = position_x
        self.position_y = position_y
        self.speed = speed
        self.trial_timestamp = trial_timestamp
        if filter is None and filtered_spikes is not None:
            self.filtered_spikes = None
            self.set_filter(filter)
        self._is_convolved = False
        self._filter = None

    def __getitem__(self, time_slice):
        if not isinstance(time_slice, slice):
            raise TypeError("Key must be a slice")
        pass

    def slice(self, start, stop):
        """ returns a slice object containing a subsection of the time"""
        spikes = np.array([splice_nparray_at_start_stop(x, start, stop) for x in self.spikes])
        # TODO all entries filters are not ready
        # spikes_dense = np.array([[y for ind2, y in enumerate(x) if stop.ms>ind2>start.ms] for ind1, x in enumerate(self.spikes_dense[...])])  # TODO: not functional
        licks = np.array([splice_nparray_at_start_stop(x, start, stop) for x in self.licks])
        position_x = np.array([x for ind, x in enumerate(self.position_x) if stop.ms > ind >= start.ms])
        position_y = np.array([x for ind, x in enumerate(self.position_y) if stop.ms > ind >= start.ms])
        speed = np.array([x for ind, x in enumerate(self.speed) if stop.ms > ind >= start.ms])
        trial_timestamp = np.asarray(splice_nparray_at_start_stop2([x for x in self.trial_timestamp], start, stop))
        return Slice.from_data(spikes=spikes, licks=licks, position_x=position_x,
                     position_y=position_y, speed=speed, trial_timestamp=trial_timestamp,
                     slice_metadata=slice_metadata,
                     enriched_metadata=enriched_metadata)