import numpy as np
from session_loader import read_file
from settings import save_as_pickle, load_pickle
import matplotlib.pyplot as plt


class Slice:
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
        start = self.trial_timestamp[trial_id][1]
        stop = self.trial_timestamp[trial_id + 1][1]
        start_well = self.trial_timestamp[trial_id - 1][0]
        stop_well = self.trial_timestamp[trial_id - 1][0]
        return self.make_slice(start, stop).make_trial(trial_id=trial_id, start_time=start, stop_time=stop,
                                                       start_well=start_well, stop_well=stop_well, trial_metadata=None)

    def get_trial_by_time(self, trial_time):
        start = np.argmax(self.trial_timestamp[..., 1] > trial_time)
        trial_id = np.where(self.trial_timestamp[..., 1] >= start)
        trial_id = trial_id[0][0]
        return self.get_trial_by_id(trial_id)

    def get_all_trials_by_time(self, start=None, stop=None, max_length=None, min_length=None):
        """ returns a list of Trial objects corresponding to the trials contained in that slice """
        for ind in range(1, len(self.trial_timestamp)):  # the first intended lick is always at well 1
            return_array = np.empty(0, dtype=object)
            last_well = self.trial_timestamp[ind - 1][0]
            last_time = self.trial_timestamp[ind - 1][1]
            current_well = self.trial_timestamp[ind][0]
            current_time = self.trial_timestamp[ind][1]
            trial_duration = current_time - last_time
            trial_id = self.trial_timestamp[1]
            if start <= last_time or start is None:
                if stop >= current_time or stop is None:
                    if max_length is None or max_length < trial_duration:
                        if min_length is None or min_length > trial_duration:
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



def slice_spikes(spikes, time_slice):
    return [t for t in spikes if time_in_slice_list(t, time_slice)]


def slice_list_of_dict(li, time_slice):
    return [d for d in li if time_in_slice(d["time"], time_slice)]


def slice_array(a, time_slice, sample_freq=1000):
    start = int(time_slice.start * sample_freq / 1000)
    stop = int(time_slice.stop * sample_freq / 1000)
    return a[start:stop]


def time_in_slice(time, time_slice):
    return time_slice.start < time < time_slice.stop

def time_in_slice_list(time, time_slice):
    return [x for ind, x in enumerate(time) if time_slice.start < x < time_slice.stop]

class Trial:
    def set_filter(self, filter):
        self._filter = filter
        self._convolve()

    def _convolve(self):
        # do some stuff
        self._is_convolved = True

    @property
    def is_convolved(self):
        return self._is_convolved

    def __getitem__(self, time_slice):
        if not isinstance(time_slice, slice):
            raise TypeError("Key must be a slice, got {}".format(type(time_slice)))
        spikes = slice_spikes(self.spikes, time_slice)
        licks = slice_list_of_dict(self.licks, time_slice)
        position_x = slice_array(self.position_x, time_slice)
        position_y = slice_array(self.position_y, time_slice)
        speed = slice_array(self.speed, time_slice)
        trial_timestamp = slice_list_of_dict(self.trial_timestamp, time_slice)
        _filter = None
        return Slice(spikes, licks, position_x, position_y, speed,
                     trial_timestamp, _filter=_filter)

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
    def __init__(self, spikes, licks, position_x, position_y, speed,
                 trial_timestamp, _filter=None):
        # list for every neuron of lists of times at which a spike occured
        self.spikes = spikes
        # list of dictionaries each dict being:
        # {"time_detection":float, "time":float, "well":0/1/2/3/4/5, "correct":True/False}
        self.licks = licks
        # np arrays
        self.position_x = position_x
        self.position_y = position_y
        self.speed = speed
        # list of dictionaries {"time":float, "well":0/1/2/3/4/5} for every rewarded lick
        self.trial_timestamp = trial_timestamp
        if _filter is None:
            self.filtered_spikes = None
        else:
            self.set_filter(filter)

    @classmethod
    def from_path(cls, load_from=None, save_as=None):
        """ Contains all data in one session. A session object can be stored or loaded from a pickle file.
        If no session_file is specified, Session is generated by reading the source files specified in settings.py"""
        if load_from is None:
            session_dict = read_file()
        else:
            session_dict = load_pickle(load_from)
        if save_as is not None:
            save_as_pickle(save_as, session_dict)
        spikes = session_dict["spikes"]
        licks = session_dict["licks"]
        position_x = session_dict["position_x"]
        position_y = session_dict["position_y"]
        speed = session_dict["speed"]
        trial_timestamp = session_dict["trial_timestamp"]
        return cls(spikes, licks, position_x, position_y, speed,
                   trial_timestamp)

    def get_all_trials(self, time_slice):
        """ returns a list of Trial objects corresponding to the trials contained in that slice """
        if not isinstance(time_slice, slice):
            raise TypeError("Key must be a slice, got {}".format(type(time_slice)))
        start = time_slice.start
        stop = time_slice.stop
        for ind in range(1, len(self.trial_timestamp)):  # the first intended lick is always at well 1
            return_array = np.empty(0, dtype=object)
            last_well = self.trial_timestamp[ind - 1][0]
            last_time = self.trial_timestamp[ind - 1][1]
            current_well = self.trial_timestamp[ind][0]
            current_time = self.trial_timestamp[ind][1]
            trial_id = self.trial_timestamp[1]
            if start <= last_time or start is None:
                if stop >= current_time or stop is None:
                    return_array = np.concatenate((return_array, [
                        self.make_trial(trial_id=trial_id, start_time=last_time, stop_time=current_time,
                                        start_well=last_well, stop_well=current_well, trial_metadata=None)]),
                                                  axis=0)
        return return_array