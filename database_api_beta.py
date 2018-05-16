import numpy as np
from math import ceil
from session_loader import read_file, find_max_time
from settings import save_as_pickle, load_pickle
import matplotlib.pyplot as plt


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
        # for i in range(0, len(self.spikes)):
        #     self.filtered_spikes = np.convolve(self.spikes,
        #                                        self._filter)  # TODO spikes would have to be dense here for proper convolution
        self._is_convolved = True

    def bin_spikes(self, binarize=False, bin_size=1):
        if self._filter is None:
            self.bin_index_spikes(bin_size=bin_size)
        for i in range(0, len(self.filtered_spikes)):
            bin_amount = ceil(find_max_time(self.spikes))*bin_size #TODO
            new_spikes = np.zeros(bin_amount*bin_size, dtype=int)
            for j in range(0, len(self.filtered_spikes[i])):
                new_spikes[j] = new_spikes[self.filtered_spikes[i][j]] + 1
            if binarize is True:
                new_spikes = np.where(new_spikes > 0, 1, 0)
            self.filter = "bins"
            self.filtered_spikes[i] = np.ndarray.tolist(new_spikes)

    def bin_index_spikes(self, bin_size=1):
        """ sets filtered_spikes to the index of spikes in a binned list with len(spikes)/bin_size entries"""
        max_time = ceil(find_max_time(self.spikes))
        min_time = 0
        bins = np.arange(min_time, max_time, bin_size)
        self.filtered_spikes = self.spikes
        self.filter = "bin_index"

        for i in range(0, len(self.spikes)):
            new_spikes = np.digitize(self.spikes[i], bins)

            self.filtered_spikes[i] = np.ndarray.tolist(new_spikes)

        pass

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


class Trials:
    def __init__(self):
        self.container = []

    def __getitem__(self, time_slice):
        return self.container[time_slice]

    def add_trial(self, trial):
        # if not isinstance(trial, Trial):
        #     raise TypeError("Key must be a Trial, got {}".format(type(trial)))
        self.container.append(trial)

    def get_all(self):
        return self._container[:]


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
            self.set_filter(filter=None)
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

    def get_trial_by_id(self, trial_id):
        start = self.trial_timestamp[trial_id]["time"]
        stop = self.trial_timestamp[trial_id + 1]["time"]
        start_well = self.trial_timestamp[trial_id - 1]["trial_lickwell"]
        stop_well = self.trial_timestamp[trial_id - 1]["trial_lickwell"]
        return self[start:stop]

    def get_trial_by_time(self, trial_time):
        start = np.argmax(self.trial_timestamp[..., 1] > trial_time)
        trial_id = np.where(self.trial_timestamp[..., 1] >= start)
        trial_id = trial_id[0][0]
        return self.get_trial_by_id(trial_id)

    def get_trials(self, time_slice):
        """ returns a list of Trial objects corresponding to the trials contained in that slice """
        if not isinstance(time_slice, slice):
            raise TypeError("Key must be a slice, got {}".format(type(time_slice)))
        start = time_slice.start
        stop = time_slice.stop
        return_array = Trials()
        for ind in range(1, len(self.trial_timestamp)):  # the first intended lick is always at well 1
            last_well = self.trial_timestamp[ind - 1]["trial_lickwell"]
            last_time = self.trial_timestamp[ind - 1]["time"]
            current_well = self.trial_timestamp[ind]["trial_lickwell"]
            current_time = self.trial_timestamp[ind]["time"]
            trial_id = ind - 1
            if start <= last_time or start is None:
                if stop >= current_time or stop is None or stop == -1:
                    s = slice(last_time, current_time)
                    return_array.add_trial(self[s])
        return return_array

    def get_all_trials(self):
        s = slice(0, -1)
        return self.get_trials(s)
