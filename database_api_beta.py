import numpy as np
from math import ceil
from session_loader import read_file, find_max_time, find_min_time
from settings import save_as_pickle, load_pickle
from session_loader import make_dense_np_matrix
import matplotlib.pyplot as plt
import tensorflow as tf


def slice_spikes(spikes, time_slice):
    start = []
    stop = []
    for i in range(0, len(spikes)):
        if spikes[i] != []:
            start.append(next((ind for ind, v in enumerate(spikes[i]) if v >= time_slice.start), spikes[i][-1]))
            if time_slice.stop == None:
                stop.append(None)
            else:
                stop.append(next((ind for ind, v in enumerate(spikes[i]) if v > time_slice.stop), None))
        else:
            start.append(0)
            stop.append(None)
    asd = [slice_array(t, slice(start[ind], stop[ind])) for ind, t in enumerate(spikes)]
    return asd


def slice_list_of_dict(li, time_slice):
    return [d for d in li if time_in_slice(d["time"], time_slice)]


def slice_array(a, time_slice, sample_freq=1000):
    start = int(time_slice.start * sample_freq / 1000)
    if time_slice.stop is not None:
        stop = int(time_slice.stop * sample_freq / 1000)
    else:
        stop = None
    return a[start:stop]


def time_in_slice(time, time_slice):
    if time_slice.stop is None:
        return time_slice.start <= time
    else:
        return time_slice.start <= time <= time_slice.stop


def time_in_slice_list(time, time_slice):
    return [x for ind, x in enumerate(time) if time_slice.start < x < time_slice.stop]


def get_nth_trial_in_each_phase(phase, n):
    return_list = []
    for i in range(0, len(phase)):
        return_list.append(phase[i].get_nth_trial(n))
    return return_list


class Trial:
    def set_filter(self, filter, window):
        self._filter = filter
        if filter is not None: self._convolve(window=window)

    def _convolve(self, window, step_size=1):
        self.filtered_spikes = []
        d = make_dense_np_matrix(self.spikes)
        d = np.asarray(d, dtype=float)
        for i in range(0, len(d)):
            dense_spikes = d[i]
            filtered_spike = []
            for n in range(0, len(dense_spikes) - 1):
                c = 0
                for m in range(-window, window + 1, step_size):
                    if n - m >= 0 and n - m < len(dense_spikes):  # cut off edges
                        self._filter(dense_spikes[m])
                        c = c + (dense_spikes[n - m] * self._filter(dense_spikes[m]))
                filtered_spike.append(c)
            self.filtered_spikes.append(filtered_spike)
            if (i % 1000 == 0): print(i)

        # for i in range(0, len(self.spikes)):
        #     self.filtered_spikes = np.convolve(self.spikes,
        #                                        self._filter)  # TODO spikes would have to be dense here for proper convolution
        self._is_convolved = True
        pass

    def bin_spikes(self, binarize=False, binarize_threshold=None, bin_size=1):
        """ obsolete. to be removed"""
        """ sets filtered_spikes to bins the range of the objects spike values and increases value of each bin by one
        for each occurrence of the corresponding value in spikes. If binarize is True, all values are set to 0 or 1
        depending on binarize_threshold"""
        bin_amount = ceil(find_max_time(self.spikes) / bin_size)
        if self._filter is None:
            self.bin_index_spikes(bin_size=bin_size)
        for i in range(0, len(self.filtered_spikes)):
            new_spikes = np.zeros(bin_amount, dtype=int)
            for j in range(0, len(self.filtered_spikes[i])):
                new_spikes[self.filtered_spikes[i][j]] = new_spikes[self.filtered_spikes[i][j]] + 1
            if binarize is True:
                new_spikes = np.where(new_spikes > binarize_threshold, 1, 0)
            self.filter = "bins"
            self.filtered_spikes[i] = np.ndarray.tolist(new_spikes)
            if self.filtered_spikes[i] is []: self.filtered_spikes = np.zeros(bin_amount, dtype=int)
        pass

    def bin_index_spikes(self, bin_size=1):
        """ obsolete. to be removed"""
        """ sets filtered_spikes to the index of spikes in a binned list with len(spikes)/bin_size entries"""
        max_time = (find_max_time(self.spikes))
        min_time = int(find_min_time(self.spikes))
        bins = np.arange(min_time, max_time, bin_size)
        self.filtered_spikes = self.spikes
        self.filter = "bin_index"

        for i in range(0, len(self.spikes)):
            new_spikes = np.digitize(self.spikes[i], bins)
            new_spikes = [x - 1 for x in new_spikes]  # digitize moves index + 1 for some reason
            self.filtered_spikes[i] = new_spikes
        pass

    @property
    def is_convolved(self):
        return self._is_convolved

    def __getitem__(self, time_slice):
        if not isinstance(time_slice, slice):
            raise TypeError("Key must be a slice, got {}".format(type(time_slice)))

        # normalize start/stop for dense parameters, which always start at index = 0:
        if time_slice.stop is None:
            stop = None
        else:
            stop = time_slice.stop - self.start_time
        start = time_slice.start - self.start_time
        normalized_slice = slice(start, stop)
        spikes = slice_spikes(self.spikes, time_slice)
        licks = slice_list_of_dict(self.licks, time_slice)
        position_x = slice_array(self.position_x, normalized_slice)
        position_y = slice_array(self.position_y, normalized_slice)
        speed = slice_array(self.speed, normalized_slice)
        trial_timestamp = slice_list_of_dict(self.trial_timestamp, time_slice)
        _filter = None
        return Slice(spikes=spikes, licks=licks, position_x=position_x, position_y=position_y, speed=speed,
                     trial_timestamp=trial_timestamp, _filter=_filter, start_time=time_slice.start)

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
                 trial_timestamp, start_time, _filter=None):
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
        self.start_time = start_time
        if _filter is None:
            self.filtered_spikes = None
            self.set_filter(filter=None, window=0)
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
        start_time = 0  # trial is always normalized to zero
        return cls(spikes, licks, position_x, position_y, speed,
                   trial_timestamp, start_time)

    def get_nth_trial(self, n):
        if not isinstance(n, int):
            raise TypeError("Key must be an int, got {}".format(type(n)))

        try:
            start = self.trial_timestamp[n]["time"]
            stop = self.trial_timestamp[n + 1]["time"]
        except IndexError:
            print("Error: Trial " + str(n) + " only partially or not in slice")
            return None

        # start_well = self.trial_timestamp[trial_id - 1]["trial_lickwell"]
        # stop_well = self.trial_timestamp[trial_id - 1]["trial_lickwell"]
        return self[start:stop]

    def get_trial_by_time(self, trial_time):
        start = np.argmax(self.trial_timestamp[..., 1] > trial_time)
        trial_id = np.where(self.trial_timestamp[..., 1] >= start)
        trial_id = trial_id[0][0]
        return self.get_nth_trial(trial_id)

    def get_trials(self, time_slice):
        """ returns a list of Trial objects corresponding to the trials contained in that slice """
        if not isinstance(time_slice, slice):
            raise TypeError("Key must be a slice, got {}".format(type(time_slice)))
        start = time_slice.start
        stop = time_slice.stop
        return_array = []
        for ind in range(1, len(self.trial_timestamp)):  # trial finishing in slice is also valid
            last_time = self.trial_timestamp[ind - 1]["time"]
            current_time = self.trial_timestamp[ind]["time"]
            # last_well = self.trial_timestamp[ind - 1]["trial_lickwell"]
            # current_well = self.trial_timestamp[ind]["trial_lickwell"]
            # trial_id = ind - 1
            if start <= last_time or start is None:
                if stop >= current_time or stop is None or stop == -1:
                    s = slice(last_time, current_time)
                    return_array.append(self[s])
        return return_array

    def get_all_trials(self):
        s = slice(0, None)
        return self.get_trials(s)

    def get_phases(self, time_slice):
        """ returns a list of slices corresponding to the training phases contained in that slice """
        if not isinstance(time_slice, slice):
            raise TypeError("Key must be a slice, got {}".format(type(time_slice)))
        return_array = []
        current_time = 0
        last_time = self.trial_timestamp[0]["time"]
        for ind in range(0,
                         len(self.trial_timestamp) - 2):  # phases finishing (but not starting) in slice are also valid
            current_starting_point = self.trial_timestamp[ind]["trial_lickwell"]
            next_starting_point = self.trial_timestamp[ind + 2]["trial_lickwell"]
            if current_starting_point != next_starting_point:
                current_time = self.trial_timestamp[ind + 2]["time"]
                s = slice(last_time, current_time)
                return_array.append(self[s])
                last_time = current_time
        final_slice = slice(current_time, time_slice.stop)
        return_array.append(self[final_slice])
        return return_array

    def get_all_phases(self):
        s = slice(0, None)
        return self.get_phases(s)
