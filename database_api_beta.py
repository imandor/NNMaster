import numpy as np
from math import ceil
from session_loader import read_file, find_max_time, find_min_time
from settings import save_as_pickle, load_pickle
from session_loader import make_dense_np_matrix
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from settings import config
import plotly.plotly as py
import tensorflow as tf
import matplotlib.mlab as mlab


def subtract_from_list(li, number):
    return [x - number for x in li]


def filter_trials_by_well(trials, start_well=None, end_well=None, well=None):
    return [trial for trial in trials if
            (int(trial.trial_timestamp[0]["trial_lickwell"]) == start_well
             or start_well == None
             or int(trial.trial_timestamp[0]["trial_lickwell"]) == well)
            and
            (int(trial.trial_timestamp[1]["trial_lickwell"]) == end_well
             or end_well == None
             or int(trial.trial_timestamp[1]["trial_lickwell"]) == well)

            ]


def find_longest_trial(trials):
    max_range = 0
    for trial in trials:
        try:
            max = trial.trial_timestamp[1]["time"]
            min = trial.trial_timestamp[0]["time"]
            trial_range = max - min
            if trial_range > max_range: max_range = trial_range
        except IndexError:
            None
    return max_range

def plot_time_x_trials(trials, neuron_no, max_range=None):
    # find maximum range
    trial_spike_list = []
    if max_range is None:
        max_range = find_longest_trial(trials)

    # plot trials

    fig, axes = plt.subplots(nrows=len(trials), sharex=True, sharey=True)
    plt.yticks([])
    plt.xticks([])
    plt.suptitle(config["image_labels"]["trial_spikes_title"])
    # plt.subplots_adjust(hspace=0)
    # pylab.yaxis.set_label_position(config["image_labels"]["trial_spikes_y1_left"])
    fig.text(0.5, 0.04, config["image_labels"]["trial_spikes_x1"], ha='center', va='center')
    fig.text(0.06, 0.5, config["image_labels"]["trial_spikes_y1_left"], ha='center', va='center', rotation='vertical')
    fig.text(0.94, 0.5, config["image_labels"]["trial_spikes_y1_left"], ha='center', va='center', rotation='vertical')
    for ind in range(0, len(axes)):
        trial = trials[ind]
        xmin = trial.start_time
        xmax = trial.start_time + max_range
        ax = axes[ind]
        data = subtract_from_list(trial.spikes[neuron_no], xmin)
        start_lick = trial.trial_timestamp[0]["time"] - xmin
        end_lick = trial.trial_timestamp[1]["time"] - xmin
        ax.vlines(start_lick, 0, 1, colors=['g'])
        ax.vlines(end_lick, 0, 1, colors=['r'])
        ax.set_xlim(left=0, right=xmax - xmin)
        ax.set_ylabel(trial.trial_timestamp[0]["trial_id"], rotation="horizontal", labelpad=10,verticalalignment="center")
        ax.vlines(data, 0, 1)
        ax.set_yticks([])
    save_path = config["paths"]["figure_path"] + "spike_times_single_neuron" + str(neuron_no) + "_" + ".png"
    plt.savefig(save_path, bbox_inches='tight')  # TODO add session date to session object and save image to file
    # plt.show(block=True)
    plt.close(fig)
    pass


def map_spikes_to_position(time_slice,neuron_nos = None):
    """ returns an array containing the positions corresponding to each spike for each neuron. if neuron_nos is given, only the given neurons are returned"""
    spikes = time_slice.spikes
    position_x = time_slice.position_x
    start_time = int(time_slice.start_time)
    max_position = int(np.amax(position_x))
    return_array = []
    if neuron_nos is None or []:
        i_range = range(0,len(spikes))
    else:
        i_range = neuron_nos
    for i in i_range:
        dense_spikes = [0]*max_position
        for j in range(0,len(spikes[i])):
            pos = int(position_x[int(spikes[i][j]-start_time)])-1
            dense_spikes[pos] = dense_spikes[pos]+1
        # np.trim_zeros(dense_spikes, trim='b') # remove trailing zeros
        return_array.append(dense_spikes)
    return return_array


def plot_positionx_x_trials(trials, neuron_no, max_range=None):
    # find maximum range
    trial_spike_list = []
    if max_range is None:
        max_range = find_longest_trial(trials)

    # plot trials

    fig, axes = plt.subplots(nrows=len(trials), sharex=True, sharey=True)
    plt.yticks([])
    plt.xticks([])
    plt.suptitle(config["image_labels"]["position_spikes_title"])
    # plt.subplots_adjust(hspace=0)
    # pylab.yaxis.set_label_position(config["image_labels"]["trial_spikes_y1_left"])
    fig.text(0.5, 0.04, config["image_labels"]["position_x1"], ha='center', va='center')
    fig.text(0.06, 0.5, config["image_labels"]["position_y1_left"], ha='center', va='center', rotation='vertical')
    fig.text(0.94, 0.5, config["image_labels"]["position_y1_left"], ha='center', va='center', rotation='vertical')

    for ind in range(0, len(axes)):
        trial = trials[ind]

        ax = axes[ind]
        data = map_spikes_to_position(trial,[neuron_no])[0]

        data_trimmed = np.trim_zeros(data)
        # data_trimmed_left = np.trim_zeros(data, trim='f')
        # data_trimmed_right = np.trim_zeros(data, trim='b')
        # xmin = len(data) - len(data_trimmed_left) #TODO fix data trim or change to default trimmed data
        # xmax = len(data_trimmed_right)   # 000123400000 - 0001234
        # start_lick = trial.trial_timestamp[0]["time"] - xmin
        # end_lick = trial.trial_timestamp[1]["time"] - xmin
        # ax.vlines(start_lick, 0, 1, colors=['g'])
        # ax.vlines(end_lick, 0, 1, colors=['r'])

        ax.set_xlim(left=0)
        ax.set_ylabel(trial.trial_timestamp[0]["trial_id"], rotation="horizontal", labelpad=10,verticalalignment="center")
        # ax.vlines(data, 0, 1)
        ax.bar(range(len(data)), data, width=1, align='center', color='black', zorder=3)
        # ax.grid(b=True, which='major', color='b', linestyle='-')
        ax.set_yticks([])
    save_path = config["paths"]["figure_path"] + "spike_positions_single_neuron_" + str(neuron_no) + "_" + ".png"
    plt.savefig(save_path, bbox_inches='tight')  # TODO add session date to session object and save image to file
    # plt.show(block=True)
    plt.close(fig)
    pass


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
    return [slice_array(t, slice(start[ind], stop[ind])) for ind, t in enumerate(spikes)]


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
        trial = phase[i].get_nth_trial(n)
        if trial is not None: return_list.append(trial)
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
            for n in range(0, len(dense_spikes)):
                c = 0
                for m in range(-window, window + 1, step_size):
                    if n - m >= 0 and n - m < len(dense_spikes):  # cut off edges
                        self._filter(dense_spikes[m])
                        c = c + (dense_spikes[n - m] * self._filter(dense_spikes[m]))
                filtered_spike.append(c)
            self.filtered_spikes.append(filtered_spike)

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
            y = self.filtered_spikes[8]
        else:
            y = self.spikes[8]
        x = np.arange(len(self.position_x))
        width = y[-1] / 400
        plt.bar(y, height=1, align='center', width=width)
        # plt.xticks(y,x)
        plt.ylabel('values')
        plt.title('some plot')
        plt.show(block=True)

        # add some text for labels, title and axes ticks
        # ax.set_ylabel('Scores')
        # ax.set_title('Scores by group and gender')
        # ax.set_xticks(ind + width / 2)
        # ax.set_xticklabels(('G1', 'G2', 'G3', 'G4', 'G5'))

        # ax.legend((rects1[0]), ('Men'))
        # x = np.arange(y[0][0],y[0][-1])
        # plt.plot(y[0], label='linear')
        # plt.legend()
        # plt.show(block=True)
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
            print("Warning: Slice does not contain " + str(n) + "th trial")
            return None

        # start_well = self.trial_timestamp[trial_id - 1]["trial_lickwell"]
        # stop_well = self.trial_timestamp[trial_id - 1]["trial_lickwell"]
        return self[start:stop]

    def get_trial_by_time(self, trial_time):
        # TODO must be updated to new trial timestamp format
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
                if stop is None or stop >= current_time or stop == -1:
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
