import numpy as np
from session_loader import read_file
from src.settings import save_as_pickle, load_pickle
import matplotlib.pyplot as plt
from src.settings import config
from itertools import takewhile, dropwhile

well_to_color = {0: "#ff0000", 1: "#669900", 2: "#0066cc", 3: "#cc33ff", 4: "#003300", 5: "#996633"}


def subtract_from_list(li, number):
    """
    :param li: python list
    :param number: a number
    :return: list with number subtracted from each list entry
    """
    return [x - number for x in li]


class SliceList: # TODO: getitem returns a regular list here
    # def __init__(self, *args):
    #     list.__init__(self, *args)

    def __init__(self, container=[]): #TODO: check problems with commenting this
        self.container = container

    def __getitem__(self, time_slice):
        return self.container[time_slice]

    def add_slice(self, slice):
        """
        :param slice: Slice object
        :return: adds object to list
        """
        self.container.append(slice)

    def get_all(self):
        """
        :return: returns entire list
        """
        return self.container[:]

    def __eq__(self, other):
        """
        :param other: lists to be compared with
        :return: True if both lists are identical, else false
        """
        return self.__dict__ == other.__dict__


    def add_slice(self, slice):
        """
        :param slice: Slice object
        :return: adds object to list
        """
        self.container.append(slice)

    def get_all(self):
        """
        :return: returns entire list
        """
        return self.container[:]

    def __eq__(self, other):
        """
        :param other: lists to be compared with
        :return: True if both lists are identical, else false
        """
        return self.__dict__ == other.__dict__

    def filter_trials_by_well(self, start_well=None, end_well=None, well=None):
        """
        removes all trials which don't fit the input
        :param start_well: keep all trials containing start_well
        :param end_well:  keep all trials containing end_well
        :param well: keep all trials containing well
        :return: Slices object containing all trials fitting input
        """
        return TrialList([trial for trial in self if
                          (int(trial.trial_timestamp[0]["trial_lickwell"]) == start_well
                           or start_well == None
                           or int(trial.trial_timestamp[0]["trial_lickwell"]) == well)
                          and
                          (int(trial.trial_timestamp[1]["trial_lickwell"]) == end_well
                           or end_well == None
                           or int(trial.trial_timestamp[1]["trial_lickwell"]) == well)
                          ])


class TrialList(SliceList):
    def __init__(self, *args):
        super().__init__(*args)



    def plot_positionx_x_trials(self, neuron_no, max_range=None):
        """
        Plotting function, creates subplots containing trials by time
        :param neuron_no: neuron to be plotted
        :param max_range: length of shown range in plot
        :return: None
        """
        fig, axes = plt.subplots(nrows=len(self.get_all()), sharex=True, sharey=True)
        plt.yticks([])
        plt.xticks([])
        plt.suptitle(config["image_labels"]["position_spikes_title"])
        fig.text(0.5, 0.04, config["image_labels"]["position_x1"], ha='center', va='center')
        fig.text(0.06, 0.5, config["image_labels"]["position_y1_left"], ha='center', va='center', rotation='vertical')
        fig.text(0.94, 0.5, config["image_labels"]["position_y1_left"], ha='center', va='center', rotation='vertical')

        for ind in range(0, len(axes)):
            trial = self[ind]
            ax = axes[ind]
            data = trial.map_spikes_to_position([neuron_no])[0]
            ax.set_ylabel(trial.trial_timestamp[0]["trial_id"], rotation="horizontal", labelpad=10,
                          verticalalignment="center")
            ax.bar(range(len(data)), data, width=1, align='center', color='black', zorder=3)
            ax.set_yticks([])
        save_path = config["paths"]["figure_path"] + "spike_positions_single_neuron_" + str(neuron_no) + "_" + ".png"
        plt.savefig(save_path, bbox_inches='tight')  # TODO add session date to session object and save image to file
        plt.close(fig)
        pass

    def range_of_longest_trial(self):
        """
        :return: range of longest trial
        """
        max_range = 0
        for trial in self:
            try:
                max = trial.trial_timestamp[1]["time"]
                min = trial.trial_timestamp[0]["time"]
                trial_range = max - min
                if trial_range > max_range: max_range = trial_range
            except IndexError:
                None
        return max_range

    def set_trial_ax(self, data, trial, ind, axes, xmin, xmax):
        """

        :param data:
        :param trial:
        :param ind:
        :param axes:
        :param xmin:
        :param xmax:
        :return:
        """
        ax = axes[ind]

        start_lick = trial.trial_timestamp[0]["time"] - xmin
        end_lick = trial.trial_timestamp[1]["time"] - xmin
        ax.vlines(start_lick, 0, 1, colors=['g'])
        ax.vlines(end_lick, 0, 1, colors=['r'])
        ax.set_xlim(left=0, right=xmax - xmin)
        ax.set_ylabel(trial.trial_timestamp[0]["trial_id"], rotation="horizontal", labelpad=10,
                      verticalalignment="center")
        ax.vlines(data, 0, 1)
        ax.set_yticks([])
        return ax

    def plot_time_x_trials(self, neuron_no, max_range=None):
        """
        :param neuron_no:
        :param max_range:
        :return:
        """
        # find maximum range for image size adjusting
        if max_range is None:
            max_range = self.range_of_longest_trial()

        # plot trials
        fig, axes = plt.subplots(nrows=len(self.get_all()), sharex=True, sharey=True)
        plt.yticks([])
        plt.xticks([])
        plt.suptitle(config["image_labels"]["trial_spikes_title"])
        fig.text(0.5, 0.04, config["image_labels"]["trial_spikes_x1"], ha='center', va='center')
        fig.text(0.06, 0.5, config["image_labels"]["trial_spikes_y1_left"], ha='center', va='center',
                 rotation='vertical')
        fig.text(0.94, 0.5, config["image_labels"]["trial_spikes_y1_left"], ha='center', va='center',
                 rotation='vertical')
        for ind in range(0, len(axes)):
            trial = self[ind]
            xmin = trial.start_time
            xmax = trial.start_time + max_range
            data = subtract_from_list(trial.spikes[neuron_no], xmin)
            ax = self.set_trial_ax(data=data, trial=trial, ind=ind, axes=axes, xmin=xmin, xmax=xmax)
        save_path = config["paths"]["figure_path"] + "spike_times_single_neuron" + str(neuron_no) + "_" + ".png"
        plt.savefig(save_path, bbox_inches='tight')  # TODO add session date to session object and save image to file
        # plt.show(block=True)
        plt.close(fig)
        pass


class PhaseList(SliceList):
    def __init__(self,*args):
        super().__init__(*args)


    def get_nth_trial_in_each_phase(self, n):
        """
        :param n: trial index in list of phases
        :return: list of nth trials
        """
        trials = TrialList()
        for phase in self:
            try:
                trial = phase.get_nth_trial(n)
                trials.append(trial)
            except IndexError:
                pass
        return trials
    ## def __init__(self, container=[]): #TODO: check problems with commenting this
    ##     self.container = container
    #
    ## def __getitem__(self, time_slice):
    ##     return self.container[time_slice]
    ##
    ## def add_slice(self, slice):
    ##     """
    ##     :param slice: Slice object
    ##     :return: adds object to list
    ##     """
    ##     self.container.append(slice)
    ##
    ## def get_all(self):
    ##     """
    ##     :return: returns entire list
    ##     """
    ##     return self.container[:]
    ##
    ## def __eq__(self, other):
    ##     """
    ##     :param other: lists to be compared with
    ##     :return: True if both lists are identical, else false
    ##     """
    ##     return self.__dict__ == other.__dict__
    ##
    ## def get_nth_trial_in_each_phase(self, n):
    ##     """
    ##     :param n: trial index in list of phases
    ##     :return: list of nth trials
    ##     """
    ##     return_list = []
    ##     for i in range(0, len(self.get_all())):
    ##         slice = self[i].get_nth_trial(n)
    ##         if slice is not None: return_list.append(slice)
    ##     return return_list


class Trial:

    def __str__(self):
        return str(self.__dict__)

    def __eq__(self, other):
        if self.spikes == other.spikes and self.licks == other.licks \
                and np.array_equal(self.position_x, other.position_x) \
                and np.array_equal(self.position_y, other.position_y) \
                and np.array_equal(self.speed, other.speed) \
                and self.trial_timestamp == other.trial_timestamp \
                and self.start_time == other.start_time:
            return True
        else:
            return False

    def set_filter(self, filter, search_window_size, step_size=1):
        self._filter = filter
        self._search_window_size = search_window_size
        if filter is not None and search_window_size is not None:
            self._convolve(search_window_size=search_window_size, step_size=step_size)
        else:
            self._is_convolved = False
            self.filtered_spikes = None

    def _convolve(self, search_window_size, step_size):
        if step_size > search_window_size:
            raise ValueError("step_size must be inferior to search_window_size")
        n_bin_points = int(len(self.position_x) // step_size)
        self.filtered_spikes = np.zeros((len(self.spikes), n_bin_points + 1))
        for neuron_index, neuron_spikes in enumerate(self.spikes):
            print("[{:4d}/{:4d}] Convolving...".format(neuron_index + 1, self.n_neurons), end="\r")
            curr_search_window_min_bound = self.start_time - search_window_size / 2
            curr_search_window_max_bound = self.start_time + search_window_size / 2
            index_first_spike_in_window = 0
            for index in range(n_bin_points):
                curr_spikes_in_search_window = dropwhile(lambda x: x < curr_search_window_min_bound,
                                                         takewhile(lambda x: x < curr_search_window_max_bound,
                                                                   neuron_spikes[index_first_spike_in_window:]))
                self.filtered_spikes[neuron_index][index] = sum(map(
                    lambda x: self._filter(x - index * step_size - self.start_time),
                    curr_spikes_in_search_window))
                curr_search_window_min_bound += step_size
                curr_search_window_max_bound += step_size
                index_first_spike_in_curr_window = 0
                for spike_index, spike in enumerate(curr_spikes_in_search_window):
                    if spike >= curr_search_window_min_bound:
                        index_first_spike_in_window = spike_index
                        break
                index_first_spike_in_window += index_first_spike_in_curr_window
        print("")
        self._is_convolved = True

    @property
    def is_convolved(self):
        return self._is_convolved

    def plot_filtered_spikes(self, ax, neuron_px_height=4, normalize_each_neuron=False, margin_px=2):
        if not self.is_convolved:
            raise ValueError("This slice is not convolved. First set a filter please.")
        n_bins = self.filtered_spikes.shape[1]
        image = np.zeros((self.n_neurons * (neuron_px_height + margin_px) - margin_px, n_bins))
        for i, f in enumerate(self.filtered_spikes):
            n = (f - np.min(f)) / (np.max(f) - np.min(f)) if normalize_each_neuron else f
            image[i * (neuron_px_height + margin_px): (i + 1) * (neuron_px_height + margin_px) - margin_px] = n
        ax.imshow(image, cmap='hot', interpolation='none', aspect='auto')

    def plot_raw_spikes(self, ax):
        for i, spike in enumerate(reversed(self.spikes)):
            print("[{:4d}/{:4d}] Ploting raw spike data...".format(i + 1, self.n_neurons), end="\r")
            for s in spike:
                ax.vlines(s, i, i + 0.8)
        print("")

    def plot(self, ax_filtered_spikes=None, ax_raw_spikes=None, ax_licks=None, ax_trial_timestamps=None,
             filtered_spikes_kwargs={}):
        share_axis_set = set([ax_raw_spikes, ax_licks, ax_trial_timestamps])
        share_axis_set.discard(None)
        if len(share_axis_set) > 2:
            reference_ax = share_axis_set.pop()
            for other_ax in share_axis_set:
                reference_ax.get_shared_x_axes().join(reference_ax, other_ax)
        if ax_filtered_spikes is not None:
            self.plot_filtered_spikes(ax_filtered_spikes, **filtered_spikes_kwargs)
        if ax_raw_spikes is not None:
            self.plot_raw_spikes(ax_raw_spikes)
        if ax_licks is not None:
            self.plot_licks(ax_licks)
        if ax_trial_timestamps is not None:
            self.plot_trial_timestamps(ax_trial_timestamps)

    def plot_licks(self, ax):
        min_time = 1e10
        max_time = -1
        for d in self.licks:
            well = d["lickwell"]
            time = d["time"]
            min_time = min_time if time > min_time else time
            max_time = max_time if time < max_time else time
            bbox = dict(boxstyle="circle", fc=well_to_color[well], ec="k")
            ax.text(time, 0, str(well), bbox=bbox)
        ax.hlines(0, min_time, max_time)

    def plot_trial_timestamps(self, ax):
        min_time = 1e10
        max_time = -1
        for d in self.trial_timestamp:
            well = d["trial_lickwell"]
            time = d["time"]
            min_time = min_time if time > min_time else time
            max_time = max_time if time < max_time else time
            bbox = dict(boxstyle="circle", fc=well_to_color[well], ec="k")
            ax.text(time, 0, str(well), bbox=bbox)
        ax.hlines(0, min_time, max_time)

    def plot_metadata(self, ax1, ax2):
        self.plot_licks(ax1)
        self.plot_trial_timestamps(ax2)


class Slice(Trial):
    def __init__(self, spikes, licks, position_x, position_y, speed,
                 trial_timestamp, start_time, _filter=None, _search_window_size=None):
        # list for every neuron of lists of times at which a spike occured
        self.spikes = spikes
        self.n_neurons = len(self.spikes)
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
        self.set_filter(filter=_filter, search_window_size=_search_window_size)

    def time_in_slice(self, time, time_slice):
        if time_slice.stop is None:
            return time_slice.start <= time
        else:
            return time_slice.start <= time <= time_slice.stop

    def slice_spikes(self, time_slice):
        new_spikes = []
        start = time_slice.start
        stop = time_slice.stop
        if stop is None: stop = np.inf
        for spike in self.spikes:
            new_spikes.append([i for i in
                               dropwhile(lambda x: x < start,
                                         takewhile(lambda x: x <= stop, spike))])
        return new_spikes

    def slice_list_of_dict(self, ld, time_slice):
        return [d for d in ld if self.time_in_slice(d["time"], time_slice)]

    def slice_array(self, a, time_slice, sample_freq=1000):
        start = int(time_slice.start * sample_freq / 1000)
        if time_slice.stop is not None:
            stop = int(time_slice.stop * sample_freq / 1000)
            return a[start:stop]
        else:
            return a[start:]

    def __getitem__(self, time_slice):
        if not isinstance(time_slice, slice):
            raise TypeError("Key must be a slice, got {}".format(type(time_slice)))
        # normalize start/stop for dense parameters, which always start at index = 0:
        start = time_slice.start + self.start_time
        stop = None if time_slice.stop is None else time_slice.stop + self.start_time
        offset_slice = slice(start, stop)
        spikes = self.slice_spikes(offset_slice)
        licks = self.slice_list_of_dict(self.licks, offset_slice)
        position_x = self.slice_array(self.position_x, time_slice)
        position_y = self.slice_array(self.position_y, time_slice)
        speed = self.slice_array(self.speed, time_slice)
        trial_timestamp = self.slice_list_of_dict(self.trial_timestamp, offset_slice)
        _filter = self._filter
        return Slice(spikes, licks, position_x, position_y, speed, trial_timestamp, start,
                     _filter=_filter, _search_window_size=self._search_window_size)

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
        if n + 1 >= len(self.trial_timestamp):
            raise IndexError("Slice does not contain {} trials but only {}".format(n, len(self.trial_timestamp) - 1))
        start = self.trial_timestamp[n]["time"]
        stop = self.trial_timestamp[n + 1]["time"]
        return self[start:stop]

    ## def get_trial_by_time(self, trial_time):
    ##     # TODO must be updated to new trial timestamp format
    ##     start = np.argmax(self.trial_timestamp[..., 1] > trial_time)
    ##     trial_id = np.where(self.trial_timestamp[..., 1] >= start)
    ##     trial_id = trial_id[0][0]
    ##     return self.get_nth_trial(trial_id)

    def get_trials(self, time_slice):
        """
        returns a list of Trial objects corresponding to the trials contained in that slice
        :param time_slice: Slice object
        :return: Slices object containing all trials in time_slice
        """
        if not isinstance(time_slice, slice):
            raise TypeError("Key must be a slice, got {}".format(type(time_slice)))
        start = time_slice.start
        stop = time_slice.stop
        return_array = []
        for ind in range(1, len(self.trial_timestamp)):  # trial finishing in slice is also valid
            last_time = self.trial_timestamp[ind - 1]["time"]
            current_time = self.trial_timestamp[ind]["time"]
            if start <= last_time or start is None:
                if stop is None or stop >= current_time or stop == -1:
                    s = slice(last_time, current_time)
                    return_array.append(self[s])
        return SliceList(return_array)

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
        return PhaseList(return_array)

    def get_all_phases(self):
        s = slice(0, None)
        return self.get_phases(s)

    def map_spikes_to_position(self, neuron_nos=None):
        """
        returns an array containing the positions corresponding to each spike for each neuron.
        :param neuron_nos: if neuron_nos is given, only the given neurons are returned
        :return: list of number of spikes for each position
        """
        spikes = self.spikes
        position_x = self.position_x
        start_time = int(self.start_time)
        max_position = int(np.amax(position_x))
        return_array = np.zeros((self.n_neurons, max_position))
        neuron_nos = neuron_nos if neuron_nos is not None else range(self.n_neurons)
        for i in neuron_nos:
            for j in range(0, len(spikes[i])):
                pos = int(position_x[int(spikes[i][j] - start_time)]) - 1
                return_array[i][pos] += 1
        return return_array
