from multiprocessing.pool import ThreadPool
import numpy as np
from itertools import takewhile, dropwhile, repeat
from src import OpenEphys
import scipy
import bisect
import glob
import pickle
from random import seed, randint
from src.preprocessing import generate_counter, fill_counter, normalize_well
import time
import random

well_to_color = {0: "#ff0000", 1: "#669900", 2: "#0066cc", 3: "#cc33ff", 4: "#003300", 5: "#996633"}


def hann(x):
    if np.abs(x) < 1:
        return (1 + np.cos(x * np.pi)) / 2
    else:
        return 0


def bin(x):
    if np.abs(x) < 1:
        return 1
    else:
        return 0


class Filter:
    def __init__(self, func, search_radius, step_size):
        self._func = func
        self.search_radius = search_radius
        self.step_size = step_size
        X = np.linspace(-1, 1, self.search_radius)
        # for x in X:
        #     print(x,":",self._func(x),", ",)
        # print("X - X:",X[1] - X[0])
        self._integral_on_search_window = sum([self._func(x) for x in X]) * (X[1] - X[0])
        if self._integral_on_search_window is np.inf:
            self._integral_on_search_window = 1

    def __call__(self, x):
        # print("func:",self._func(x / self.search_radius), " value is:",x)
        return self._func(x)  # self._func(x / self.search_radius) / self._integral_on_search_window


hann_500_500 = Filter(hann, 500, 500)
hann_500_250 = Filter(hann, 500, 250)
hann_500_100 = Filter(hann, 500, 100)
hann_250_250 = Filter(hann, 250, 250)
hann_250_100 = Filter(hann, 250, 100)
hann_250_50 = Filter(hann, 250, 50)
hann_100_100 = Filter(hann, 100, 100)
hann_100_50 = Filter(hann, 100, 50)
hann_100_25 = Filter(hann, 100, 25)
hann_50_50 = Filter(hann, 50, 50)
hann_50_25 = Filter(hann, 50, 25)
bin_500_500 = Filter(bin, 500, 500)
bin_500_250 = Filter(bin, 500, 250)
bin_500_100 = Filter(bin, 500, 100)
bin_250_250 = Filter(bin, 250, 250)
bin_250_100 = Filter(bin, 250, 100)
bin_250_50 = Filter(bin, 250, 50)
bin_100_100 = Filter(bin, 100, 100)
bin_100_50 = Filter(bin, 100, 50)
bin_100_25 = Filter(bin, 100, 25)
bin_50_50 = Filter(bin, 50, 50)
bin_50_25 = Filter(bin, 50, 25)


def _convolve_thread_func(filter_func, n_bin_points, neuron_counter, n_neurons, neuron_spikes):
    # print("Convolving neuron ",neuron_counter, " of ", n_neurons,"...")
    curr_search_window_min_bound = - filter_func.search_radius
    curr_search_window_max_bound = + filter_func.search_radius
    filtered_spikes = np.zeros(n_bin_points)
    index_first_spike_in_window = 0
    for index in range(n_bin_points):
        curr_spikes_in_search_window = dropwhile(lambda x: x < curr_search_window_min_bound,
                                                 takewhile(lambda x: x < curr_search_window_max_bound,
                                                           neuron_spikes[index_first_spike_in_window:]))
        curr_spikes_in_search_window = list(curr_spikes_in_search_window)
        curr_search_window_min_bound += filter_func.step_size
        curr_search_window_max_bound += filter_func.step_size
        if len(curr_spikes_in_search_window) == 0:
            continue
        # filtered_spikes[index] = (np.average(curr_spikes_in_search_window) - np.average(
        #     [curr_search_window_min_bound, curr_search_window_max_bound])) / len(curr_spikes_in_search_window)
        filtered_spikes[index] = sum(map(
            lambda x: filter_func((x - index * filter_func.step_size) / filter_func.search_radius),
            curr_spikes_in_search_window))
        for spike_index, spike in enumerate(neuron_spikes[
                                            index_first_spike_in_window:index_first_spike_in_window + curr_search_window_max_bound]):  # upper bound because a maximum of 1 spike per ms can occurr and runtime of slice operation is O(i2-i1)
            if spike >= curr_search_window_min_bound:
                index_first_spike_in_window = index_first_spike_in_window + spike_index
                break

    # print("Finished convolving neuron ",neuron_counter, " of ", n_neurons,"...")
    return filtered_spikes


class Lick():
    def __init__(self, lickwell, time, rewarded, lick_id, target=None):
        self.time = time
        self.lickwell = lickwell
        self.rewarded = rewarded
        self.lick_id = lick_id
        self.target = target

class Evaluated_Lick(Lick):  # object containing list of metrics by cross validation partition
    def __init__(self, lickwell, time, rewarded, lick_id, prediction=None, next_lick_id=None, last_lick_id=None,
                 fraction_decoded=None, total_decoded=None,target=None,fraction_predicted=None):
        Lick.__init__(self, lickwell, time, rewarded, lick_id,target)
        self.next_lick_id = next_lick_id
        self.last_lick_id = last_lick_id
        self.fraction_decoded = fraction_decoded
        self.total_decoded = total_decoded
        self.prediction = prediction
        self.fraction_predicted = fraction_predicted
class Net_data:
    WIN_SIZE = 100
    SEARCH_RADIUS = WIN_SIZE * 2

    def __init__(self,
                 model_path,
                 raw_data_path,
                 filtered_data_path="slice.pkl",
                 stride=100,
                 y_slice_size=100,
                 network_type="MLP",
                 epochs=20,
                 evaluate_training=False,
                 session_filter=Filter(func=hann, search_radius=SEARCH_RADIUS, step_size=WIN_SIZE),
                 time_shift_steps=1,
                 shuffle_data=True,
                 shuffle_factor=500,
                 time_shift_iter=200,
                 initial_timeshift=0,
                 metric_iter=1,
                 batch_size=50,
                 slice_size=1000,
                 x_max=240,
                 y_max=190,
                 x_min=0,
                 y_min=100,
                 x_step=3,
                 y_step=3,
                 win_size=WIN_SIZE,
                 early_stopping=False,
                 search_radius=SEARCH_RADIUS,
                 naive_test=False,
                 valid_ratio=0.1,
                 testing_ratio=0.1,
                 k_cross_validation=1,
                 load_model=False,
                 train_model=True,
                 keep_neuron=-1,
                 neurons_kept_factor=1.0,
                 lw_classifications=None,
                 lw_normalize=False,
                 lw_differentiate_false_licks=True,
                 num_wells=5,
                 metric="map",
                 licks=None,
                 valid_licks=None,
                 filter_tetrodes=None,
                 phase_change_ids=None):
        self.evaluate_training = evaluate_training
        self.stride = stride
        self.train_model = train_model
        self.y_slice_size = y_slice_size
        self.network_type = network_type
        self.epochs = epochs
        self.session_filter = session_filter
        self.time_shift_steps = time_shift_steps
        self.shuffle_data = shuffle_data
        self.shuffle_factor = shuffle_factor
        self.time_shift_iter = time_shift_iter
        self.model_path = model_path
        self.learning_rate = "placeholder"  # TODO
        self.initial_timeshift = initial_timeshift
        self.metric_iter = metric_iter
        self.batch_size = batch_size
        self.slice_size = slice_size
        self.raw_data_path = raw_data_path
        self.x_max = x_max
        self.y_max = y_max
        self.x_min = x_min
        self.y_min = y_min
        self.x_step = x_step
        self.y_step = y_step
        self.win_size = win_size
        self.search_radius = search_radius
        self.early_stopping = early_stopping
        self.naive_test = naive_test
        self.valid_ratio = valid_ratio
        self.k_cross_validation = k_cross_validation
        self.n_neurons = None
        self.X_train = None
        self.X_test = None
        self.y_test = None
        self.y_train = None
        self.metadata_train = None,
        self.X_valid = None
        self.y_valid = None
        self.metadata_valid = None
        self.X_eval = None
        self.y_eval = None
        self.metadata_eval = None
        self.load_model = load_model
        self.keep_neuron = keep_neuron
        self.metric = metric
        self.neurons_kept_factor = neurons_kept_factor
        self.x_shape = None
        self.y_shape = None
        self.lw_classifications = lw_classifications
        self.lw_normalize = lw_normalize
        self.lw_differentiate_false_licks = lw_differentiate_false_licks
        self.num_wells = num_wells
        self.testing_ratio = testing_ratio
        self.filtered_data_path = filtered_data_path
        self.licks = licks
        self.valid_licks = valid_licks
        self.filter_tetrodes = filter_tetrodes
        self.phase_change_ids = None

    def clear_io(self):
        self.X_train = None
        self.y_train = None
        self.X_test = None
        self.y_test = None
        self.metadata_train = None,
        self.X_valid = None
        self.y_valid = None
        self.metadata_valid = None
        self.X_eval = None
        self.y_eval = None
        self.metadata_eval = None
        self.licks = None
        self.valid_licks = None
        self.filter_tetrodes = None

    def get_all_valid_lick_ids(self, session, start_well=1,shift=1):
        """

        :param session: session object
        :param start_well: dominant well in training phase (usually well 1)
        :return: a list of lick_ids of licks corresponding to filter
        """
        licks = session.licks
        filtered_licks = []
        if shift == 1:
            for i,lick in enumerate(licks[0:-2]):
                well = lick.lickwell
                next_well = licks[i + 1].lickwell
                if well == start_well and next_well != start_well:
                    filtered_licks.append(lick.lick_id)

        else:
            for i, lick in enumerate(licks[1:-1]):
                well = lick.lickwell
                next_well = licks[i - 1].lickwell
                if well == start_well and next_well != start_well:
                    filtered_licks.append(lick.lick_id)

        self.valid_licks = filtered_licks

    def get_all_phase_change_ids(self, session):
        """
        :param session: session object
        :param start_well: dominant well in training phase (usually well 1)
        :param change_is_valid: if True exclude licks without change in training phase, if False licks with change
        :return: a list of lick_ids of licks corresponding to filter
        """

        licks = session.licks
        rewarded_list = [lick.rewarded for lick in licks]
        licks = [lick for i, lick in enumerate(licks) if rewarded_list[i] == 1]

        current_phase_well = None
        filtered_licks = []
        for i, lick in enumerate(licks[0:-2]):
            well = lick.lickwell
            for j in range(1,len(licks[0:-2])): # find next valid well licked
                if i + j>=len(licks):
                    next_well = None
                    break
                if licks[i+j].rewarded == 1:
                    next_well = licks[i + j].lickwell
                    break
            if current_phase_well is None:  # set well that is currently being trained for
                current_phase_well = well
            if current_phase_well is not None and well == 1:  # append lick if it fits valid_filter
                if next_well != current_phase_well:
                    filtered_licks.append(lick.lick_id)
                if next_well != current_phase_well:  # change phase if applicable
                    current_phase_well = next_well

        self.phase_change_ids = filtered_licks


    def assign_training_testing(self, X, y, k):
        if self.k_cross_validation == 1:
            valid_length = int(len(X) * self.valid_ratio)
            self.X_train = X[valid_length:]
            self.y_train = y[valid_length:]
            self.X_valid = X[:valid_length // 2]
            self.y_valid = y[:valid_length // 2]
            self.X_test = X[valid_length // 2:valid_length]
            self.y_test = y[valid_length // 2:valid_length]
        else:
            k_len = int(len(X) // self.k_cross_validation)
            k_slice_test = slice(k_len * k, int(k_len * (k + 0.5)))
            k_slice_valid = slice(int(k_len * (k + 0.5)), k_len * (k + 1))
            not_k_slice_1 = slice(0, k_len * k)
            not_k_slice_2 = slice(k_len * (k + 1), len(X))
            self.X_train = X[not_k_slice_1] + X[not_k_slice_2]
            self.y_train = y[not_k_slice_1] + y[not_k_slice_2]
            self.X_test = X[k_slice_test]
            self.y_test = y[k_slice_test]
            self.X_valid = X[k_slice_valid]
            self.y_valid = y[k_slice_valid]
        if self.keep_neuron != -1:
            for i in range(len(self.X_valid)):
                for j in range(len(self.X_valid[0])):
                    if j != self.keep_neuron:
                        self.X_valid[i][j] = np.zeros(self.X_valid[i][j].shape)

    def assign_training_testing_lickwell(self, X, y, k, excluded_wells=[1], normalize=False):
        """"
        Splits data in to training and testing, supports cross validation
        """
        valid_ratio = self.valid_ratio
        if self.k_cross_validation == 1:
            valid_length = int(len(X) * valid_ratio)
            self.X_train = X[valid_length:]
            self.y_train = y[valid_length:]
            self.X_valid = X[:valid_length // 2]
            self.y_valid = y[:valid_length // 2]
            self.X_test = X[valid_length // 2:valid_length]
            self.y_test = y[valid_length // 2:valid_length]
        else:
            k_len = int(len(X) * valid_ratio)
            k_slice_test = slice(k_len * k, int(k_len * (k + 0.5)))
            k_slice_valid = slice(int(k_len * (k + 0.5)), k_len * (k + 1))
            not_k_slice_1 = slice(0, k_len * k)
            not_k_slice_2 = slice(k_len * (k + 1), len(X))
            self.X_train = X[not_k_slice_1] + X[not_k_slice_2]
            self.y_train = y[not_k_slice_1] + y[not_k_slice_2]
            self.X_test = X[k_slice_test]
            self.y_test = y[k_slice_test]
            self.X_valid = X[k_slice_valid]
            self.y_valid = y[k_slice_valid]
            if self.early_stopping is False:
                self.X_valid = self.X_valid + self.X_test
                self.y_valid = self.y_valid + self.y_test

            # print(np.sum(self.y_train, axis=0))
            # print(np.sum(self.y_valid, axis=0))

            if normalize is True:
                self.X_train, self.y_train = self.normalize_discrete(self.X_train, self.y_train,
                                                                     excluded_wells=excluded_wells)
        if self.keep_neuron != -1:
            for i in range(len(self.X_valid)):
                for j in range(len(self.X_valid[0])):
                    if j != self.keep_neuron:
                        self.X_valid[i][j] = np.zeros(self.X_valid[i][j].shape)
        # print(np.sum(self.y_train, axis=0))
        # print(np.sum(self.y_valid, axis=0))
        # print("fin")

    def normalize_discrete(self, x, y, excluded_wells=[]):
        """
        :param x:
        :param y:
        :param nd:
        :param excluded_wells: list of wells which aren't included in the normalization. Useful ie if well 1 licks are over/underrepresented
        :return:
        """
        seed(1)
        """ artificially increases the amount of underrepresented samples. Note that the samples are shuffled, so overlaps will also be mixed in randomly"""
        x_return = x.copy()
        y_return = y.copy()
        x_new = x.copy()
        y_new = y.copy()  # [y[i] for i in range(0, len(y), lick_batch_size)]
        counts = fill_counter(self.num_wells, excluded_wells, y)  # total number of licks by well
        while len(y_new) > 0: # if endless loop occurs here, check if number of wells in net_data object is correct
            i = randint(0, len(y_new) - 1)
            well = normalize_well(well=y_new[i].target,num_wells=self.num_wells,excluded_wells=excluded_wells)
            if counts[well] < max(counts):  # if count at well position smaller than max
                x_return.append(x_new[i])
                y_return.append(y_new[i])
                counts[well] += 1
            else:
                y_new.pop(i)
                x_new.pop(i)
        asd = self.filter_overrepresentation_discrete(x_return, y_return, min(counts), excluded_wells)
        return asd

    def filter_overrepresentation_discrete(self, x, y, max_occurrences, excluded_wells):
        x_return = []
        y_return = []
        y = np.array(y)
        counts = generate_counter(self.num_wells, excluded_wells)
        for i, e in enumerate(y):
            y_pos = normalize_well(e.target, self.num_wells, excluded_wells)
            if counts[y_pos] < max_occurrences:
                x_return.append(x[i])
                counts[y_pos] += 1
                y_return.append(e)
        return x_return, y_return


class Slice:
    def __init__(self, spikes, licks, position_x, position_y, speed, trial_timestamp):
        self.spikes = spikes
        self.n_neurons = len(self.spikes)
        self.licks = licks
        if len(position_x) != len(position_y):
            raise ValueError("position_x and position_y must have the same length")
        self.position_x = position_x
        self.position_y = position_y
        self.speed = speed
        self.trial_timestamp = trial_timestamp
        self.end_time = self.position_x.shape[0]
        self.absolute_time_offset = 0

    def __str__(self):
        return str(self.__dict__)

    def __eq__(self, other):
        return self.spikes == other.spikes and self.licks == other.licks \
               and np.array_equal(self.position_x, other.position_x) \
               and np.array_equal(self.position_y, other.position_y) \
               and self.trial_timestamp == other.trial_timestamp

    def set_filter(self, filter):
        self._filter = filter
        self._convolve()

    def _convolve(self):
        search_radius, step_size = self._filter.search_radius, self._filter.step_size
        if step_size > 2 * search_radius:
            raise ValueError("step_size must be inferior to search_radius")
        n_bin_points = int(len(self.position_x) // step_size)
        self.filtered_spikes = np.zeros((len(self.spikes), n_bin_points))
        with ThreadPool(processes=1) as p:
            ret = p.starmap(_convolve_thread_func,
                            zip(repeat(self._filter), repeat(n_bin_points), range(0, len(self.spikes)),
                                repeat(len(self.spikes)), self.spikes))
            for i, r in enumerate(ret):
                self.filtered_spikes[i] = r

    def plot_filtered_spikes(self, ax, neuron_px_height=4, normalize_each_neuron=False, margin_px=2):
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

    def print_details(self):
        average_spikes_per_second = np.mean([len(x) for x in self.spikes]) * 1000 / len(self.position_x)
        session_length_in_minutes = len(self.position_x) / (1000 * 60)
        average_speed = np.average(self.speed)
        maximum_speed = np.max(self.speed)
        print("Average spikes per second:", average_spikes_per_second)
        print("Session length in minutes:", session_length_in_minutes)
        print("Average speed in cm/s:", average_speed)
        print("Maximum speed in cm/s:", maximum_speed)

    def timeshift_position(self, timeshift):
        if timeshift == 0: return self
        if timeshift > 0:
            other = self[:-timeshift]
            other.position_x = self.position_x[timeshift:]
            other.position_y = self.position_y[timeshift:]
        else:
            other = self[-timeshift:]
            other.position_x = self.position_x[:timeshift]
            other.position_y = self.position_y[:timeshift]

        return other

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

    def slice_spikes(self, time_slice):
        new_spikes = []
        start = 0 if time_slice.start is None else time_slice.start
        stop = self.end_time if time_slice.stop is None else time_slice.stop
        for spike in self.spikes:
            until_stop = takewhile(lambda x: x < stop, spike)
            from_start = dropwhile(lambda x: x < start, until_stop)
            new_spikes.append([x - start for x in list(from_start)])
        return new_spikes

    def slice_list_of_licks(self, licks, time_slice):
        if licks == []:
            return []
        start = 0 if time_slice.start is None else time_slice.start
        if time_slice.stop is None:
            stop = self.end_time
        else:
            if time_slice.stop < 0:
                stop = len(self.position_x) + time_slice.stop
            else:
                stop = time_slice.stop
        keys_containing_time = [k.time for k in licks]
        ret = []
        for d in licks:
            if start <= d.time < stop:
                ret.append(d)
        return ret

    def slice_list_of_dict(self, ld, time_slice):
        if ld == []:
            return []
        start = 0 if time_slice.start is None else time_slice.start
        if time_slice.stop is None:
            stop = self.end_time
        else:
            if time_slice.stop < 0:
                stop = len(self.position_x) + time_slice.stop
            else:
                stop = time_slice.stop
        keys_containing_time = [k for k in ld[0] if "time" in k]
        ret = []
        for d in ld:
            if start <= d["time"] < stop:
                copy = d.copy()
                for k in keys_containing_time:
                    copy[k] -= start
                ret.append(copy)
        return ret

    def __getitem__(self, time_slice):
        if not isinstance(time_slice, slice):
            raise TypeError("Key must be a slice, got {}".format(type(time_slice)))
        start = 0 if time_slice.start is None else time_slice.start

        if time_slice.stop is None:
            stop = self.end_time
        else:
            if time_slice.stop < 0:
                stop = len(self.position_x) + time_slice.stop
            else:
                stop = time_slice.stop
        spikes = self.slice_spikes(time_slice)
        licks = self.slice_list_of_licks(self.licks, time_slice)
        position_x = self.position_x[time_slice]
        position_y = self.position_y[time_slice]
        speed = self.speed[time_slice]
        trial_timestamp = self.slice_list_of_dict(self.trial_timestamp, time_slice)
        new = Slice(spikes, licks, position_x, position_y, speed, trial_timestamp)
        if hasattr(self, "_filter"):
            new._filter = self._filter
            if hasattr(self, "filtered_spikes"):
                fstart = start / self._filter.step_size
                fstop = (stop / self._filter.step_size)
                frange = fstop - fstart
                fstart = int(fstart)
                fstop = int(fstart + frange)
                new.filtered_spikes = self.filtered_spikes[:, fstart:fstop]

        new.absolute_time_offset = self.absolute_time_offset + start
        return new

    @classmethod
    def from_raw_data(cls, path, filter_tetrodes=None):
        """

        :param path:
        :param filter_tetrodes: list or range of tetrodes to be removed from session
        :return:
        """
        print("start loading session")
        foster_path = glob.glob(path + "/*_fostertask.dat")[0]
        all_channels_path = path + "/all_channels.events"
        spiketracker_path = path + "/probe1/session1/spike_tracker_data.mat"

        # Extract spiketracker data

        spiketracker_data = scipy.io.loadmat(spiketracker_path)
        if filter_tetrodes is not None:  # This value apparently only exists in session 3 but is necessary to distinguish pfc and hc
            tetrode_channel_list = spiketracker_data["waveform"]["tetrode_n"][0][0][0]

        # param: head_s, pixelcm, speed_s, spike_templates, spike_times, waveform,x_s, y_s

        position_x = spiketracker_data["x_s"].squeeze().astype(float)
        position_y = spiketracker_data["y_s"].squeeze().astype(float)
        speed = spiketracker_data["speed_s"].squeeze().astype(float)
        spikes_raw = spiketracker_data["spike_times"]
        spikes_raw = [s[0].tolist() for s in spikes_raw[0]]  # remove useless dims
        spikes = []
        for i, spike in enumerate(spikes_raw):
            # the raw data contains spikes outside the session scope
            spikes.append(spikes_raw[i][:bisect.bisect_left(spikes_raw[i], len(position_x))])
            if filter_tetrodes is not None and tetrode_channel_list[i] in filter_tetrodes:
                spikes.pop(-1)
        # load foster data

        with open(foster_path, "r") as f:
            # contains tripels with data about 0: rewarded, 1: lickwell, 2: duration
            foster_data = np.fromfile(f, dtype=np.uint16)
        foster_data = np.reshape(foster_data, [foster_data.size // 3, 3])  # 0: rewarded, 1: lickwell, 2: duration
        initial_detection = [x == 2 for x in foster_data[:, 0]]

        # load data from all_channels.event file with ephys

        # test_dict = OpenEphys.load(path + "/TT0.spikes")

        all_channels_dict = OpenEphys.load(all_channels_path)
        # param: eventType, sampleNum, header, timestamps, recordingNumber, eventId, nodeId, channel
        timestamps = all_channels_dict["timestamps"]
        eventch_ttl = all_channels_dict["channel"]  # channel number
        eventtype_ttl = all_channels_dict["eventType"]  # 3 for TTL events
        eventid_ttl = all_channels_dict["eventId"]  # 1: for on, 0 for off
        recording_num = all_channels_dict["recordingNumber"]
        sample_rate = float(all_channels_dict["header"]["sampleRate"])
        # timestamp for foster data
        foster_timestamp = [(x - timestamps[0]) / sample_rate * 1000 for ind, x in enumerate(timestamps)
                            if (eventtype_ttl[ind], eventch_ttl[ind], eventid_ttl[ind], recording_num[ind]) == (
                                3, 2, 1, 0)]

        if len(initial_detection) > len(foster_timestamp):
            foster_data = foster_data[:, 0:foster_timestamp.size]
            initial_detection = initial_detection[1:foster_timestamp.size]
        foster_data = [x for ind, x in enumerate(foster_data) if initial_detection[ind] != 1]
        detected_events = [ind for ind, x in enumerate(initial_detection) if
                           x == False]  # index positions of non detection events
        initial_detection_timestamp = [foster_timestamp[ind - 1] for ind in
                                       detected_events]  # excludes initial event
        initial_detection_timestamp = initial_detection_timestamp
        rewarded = [item[0] for item in foster_data]
        lickwells = [item[2] for item in foster_data]
        # trial timestamp
        trial_timestamp = [{"time": float(initial_detection_timestamp[ind]),
                            "trial_lickwell": int(well),
                            "trial_id": ind}
                           for ind, well in enumerate(lickwells) if rewarded[ind] == 1]
        # licks

        licks = [Lick(time=float(initial_detection_timestamp[i]), lickwell=int(lickwells[i]), rewarded=int(rewarded[i]),
                      lick_id=i)
                 for i in
                 range(1, len(initial_detection_timestamp)) if rewarded[i]==0 or rewarded[i] == 1]  # Please note that first lick is deleted by default TODO
        print("finished loading session")
        return cls(spikes, licks, position_x, position_y, speed, trial_timestamp)

    def filter_neurons_randomly(self, factor):
        neurons_removed = int(len(self.spikes) * (1 - factor))
        for i in range(neurons_removed):
            neuron_index = np.random.randint(0, len(self.spikes))
            del self.spikes[neuron_index]
            self.filtered_spikes = np.delete(self.filtered_spikes, neuron_index, axis=0)

        self.n_neurons = len(self.spikes)

    def filter_neurons(self, minimum_spikes):

        """
        removes neurons from spikes that contain less than a given number of spikes
        :param spikes: list of spikes from raw data
        :param minimum_spikes: minimum spikes
        :return: new spikes list
        """
        for i in range(len(self.spikes) - 1, -1, -1):
            if len(self.spikes[i]) < minimum_spikes:
                del self.spikes[i]
        self.n_neurons = len(self.spikes)

    @classmethod
    def from_pickle(cls, path):
        with open(path, 'rb') as f:
            data = pickle.load(f)
            return data

    def to_pickle(self, path):
        with open(path, 'wb') as f:
            pickle.dump(self, f)
