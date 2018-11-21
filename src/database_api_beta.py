from multiprocessing.pool import ThreadPool
import numpy as np
from itertools import takewhile, dropwhile, repeat
from src import OpenEphys
import scipy
import bisect
import glob
import pickle
from random import seed,randint
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


class Net_data:

    WIN_SIZE = 20
    SEARCH_RADIUS = WIN_SIZE * 2

    def __init__(self,
                 MODEL_PATH,
                 RAW_DATA_PATH,
                 MAKE_HISTOGRAM=False,
                 STRIDE = 100,
                 Y_SLICE_SIZE = 200,
                 network_type="MLP",
                 EPOCHS=20,
                 session_filter=Filter(func=hann, search_radius=SEARCH_RADIUS, step_size=WIN_SIZE),
                 TIME_SHIFT_STEPS = 1,
                 SHUFFLE_DATA = True,
                 SHUFFLE_FACTOR = 500,
                 TIME_SHIFT_ITER = 200,
                 r2_scores_train = [],
                 r2_scores_valid = [],
                 acc_scores_train = [],
                 acc_scores_valid = [],
                 avg_scores_train = [],
                 avg_scores_valid = [],
                 INITIAL_TIMESHIFT = 0,
                 METRIC_ITER = 1,
                 BATCH_SIZE = 50,
                 SLICE_SIZE=1000,
                 X_MAX = 240,
                 Y_MAX = 190,
                 X_MIN = 0,
                 Y_MIN = 100,
                 X_STEP = 3,
                 Y_STEP = 3,
                 WIN_SIZE=WIN_SIZE,
                 EARLY_STOPPING = False,
                 SEARCH_RADIUS=SEARCH_RADIUS,
                 NAIVE_TEST = False,
                 VALID_RATIO = 0.1,
                 K_CROSS_VALIDATION = 1,
                 LOAD_MODEL = False,
                 TRAIN_MODEL = True,
                 keep_neuron = -1,
                 NEURONS_KEPT_FACTOR = 1.0,
                 lw_classifications = None,
                 lw_normalize = False,
                 lw_differentiate_false_licks = True,
                 num_wells = 5,
                 metric="map"):
        self.MAKE_HISTOGRAM = MAKE_HISTOGRAM
        self.STRIDE = STRIDE
        self.TRAIN_MODEL = TRAIN_MODEL
        self.Y_SLICE_SIZE = Y_SLICE_SIZE
        self.network_type = network_type
        self.EPOCHS = EPOCHS
        self.session_filter = session_filter
        self.TIME_SHIFT_STEPS = TIME_SHIFT_STEPS
        self.SHUFFLE_DATA = SHUFFLE_DATA
        self.SHUFFLE_FACTOR = SHUFFLE_FACTOR
        self.TIME_SHIFT_ITER = TIME_SHIFT_ITER
        self.MODEL_PATH = MODEL_PATH
        self.learning_rate = "placeholder"  # TODO
        self.r2_scores_train = r2_scores_train
        self.r2_scores_valid = r2_scores_valid
        self.acc_scores_train = acc_scores_train
        self.acc_scores_valid = acc_scores_valid
        self.avg_scores_train = avg_scores_train
        self.avg_scores_valid = avg_scores_valid
        self.INITIAL_TIMESHIFT = INITIAL_TIMESHIFT
        self.METRIC_ITER = METRIC_ITER
        self.BATCH_SIZE = BATCH_SIZE
        self.SLICE_SIZE = SLICE_SIZE
        self.RAW_DATA_PATH = RAW_DATA_PATH
        self.X_MAX = X_MAX
        self.Y_MAX = Y_MAX
        self.X_MIN = X_MIN
        self.Y_MIN = Y_MIN
        self.X_STEP = X_STEP
        self.Y_STEP = Y_STEP
        self.WIN_SIZE = WIN_SIZE
        self.SEARCH_RADIUS = SEARCH_RADIUS
        self.EARLY_STOPPING = EARLY_STOPPING
        self.NAIVE_TEST = NAIVE_TEST
        self.VALID_RATIO = VALID_RATIO
        self.K_CROSS_VALIDATION = K_CROSS_VALIDATION
        self.N_NEURONS = None
        self.X_train = None
        self.y_train = None
        self.X_valid = None
        self.y_valid = None
        self.X_eval = None
        self.y_eval = None
        self.LOAD_MODEL = LOAD_MODEL
        self.keep_neuron = keep_neuron
        self.metric = metric
        self.NEURONS_KEPT_FACTOR = NEURONS_KEPT_FACTOR
        self.x_shape = None
        self.y_shape = None
        self.lw_classifications = lw_classifications
        self.lw_normalize = lw_normalize
        self.lw_differentiate_false_licks = lw_differentiate_false_licks
        self.num_wells = num_wells

    def split_data(self, X, y,k,normalize = False):
        """"
        Splits data in to training and testing, supports cross validation
        """
        valid_ratio = self.VALID_RATIO
        if self.K_CROSS_VALIDATION == 1:
            valid_length = int(len(X) * valid_ratio)
            self.X_train = X[valid_length:]
            self.y_train = y[valid_length:]
            self.X_valid = X[:valid_length // 2]
            self.y_valid = y[:valid_length // 2]
            self.X_test = X[valid_length // 2:valid_length]
            self.y_test = y[valid_length // 2:valid_length]
        else:
            k_len = int(len(X) // valid_ratio)
            if normalize is True:
                counts = np.sum(y, axis=0)
                counts = counts[1:] # TODO remove counts of well 1
                k_len = int(self.VALID_RATIO * self.num_wells * min(counts)) # excludes area of samples which is not evenly spread over well types
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
            if normalize is True:

                self.X_train, self.y_train = self.normalize_discrete(self.X_train, self.y_train,exclude_wells=[1])
        if self.keep_neuron != -1:
            for i in range(len(self.X_valid)):
                for j in range(len(self.X_valid[0])):
                    if j != self.keep_neuron:
                        self.X_valid[i][j] = np.zeros(self.X_valid[i][j].shape)

    def normalize_discrete(self,x,y,exclude_wells=[]):
        """

        :param x:
        :param y:
        :param nd:
        :param exclude_wells: list of wells which aren't included in the normalization. Useful ie if well 1 licks are over/underrepresented
        :return:
        """
        seed(1)
        """ artificially increases the amount of underrepresented samples. Note that the samples are shuffled, so overlaps will also be mixed in randomly"""
        x_return = x.copy()
        y_return = y.copy()
        x_new = x.copy()
        y_new = y.copy() # [y[i] for i in range(0, len(y), lick_batch_size)]
        counts = np.sum(y_return,axis=0) # total number of licks by well
        while len(y_new) > 0:
            i = randint(0, len(y_new) - 1)
            if max(counts*y_new[i]) < max(counts): # if count at well position smaller than max
                x_return.append(x_new[i])
                y_return.append(y_new[i])
                counts[np.argmax(y_new[i])] += 1
            else:
                y_new.pop(i)
                x_new.pop(i)
        for i in exclude_wells:
            counts = np.delete(counts,i-1)
        return self.filter_overrepresentation_discrete(x_return, y_return, min(counts))


    def filter_overrepresentation_discrete(self, x, y, max_occurrences):
        x_return = []
        y_return = []
        y = np.array(y)
        pos_counter = np.zeros(y[0].size)
        for i, e in enumerate(y):
            y_pos = np.argmax(e)
            if pos_counter[y_pos] < max_occurrences:
                x_return.append(x[i])
                pos_counter[y_pos] += 1
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
        licks = self.slice_list_of_dict(self.licks, time_slice)
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
    def from_raw_data(cls, path):
        print("start loading session")
        foster_path = glob.glob(path + "/*_fostertask.dat")[0]
        all_channels_path = path + "/all_channels.events"
        spiketracker_path = path + "/probe1/session1/spike_tracker_data.mat"

        # Extract spiketracker data

        spiketracker_data = scipy.io.loadmat(spiketracker_path)

        # param: head_s, pixelcm, speed_s, spike_templates, spike_times, waveform,x_s, y_s

        position_x = spiketracker_data["x_s"].squeeze().astype(float)
        position_y = spiketracker_data["y_s"].squeeze().astype(float)
        speed = spiketracker_data["speed_s"].squeeze().astype(float)
        spikes = spiketracker_data["spike_times"]
        spikes = [s[0].tolist() for s in spikes[0]]  # remove useless dims

        for i in range(len(spikes)):
            # the raw data contains spikes outside the session scope
            spikes[i] = spikes[i][:bisect.bisect_left(spikes[i], len(position_x))]

        # load foster data

        with open(foster_path, "r") as f:
            # contains tripels with data about 0: rewarded, 1: lickwell, 2: duration
            foster_data = np.fromfile(f, dtype=np.uint16)
        foster_data = np.reshape(foster_data, [foster_data.size // 3, 3])  # 0: rewarded, 1: lickwell, 2: duration
        initial_detection = [x == 2 for x in foster_data[:, 0]]

        # load data from all_channels.event file with ephys

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
        licks = [{"time": float(initial_detection_timestamp[i]),
                  "lickwell": int(lickwells[i]),
                  "rewarded": int(rewarded[i])}
                 for i in
                 range(1, len(initial_detection_timestamp))]  # Please note that first lick is deleted by default TODO
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
