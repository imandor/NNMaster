from multiprocessing.pool import ThreadPool
from itertools import takewhile, dropwhile, repeat
from src import OpenEphys
import scipy
import bisect
import glob
import pickle
from random import seed, randint
from src.preprocessing import generate_counter, fill_counter, normalize_well, shuffle_list_key
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
from numpy.random import multivariate_normal
import matplotlib.colors as colors
import random

well_to_color = {0: "#ff0000", 1: "#669900", 2: "#0066cc", 3: "#cc33ff", 4: "#003300", 5: "#996633"}


def hann(x):
    """
    :param x:
    :return: hann filter of input
    """
    if np.abs(x) < 1:
        return (1 + np.cos(x * np.pi)) / 2
    else:
        return 0


def bin(x):
    """

    :param x:
    :return: bin filter of input
    """
    if np.abs(x) < 1:
        return 1
    else:
        return 0


class Filter:
    # object for convolving input data
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


def _convolve_thread_func(filter_func, n_bin_points, neuron_spikes):
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
        filtered_spikes[index] = sum(map(
            lambda x: filter_func((x - index * filter_func.step_size) / filter_func.search_radius),
            curr_spikes_in_search_window))
        for spike_index, spike in enumerate(neuron_spikes[
                                            index_first_spike_in_window:index_first_spike_in_window + curr_search_window_max_bound]):
            # upper bound because a maximum of 1 spike per ms can occur and runtime of slice operation is O(i2-i1)
            if spike >= curr_search_window_min_bound:
                index_first_spike_in_window = index_first_spike_in_window + spike_index
                break
    return filtered_spikes


class Lick():
    # object describes a lick-event from session data
    def __init__(self, lickwell, time, rewarded, lick_id, target=None, next_phase=None, last_phase=None, phase=None):
        self.time = time
        self.lickwell = lickwell
        self.rewarded = rewarded
        self.lick_id = lick_id
        self.target = target
        self.next_phase = next_phase
        self.last_phase = last_phase
        self.phase = phase


class Evaluated_Lick(Lick):
    # contains additional metrics regarding each lick event after the network finished training. Should only appear
    # inside output files (but not inside slice objects)
    def __init__(self, lickwell, time, rewarded, lick_id, phase, next_phase, last_phase, prediction=None,
                 next_lick_id=None, last_lick_id=None,
                 fraction_decoded=None, total_decoded=None, target=None, fraction_predicted=None, ):
        Lick.__init__(self, lickwell, time, rewarded, lick_id, target, phase=phase, next_phase=next_phase,
                      last_phase=last_phase)
        self.next_lick_id = next_lick_id
        self.last_lick_id = last_lick_id
        self.fraction_decoded = fraction_decoded
        self.total_decoded = total_decoded
        self.prediction = prediction
        self.fraction_predicted = fraction_predicted

class Net_data:
    # contains all data necessary for training the network. Unlike session data, this data is mutable during runtime
    # and contains information not directly related to raw session data. Kept separate, since session data is usually very large
    # and usually only needs to be created once while the neural network specifics often change depending on the study
    WIN_SIZE = 100
    SEARCH_RADIUS = WIN_SIZE * 2

    def __init__(self,
                 model_path,  # where trained network is saved to
                 raw_data_path,  # where session data set(for Slice object) is saved
                 dropout,  # where slice object for session is saved
                 filtered_data_path=None,
                 stride=100,
                 # stride [ms]with which output samples are generated (making it smaller than y_slice_size creates overlapping samples)
                 y_slice_size=100,  # size of each output sample in [ms]
                 network_type="MLP",
                 # originally multiple network types were tested. This is a descriptor for easier recognition of network type in file data
                 epochs=20,  # how many epochs the network is trained
                 network_shape=10,  # currently not used, originally showed how many samples are in each batch
                 from_raw_data=True,
                 # if True, loads session from raw data path, if False loads session from filtered data path (which is much faster)
                 evaluate_training=False,
                 # if set to True, the network evaluates training performance at runtime (slows down training)
                 session_filter=Filter(func=hann, search_radius=SEARCH_RADIUS, step_size=WIN_SIZE),
                 # filter function with which to convolve raw data
                 time_shift_steps=1,
                 # after each step, the network resets with new timeshift and reassigns new ys depending on the current timeshift. Determines how many steps are performed at runtime (see time_shift_iter)
                 shuffle_data=True,
                 # if data is to be shuffled before assigning to training and testing (see shuffle-factor parameter)
                 shuffle_factor=500,
                 # shuffling occurs batch-wise (to minimize overlap if stride is lower than y_slice_size). Indicates how many samples are to be shuffled in a batch.
                 time_shift_iter=500,  # determines, how much the time is shifted after each time_shift_step
                 initial_timeshift=0,
                 # initial time_shift for position decoding. For well-decoding, I appropiated this parameter to determine if the previous or next well is to be decoded (sorry). +1 indicates next well, -1 previous well
                 metric_iter=1,  # after how many epochs the network performance is to be evaluated
                 batch_size=50,  # how many samples are to be given into the network at once (google batch_size)
                 slice_size=1000,  # how many [ms] of neural spike data are in each sample
                 x_max=240,  # determines shape of the track the rat is located on [cm]
                 y_max=190,  # determines shape of the track the rat is located on [cm]
                 x_min=0,  # determines shape of the track the rat is located on [cm]
                 y_min=100,  # determines shape of the track the rat is located on [cm]
                 x_step=3,  # determines, what shape the position bins for the samples are [cm] (3*3cm per bin default)
                 y_step=3,  # determines, what shape the position bins for the samples are [cm] (3*3cm per bin default)
                 win_size=WIN_SIZE,  # size of convolution window for filter function
                 early_stopping=False,
                 # if True, checks if network performance degrades after a certain amount of steps (see network) and stops training early if yes
                 search_radius=SEARCH_RADIUS,  # size of search radius for filter function
                 naive_test=False,
                 # if True, doesn't reassign y_values after each time_shift step. Necessary to determine what part of the network performance is due to similarities between different time_shift step neural data
                 valid_ratio=0.1,  # ratio between training and validation data
                 testing_ratio=0.1,  # ration between training and testing data
                 k_cross_validation=1,
                 # if more than 1, the data is split into k different sets and results are averaged over all set-performances
                 load_model=False,
                 # if True, the saved model is loaded for training instead of a new one. Is set to True if naive testing is True and time-shift is != 0
                 train_model=True,
                 # if True, the model is not trained during runtime. Is set to True if naive testing is True and time-shift is != 0
                 keep_neuron=-1,  # TODO not sure what this was supposed to do
                 neurons_kept_factor=1.0,
                 # if less than one, a corresponding fraction of neurons are randomly removed from session before training
                 lw_classifications=None,  # for well decoding: how many classes exist
                 lw_normalize=False,
                 # lickwell_data: if True, training and validation data is normalized (complex rules)
                 lw_differentiate_false_licks=False,
                 # Not used anymore due to too small sample sizes and should currently give an error if True. if True, the network specifically trains to distinguish between correct and false licks. Should work with minimal code editing if ever necessary to implement
                 num_wells=5,
                 # number of lickwells in data set. Really only in object because of lw_differentiate_false_licks, but there is no reason to remove it either
                 metric="map",  # with what metric the network is to be evaluated (depending on study)
                 valid_licks=None,  # List of licks which are valid for well-decoding study
                 filter_tetrodes=None,
                 # removes tetrodes from raw session data before creating slice object. Useful if some of the tetrodes are e.g. hippocampal and others for pfc
                 phases=None,  # contains list of training phases
                 phase_change_ids=None,  # contains list of phase change lick_ids
                 number_of_bins = 10, # number of win sized bins going into the input
                 start_time_by_lick_id = None, # list of tuples (lick_id,start time) which describe the time at which a lick "officially starts relative to the lick time described in the lick object. Defaults to zero but can be changed if a different range is to be observed
                 behavior_component_filter = None # filters session data, string
                 ):
        self.dropout = dropout
        self.session_from_raw = from_raw_data
        self.network_shape = network_shape
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
        self.learning_rate = "placeholder"  # can be added if seen as necessary
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
        self.valid_licks = valid_licks
        self.filter_tetrodes = filter_tetrodes
        self.phase_change_ids = phase_change_ids
        self.phases = phases
        self.number_of_bins = number_of_bins
        self.start_time_by_lick_id = start_time_by_lick_id
        self.behavior_component_filter = behavior_component_filter
    def clear_io(self):
        self.network_shape = None
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
        self.valid_licks = None
        self.filter_tetrodes = None

    def get_all_valid_lick_ids(self, session, lickstart, lickstop, start_well=1, shift=1):
        """
        :param session: session object
        :param start_well: dominant well in training phase (usually well 1)
        :return: a list of lick_ids of licks corresponding to filter
        """
        filtered_licks = []
        if shift == 1:
            licks = session.licks
            for i, lick in enumerate(licks[0:-1]):
                well = lick.lickwell
                next_well = licks[i + 1].lickwell
                # print(well,next_well,end="")
                if well == start_well and next_well != start_well:
                    filtered_licks.append(lick.lick_id)
                    # print("ding!")
                # else:
                # print("")
        else:
            licks = session.licks[1:-1]
            for i, lick in enumerate(licks):
                well = lick.lickwell
                next_well = licks[i - 1].lickwell
                if well == start_well and next_well != start_well:
                    filtered_licks.append(lick.lick_id)
        # TODO additional filter function to exclude instances when rat moves around too much
        # new_filtered_licks = []
        # for lick_id in filtered_licks:
        #     lick = get_lick_from_id(lick_id,session.licks)
        #     time_start = lick.time + lickstart
        #     time_stop = lick.time + lickstop
        #     slice = session[int(time_start):int(time_stop)]
        #     min = slice.position_x[0]
        #     max = slice.position_x[lickstop-lickstart-1]
        #     # std_lower.append(np.min(slice.position_x))
        #     # std_upper.append(np.max(slice.position_x))
        #     print(lick.lick_id,min,max)
            # if max-min < 30:
            #     new_filtered_licks.append(lick.lick_id)
        self.valid_licks = filtered_licks

    def assign_training_testing(self, X, y, k):
        """

        :param X: network input
        :param y: network output
        :param k: cross validation factor
        :return: None. Assigns X_train, y_train, x_valid, y_valid, X_test_, y_test for self
        """
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

        if self.early_stopping is False:
            self.X_valid = self.X_valid + self.X_test
            self.y_valid = self.y_valid + self.y_test
            self.X_test = []
            self.y_test = []

        if self.keep_neuron != -1:
            for i in range(len(self.X_valid)):
                for j in range(len(self.X_valid[0])):
                    if j != self.keep_neuron:
                        self.X_valid[i][j] = np.zeros(self.X_valid[i][j].shape)

    def assign_training_testing_lickwell(self, X, y, k, excluded_wells=[1], normalize=False):
        """
        :param X: network input
        :param y: network output
        :param k: cross validation factor
        :param excluded_wells: excludes target wells from being considered when assigning
        :param normalize: if True, normalizes data
        :return: Splits data in to training and testing, supports cross validation
        """
        """"
        """
        valid_ratio = self.valid_ratio

        k_len = int(len(X) * valid_ratio)
        valid_slice = slice(k_len * k, k_len * (k + 1))
        train_slice_1 = slice(0, k_len * k)
        train_slice_2 = slice(k_len * (k + 1), len(X))
        X_train = X[train_slice_1] + X[train_slice_2]
        y_train = y[train_slice_1] + y[train_slice_2]
        X_valid = X[valid_slice]
        y_valid = y[valid_slice]
        if k == int(1/valid_ratio)-1: # last index adds rest of validation slice
            X_valid = X_valid + X[valid_slice.stop:len(X)]
            y_valid = y_valid + y[valid_slice.stop:len(X)]
            X_train = X[train_slice_1]
            y_train = y[train_slice_1]

        self.X_train,self.y_train = process_samples(self,X_train,y_train)
        self.X_valid,self.y_valid = process_samples(self,X_valid,y_valid)

        if normalize is True:
            self.X_train, self.y_train = self.normalize_discrete(self.X_train, self.y_train,
                                                                 excluded_wells=excluded_wells)
        if self.keep_neuron != -1:
            for i in range(len(self.X_valid)):
                for j in range(len(self.X_valid[0])):
                    if j != self.keep_neuron:
                        self.X_valid[i][j] = np.zeros(self.X_valid[i][j].shape)

    def normalize_discrete(self, x, y, excluded_wells=[]):
        """
        :param x: network input
        :param y: network output
        :param excluded_wells: list of wells which aren't included in the normalization. Useful ie if well 2 licks are extremely over/underrepresented
        :return: artificially increases the amount of underrepresented samples and removes overrepresented samples afterwards.
        Note that the samples are shuffled, so overlaps will also be mixed in randomly
        """
        seed(0)
        counts = fill_counter(self.num_wells, excluded_wells, y)  # total number of licks by well
        # shuffle values so they are picked randomly
        r = shuffle_list_key(length=len(y), shuffle_batch_size=1, seed_no=1)
        x = [x[j] for j in r]
        y = [y[j] for j in r]
        x_return = x.copy()
        y_return = y.copy()

        for i, e in enumerate(y):
            well_index = normalize_well(well=e.target, num_wells=self.num_wells, excluded_wells=excluded_wells)
            if counts[well_index] < max(counts):  # if count at well position smaller than max
                x_return.append(x[i])
                y_return.append(y[i])
                counts[well_index] += 1
        return x_return, y_return

    def plot_validation_position_histogram(self, k):
        """
        creates and saves a 2d histogram which shows the positions in validation data set
        """
        fig, ax = plt.subplots()
        pos_sum = np.sum(self.y_valid, axis=0)
        x = []
        y = []
        w = []
        for i, row in enumerate(pos_sum):
            for j, value in enumerate(row):
                if value != 0:
                    w.append(value)
                    x.append(i)
                    y.append(j)
        ax.set_title('Histogram of coordinates in validation set')
        ax.hist2d(x, y, weights=w, norm=colors.LogNorm(vmin=1, vmax=5000), bins=50, cmap="binary")
        ax.set_xlabel("x [cm]")
        ax.set_ylabel("y [cm]")
        fig.tight_layout()
        plt.savefig("C:/Users/NN/Desktop/Master/experiments/Histogram of positions/_" + str(k))


def process_samples(nd,slice_list,licks):
    """ generates samples from list of slices"""
    y = []
    X = []
    for i, lick in enumerate(licks):
        if nd.start_time_by_lick_id is not None:
            offset = [i[1] for i in nd.start_time_by_lick_id if i[0]==lick.lick_id][0]
        else:
            offset = 0
        slice = slice_list[i]
        lick_start = slice.absolute_time_offset + offset
        lick_stop = lick_start + len(slice.position_x)
        for j in range(0, (lick_stop - lick_start) // nd.win_size - nd.number_of_bins):
            bins_to_x = [c[j:j + nd.number_of_bins] for c in slice.filtered_spikes]
            bins_to_x = np.reshape(bins_to_x, [len(bins_to_x), len(bins_to_x[0])])
            X.append(bins_to_x)
            y.append(lick)
    return X,y


class Slice:
    """
    Represents session object. Should not be changed at runtime aside from if convolving spike data. Should be saved to file to save execution time and can be loaded from file with setting from_raw_data command in Net_data object
    """

    def __init__(self, spikes, licks, position_x, position_y, speed, trial_timestamp):
        self.spikes = spikes  # contains neural spike data in [ms]
        self.n_neurons = len(self.spikes)  # number of neurons
        self.licks = licks  # list of lick objects signifying lick-events
        if len(position_x) != len(position_y):
            raise ValueError("position_x and position_y must have the same length")
        self.position_x = position_x  # x axis position of the rat in [ms]
        self.position_y = position_y  # y axis position of the rat in [ms]
        self.speed = speed  # list of speeds of the rat in [cm/s] at each [ms] of the data set
        self.trial_timestamp = trial_timestamp  # list of times at which trials occurred
        self.end_time = self.position_x.shape[0]  # timestamp of last event
        self.absolute_time_offset = 0  # offset of Slice relative to beginning of recording in [ms]

    def __str__(self):
        return str(self.__dict__)

    def __eq__(self, other):
        return self.spikes == other.spikes and self.licks == other.licks \
               and np.array_equal(self.position_x, other.position_x) \
               and np.array_equal(self.position_y, other.position_y) \
               and self.trial_timestamp == other.trial_timestamp

    def set_filter(self, filter):
        """

        :param filter: convolution filter for spike data
        :return: sets filter
        """
        self._filter = filter
        self._convolve()

    def add_lick_data_to_session_and_net_data(self, nd):
        """

        :param nd: Net-data Object. Necessary to add metadata and is returned altered
        :return: adds metadata about licks to self and a returned Net_data object. Specifically phases, next phase, last phase, phase
        """
        licks = self.licks
        shift = 1
        for i, lick in enumerate(licks):
            if i + shift < len(licks) and i + shift >= 0:
                self.licks[i].target = licks[i + shift].lickwell

        # get ids of phase changes and corresponding phases as respectively filtered_lick_ids and filtered_lickwells

        filtered_lickwells = [lick.target for i, lick in enumerate(self.licks[0:-2]) if
                              licks[i + 1].rewarded == 1 and lick.lickwell == 1]
        filtered_lickwell_ids = [lick.lick_id for i, lick in enumerate(self.licks[0:-2]) if
                                 licks[i + 1].rewarded == 1 and lick.lickwell == 1]
        filtered_lickwell_phasechange_ids = [filtered_lickwell_ids[i] for i, lickwell in
                                             enumerate(filtered_lickwells[0:-2]) if
                                             filtered_lickwells[i] != filtered_lickwells[i + 1]] + [licks[-1].lick_id]
        filtered_lickwells = [lickwell for i, lickwell in enumerate(filtered_lickwells[0:-2]) if
                              filtered_lickwells[i] != filtered_lickwells[i + 1]] + [filtered_lickwells[-1]]
        filtered_lickwells = np.array(filtered_lickwells)

        # set phases for each lick

        current_phase_counter = 0
        for i, lick in enumerate(self.licks):
            lick.phase = filtered_lickwells[current_phase_counter]
            if current_phase_counter != 0:
                lick.last_phase = filtered_lickwells[current_phase_counter - 1]
            if current_phase_counter != len(filtered_lickwells) - 1:
                lick.next_phase = filtered_lickwells[current_phase_counter + 1]
            if lick.lick_id in filtered_lickwell_phasechange_ids:
                current_phase_counter += 1
            self.licks[i] = lick
        # set nd data
        nd.phase_change_ids = filtered_lickwell_phasechange_ids
        nd.phases = filtered_lickwells

        return nd

    def _convolve(self):
        # convolves spike data with the preset filter function
        search_radius, step_size = self._filter.search_radius, self._filter.step_size
        if step_size > 2 * search_radius:
            raise ValueError("step_size must be inferior to search_radius")
        n_bin_points = int(len(self.position_x) // step_size)
        self.filtered_spikes = np.zeros((len(self.spikes), n_bin_points))
        with ThreadPool(processes=1) as p:
            ret = p.starmap(_convolve_thread_func,
                            zip(repeat(self._filter), repeat(n_bin_points), self.spikes))
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

    def plot_positions(self):
        """
        :return: shows a binary 2d histogram of each visited location
        """
        fig, ax = plt.subplots()
        ax.set_title('Normalized histogram of visited locations')
        ax.hist2d(self.position_x, self.position_y, norm=colors.LogNorm(vmin=1, vmax=5000), bins=50, cmap="binary")
        ax.set_xlabel("x [cm]")
        ax.set_ylabel("y [cm]")
        fig.tight_layout()
        plt.show()

    def print_details(self):
        """
        :return: prints session data
        """
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
        """

        :param time_slice:
        :return: help function for slicing session
        """
        new_spikes = []
        start = 0 if time_slice.start is None else time_slice.start
        stop = self.end_time if time_slice.stop is None else time_slice.stop
        for spike in self.spikes:
            until_stop = takewhile(lambda x: x < stop, spike)
            from_start = dropwhile(lambda x: x < start, until_stop)
            new_spikes.append([x - start for x in list(from_start)])
        return new_spikes

    def slice_list_of_licks(self, licks, time_slice):
        """

        :param licks:
        :param time_slice:
        :return: help function for slicing session
        """
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
        """

        :param ld:
        :param time_slice:
        :return: help function for slicing session
        """
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
        """

        :param time_slice: slice object
        :return: returns slice of session object (e.g. session[20:40] returns a slice of the session between 20 and 40 [ms]
        """
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

        :param path: file path
        :param filter_tetrodes: list or range of tetrodes to be kept in session
        :return: loads session from raw data set
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
            if filter_tetrodes is not None and tetrode_channel_list[i] in filter_tetrodes:
                spikes.append(spikes_raw[i][:bisect.bisect_left(spikes_raw[i], len(position_x))])
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
        # for i,lick in enumerate(lickwells):
        #     if lick !=1:
        #         lickwells[i] = random.randint(2,5)
        # trial timestamp
        trial_timestamp = [{"time": float(initial_detection_timestamp[ind]),
                            "trial_lickwell": int(well),
                            "trial_id": ind}
                           for ind, well in enumerate(lickwells) if rewarded[ind] == 1]
        # licks

        licks = [Lick(time=float(initial_detection_timestamp[i]), lickwell=int(lickwells[i]), rewarded=int(rewarded[i]),
                      lick_id=i)
                 for i in
                 range(1, len(initial_detection_timestamp)) if
                 rewarded[i] == 0 or rewarded[i] == 1]  # Please note that first lick is deleted by default TODO

        print("finished loading session")
        return cls(spikes, licks, position_x, position_y, speed, trial_timestamp)

    def filter_neurons_randomly(self, factor):
        seed(0)
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
        """ saves session to file"""
        with open(path, 'rb') as f:
            data = pickle.load(f)
            return data

    def to_pickle(self, path):
        """ loads session from file"""
        with open(path, 'wb') as f:
            pickle.dump(self, f)


def get_lick_from_id(id, licks, shift=0,get_next_best=False,dir=1):
    """

    :param id: id of lick being searched
    :param licks: list of lick objects
    :param shift: if set, gets lick before or after id'd lick
    :param get_next_best: if True, takes the next available lick if requested lick not found
    :param dir: search direction of get_next_best through licks

    :return: corresponding lick or next best lick
    """
    if id is None: # catch no id
        return None
    lick_id_list = np.array([lick.lick_id for lick in licks])
    try:
        return [licks[i + shift] for i, lick in enumerate(licks) if lick.lick_id == id][0]
    except IndexError:
        if get_next_best is True:
            try:
                if dir == 1:
                    index = np.argmax(lick_id_list>id)
                    id = lick_id_list[index]
                if dir == -1:
                    id = [a for a in lick_id_list if a<id][-1]

                return [licks[i + shift] for i, lick in enumerate(licks) if lick.lick_id == id][0]
            except IndexError:
                return None
