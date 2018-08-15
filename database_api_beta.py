from multiprocessing.pool import ThreadPool
import numpy as np
import matplotlib.pyplot as plt
from itertools import takewhile, dropwhile, repeat
from src import OpenEphys
import scipy
import bisect
import glob
import pickle
import time

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
        return self._func(x / self.search_radius) / self._integral_on_search_window


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
    printcounter = 0
    tic = time.process_time()
    tock = 0
    for index in range(n_bin_points):

        # if (printcounter == 100):
        #     toc = time.process_time()
        #     tock = toc + tock - tic
        #     print(toc - tic)
        #     print(tock)
        #     print("Spike:",neuron_spikes[index_first_spike_in_window])
        #     tic = time.process_time()
        #     printcounter = 0
        # printcounter += 1
        curr_spikes_in_search_window = dropwhile(lambda x: x < curr_search_window_min_bound,
                                                 takewhile(lambda x: x < curr_search_window_max_bound,
                                                           neuron_spikes[index_first_spike_in_window:]))

        curr_spikes_in_search_window = list(curr_spikes_in_search_window)
        curr_search_window_min_bound += filter_func.step_size
        curr_search_window_max_bound += filter_func.step_size
        if len(curr_spikes_in_search_window) == 0:
            continue
        filtered_spikes[index] = sum(map(
        lambda x: filter_func(x - index * filter_func.step_size),
        curr_spikes_in_search_window))


        for spike_index, spike in enumerate(neuron_spikes[index_first_spike_in_window:index_first_spike_in_window+curr_search_window_max_bound]): # upper bound because a maximum of 1 spike per ms can occurr and runtime of slice operation is O(i2-i1)
            if spike >= curr_search_window_min_bound:
                index_first_spike_in_window = index_first_spike_in_window + spike_index
                break
        # index_first_spike_in_window += index_first_spike_in_curr_window
        # toc = time.process_time()
        # print(toc - tic)

    # print("Finished convolving neuron ",neuron_counter, " of ", n_neurons,"...")
    return filtered_spikes

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
            ret = p.starmap(_convolve_thread_func, zip(repeat(self._filter), repeat(n_bin_points),range(0,len(self.spikes)), repeat(len(self.spikes)),self.spikes))
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

    def plot(self, ax_filtered_spikes=None, ax_raw_spikes=None, ax_licks=None, ax_trial_timestamps=None, filtered_spikes_kwargs={}):
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
        stop = self.end_time if time_slice.stop is None else time_slice.stop
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
        stop = self.end_time if time_slice.stop is None else time_slice.stop
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
                            if (eventtype_ttl[ind], eventch_ttl[ind], eventid_ttl[ind], recording_num[ind]) == (3, 2, 1, 0)]

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
        trial_timestamp = [{"time": initial_detection_timestamp[ind],
                            "trial_lickwell": well,
                            "trial_id": ind}
                           for ind, well in enumerate(lickwells) if rewarded[ind] == 1]
        # licks
        licks = [{"time": initial_detection_timestamp[i],
                  "lickwell": lickwells[i],
                  "rewarded": rewarded[i]}
                 for i in range(len(initial_detection_timestamp))]
        print("finished loading session")
        return cls(spikes, licks, position_x, position_y, speed, trial_timestamp)

    @classmethod
    def from_pickle(cls, path):
        with open(path, 'rb') as f:
            data = pickle.load(f)
            return data

    def to_pickle(self, path):
        with open(path, 'wb') as f:
            pickle.dump(self, f)
