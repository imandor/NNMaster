from database_api_beta import Slice
from random import seed,shuffle
import numpy as np
from scipy import stats, spatial
from itertools import takewhile, dropwhile, repeat

def position_as_map(pos_list, xstep, ystep, X_MAX, X_MIN, Y_MAX, Y_MIN):
    pos_list = np.asarray(pos_list)
    x_list = pos_list[0, :]
    y_list = pos_list[1, :]
    x_list = ((x_list - X_MIN) // xstep).astype(int)
    y_list = ((y_list - Y_MIN) // ystep).astype(int)
    pos_list = np.dstack((x_list, y_list))[0]
    pos_list = np.unique(pos_list, axis=0)
    ret = np.zeros(((X_MAX - X_MIN) // xstep, (Y_MAX - Y_MIN) // ystep))
    for pos in pos_list:
        ret[pos[0], pos[1]] = 1
    return ret


def shuffle_io(X,y,net_dict,seed_no):

    # Shuffle data
    if net_dict["SHUFFLE_DATA"] is False:
        return X,y
    SHUFFLE_FACTOR = net_dict["SHUFFLE_FACTOR"]
    seed(seed_no)

    # crop length to fit shuffle factor

    # print("Shuffling data...")
    x_length = len(X) - (len(X) % SHUFFLE_FACTOR)
    X = X[:x_length]
    y = y[:x_length]

    # Shuffle index of data

    r = np.arange(len(X))
    r = r.reshape(-1, SHUFFLE_FACTOR)
    s = np.arange(len(r)) # shuffling r directly doesnt work
    shuffle(s)
    r = r[s]
    r = r.reshape(-1)

    # shuffle data

    X = [X[j] for j in r]
    y = [y[j] for j in r]
    # print("Finished shuffling data")
    return X,y


def time_shift_io(session,shift,net_dict):
    SLICE_SIZE = net_dict["SLICE_SIZE"]
    WIN_SIZE = net_dict["WIN_SIZE"]
    X_STEP = net_dict["X_STEP"]
    Y_STEP = net_dict["Y_STEP"]
    X_MAX = net_dict["X_MAX"]
    X_MIN = net_dict["X_MIN"]
    Y_MAX = net_dict["Y_MAX"]
    Y_MIN = net_dict["Y_MIN"]
    STRIDE = net_dict["STRIDE"]
    Y_SLICE_SIZE = net_dict["Y_SLICE_SIZE"]
    # Shift positions

    shift_in_filtered_spikes = int(shift/WIN_SIZE)
    position_x = session.position_x
    position_y = session.position_y
    filtered_spikes = session.filtered_spikes
    if shift > 0:
        position_x = position_x[shift:]
        position_y = position_y[shift:]
        filtered_spikes = [x[:-shift_in_filtered_spikes] for x in filtered_spikes]

    if shift < 0:
        position_x = position_x[:shift]
        position_y = position_y[:shift]
        filtered_spikes = [x[-shift_in_filtered_spikes:] for x in filtered_spikes]
    y = []
    pos_x = position_x
    pos_y = position_y
    BINS_IN_SAMPLE = SLICE_SIZE // WIN_SIZE
    BINS_IN_STRIDE = STRIDE // WIN_SIZE
    SURROUNDING_INDEX = int(0.5 * (SLICE_SIZE - Y_SLICE_SIZE))
    X = []
    while len(filtered_spikes[0]) >= STRIDE:

        # Input
        bins_to_X = []
        remaining_bins = []
        for spike in filtered_spikes:
            bins_to_X.append(spike[:BINS_IN_SAMPLE])
            remaining_bins.append(spike[BINS_IN_STRIDE:])
        bins_to_X = np.reshape(bins_to_X, [len(bins_to_X), len(bins_to_X[0])])
        X.append(bins_to_X)
        filtered_spikes = remaining_bins

        # output

        # norm_x = (pos_x[:SLICE_SIZE] - X_MIN)/X_MAX
        # norm_y = (pos_y[:SLICE_SIZE] - X_MIN)/X_MAX
        norm_x = pos_x[:SLICE_SIZE]
        norm_y = pos_y[:SLICE_SIZE]

        posxy_list =  [np.average(norm_x[SURROUNDING_INDEX:-SURROUNDING_INDEX]),np.average(norm_y[SURROUNDING_INDEX:-SURROUNDING_INDEX]) ] # remove surrounding positional data and form average
        y.append(posxy_list)
        pos_x = pos_x[BINS_IN_STRIDE*WIN_SIZE:]
        pos_y = pos_y[BINS_IN_STRIDE*WIN_SIZE:]
        # print(len(pos_x))


    return X, y






