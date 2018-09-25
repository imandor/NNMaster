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
    BINS_BEFORE = net_dict["BINS_BEFORE"]
    BINS_AFTER = net_dict["BINS_AFTER"]

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
    while len(pos_x)>=SLICE_SIZE:
        posxy_list = []
        posxy_list.append(pos_x[:SLICE_SIZE])
        posxy_list.append(pos_y[:SLICE_SIZE])
        y.append(position_as_map(posxy_list, X_STEP, Y_STEP, X_MAX, X_MIN, Y_MAX, Y_MIN))
        pos_x = pos_x[SLICE_SIZE:]
        pos_y = pos_y[SLICE_SIZE:]
        # print(len(pos_x))

    if BINS_AFTER != 0 or BINS_BEFORE != 0:
        print("Adding surrounding neural activity to spike bins...")
        # crop unusable values
        y = y[BINS_BEFORE:-BINS_AFTER]
    X = preprocess_raw_data(filtered_spikes, net_dict)
    return X, y




def preprocess_raw_data(filtered_spikes,net_dict):
    SLICE_SIZE = net_dict["SLICE_SIZE"]
    WIN_SIZE = net_dict["WIN_SIZE"]
    BINS_BEFORE = net_dict["BINS_BEFORE"]
    BINS_AFTER = net_dict["BINS_AFTER"]

    # bin and normalize input and create metadata


    # total = len(position_x) // SLICE_SIZE
    BINS_IN_SAMPLE = SLICE_SIZE // WIN_SIZE
    X = []
    counter = 0
    while len(filtered_spikes[0]) >= BINS_IN_SAMPLE:
        first_half = []
        second_half = []
        for spike in filtered_spikes:
            first_half.append(spike[:BINS_IN_SAMPLE])
            second_half.append(spike[BINS_IN_SAMPLE:])
        first_half = np.reshape(first_half,[len(first_half),len(first_half[0])])
        X.append(first_half)
        filtered_spikes = second_half
        counter = counter + 1
        # print("slicing", counter, "of", total)

    #Increase range of X values

    if BINS_AFTER != 0 or BINS_BEFORE != 0:
        print("Adding surrounding neural activity to spike bins...")
        # crop unusable values
        X_c = X[BINS_BEFORE:-BINS_AFTER]
        # increase x-range

        for i, x in enumerate(X_c):
            X_c[i] = np.concatenate([a for a in X[i:i + BINS_BEFORE + BINS_AFTER + 1]], axis=1)
        X = X_c

    print("Finished slicing data")
    return X

