from random import seed,shuffle
import numpy as np


def position_as_map(pos_list, xstep, ystep, X_MAX, X_MIN, Y_MAX, Y_MIN):
    pos_list = np.asarray(pos_list)
    if np.isscalar(pos_list[0]):
        x_list = np.array(pos_list[0])
        y_list = np.array(pos_list[1])
    else: # if more than one entry in pos_list (standard)
        x_list = pos_list[0, :]
        y_list = pos_list[1, :]
    asd_1 = x_list
    asd_2 = y_list
    pos_list = np.dstack((x_list, y_list))[0]
    pos_list = np.unique(pos_list, axis=0)
    ret = np.zeros(((X_MAX - X_MIN) // xstep, (Y_MAX - Y_MIN) // ystep))
    for pos in pos_list:
        try:
            # ret[pos[0], pos[1]] = 1
            ret[int(pos[0]), int(pos[1])] = 1
        except IndexError:
            print("Warning, check if pos_list is formatted correctly ([[x,x,x,x,x],[y,y,y,y,y]]")
    return ret


def filter_overrepresentation(x,y,max_occurrences,net_dict,axis=0):
    print("Filtering overrepresentation")
    x_return = []
    y_return = []
    y = np.array(y)
    # y = y[0, :, :, 0]
    if axis == 0:
        pos_counter = np.zeros((net_dict["X_MAX"]-net_dict["X_MIN"]) // net_dict["X_STEP"])
    else:
        pos_counter = np.zeros((net_dict["Y_MAX"]-net_dict["Y_MIN"]) // net_dict["Y_STEP"])

    for i,e in enumerate(y):
        y_pos = np.unravel_index(e.argmax(), e.shape)[axis] # TODO should not work 100%
        if pos_counter[y_pos]<max_occurrences:
            x_return.append(x[i])
            pos_counter[y_pos]+=1
            y_return.append(e)
            # y_return.append(position_as_map([e[0],e[1]], net_dict["X_STEP"],net_dict["Y_STEP"],net_dict["X_MAX"], net_dict["X_MIN"], net_dict["Y_MAX"], net_dict["Y_MIN"]))
    count = np.count_nonzero(pos_counter == max_occurrences)
    avg = np.average(pos_counter)
    n_samples = np.sum(pos_counter)
    return x_return,y_return


def count_occurrences(y,net_dict,axis=0):
    if axis == 0:
        pos_counter = np.zeros((net_dict["X_MAX"]-net_dict["X_MIN"]) // net_dict["X_STEP"])
    else:
        pos_counter = np.zeros((net_dict["Y_MAX"]-net_dict["Y_MIN"]) // net_dict["Y_STEP"])
    for i,e in enumerate(y):
        y_pos = np.unravel_index(e.argmax(), e.shape)[axis] # TODO should not work 100%
        pos_counter[y_pos] += 1
    return pos_counter

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
    while len(filtered_spikes[0]) >= SLICE_SIZE:

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
        right_index_border = len(norm_x) - SURROUNDING_INDEX
        x_list = (((norm_x[SURROUNDING_INDEX:right_index_border]) - X_MIN) // X_STEP).astype(int)
        y_list = (((norm_y[SURROUNDING_INDEX:right_index_border]) - Y_MIN) // Y_STEP).astype(int)

        posxy_list =  [x_list,y_list] # remove surrounding positional data and form average
        y.append(position_as_map(posxy_list, X_STEP, Y_STEP, X_MAX, X_MIN, Y_MAX, Y_MIN))
        pos_x = pos_x[BINS_IN_STRIDE*WIN_SIZE:]
        pos_y = pos_y[BINS_IN_STRIDE*WIN_SIZE:]
        # print(len(pos_x))

    return X, y




