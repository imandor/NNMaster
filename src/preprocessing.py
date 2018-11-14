from random import seed, shuffle
import numpy as np


def position_as_map(pos_list, xstep, ystep, X_MAX, X_MIN, Y_MAX, Y_MIN):
    pos_list = np.asarray(pos_list)
    if np.isscalar(pos_list[0]):
        x_list = np.array(pos_list[0])
        y_list = np.array(pos_list[1])
    else:  # if more than one entry in pos_list (standard)
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


def filter_overrepresentation_discrete(x, y, max_occurrences):
    print("Filtering overrepresentation")
    x_return = []
    y_return = []
    y = np.array(y)
    # y = y[0, :, :, 0]
    pos_counter = np.zeros(5)
    for i, e in enumerate(y):
        y_pos = np.max(e)-1
        if pos_counter[y_pos] < max_occurrences:
            x_return.append(x[i])
            pos_counter[y_pos] += 1
            y_return.append(e)
            # y_return.append(position_as_map([e[0],e[1]], net_dict["X_STEP"],net_dict["Y_STEP"],net_dict["X_MAX"], net_dict["X_MIN"], net_dict["Y_MAX"], net_dict["Y_MIN"]))
    return x_return, y_return


def filter_overrepresentation_map(x, y, max_occurrences, nd, axis=0):
    print("Filtering overrepresentation")
    x_return = []
    y_return = []
    y = np.array(y)
    # y = y[0, :, :, 0]
    if axis == 0:
        pos_counter = np.zeros((nd.X_MAX - nd.X_MIN) // nd.X_STEP)
    else:
        pos_counter = np.zeros((nd.Y_MAX - nd.Y_MIN) // nd.Y_STEP)

    for i, e in enumerate(y):
        y_pos = np.unravel_index(e.argmax(), e.shape)[axis]  # TODO should not work 100%
        if pos_counter[y_pos] < max_occurrences:
            x_return.append(x[i])
            pos_counter[y_pos] += 1
            y_return.append(e)
            # y_return.append(position_as_map([e[0],e[1]], net_dict["X_STEP"],net_dict["Y_STEP"],net_dict["X_MAX"], net_dict["X_MIN"], net_dict["Y_MAX"], net_dict["Y_MIN"]))
    return x_return, y_return


def count_occurrences(y, net_dict, axis=0):
    if axis == 0:
        pos_counter = np.zeros((net_dict["X_MAX"] - net_dict["X_MIN"]) // net_dict["X_STEP"])
    else:
        pos_counter = np.zeros((net_dict["Y_MAX"] - net_dict["Y_MIN"]) // net_dict["Y_STEP"])
    for i, e in enumerate(y):
        y_pos = np.unravel_index(e.argmax(), e.shape)[axis]  # TODO should not work 100%
        pos_counter[y_pos] += 1
    return pos_counter


def shuffle_io(X, y, nd, seed_no):
    # Shuffle data
    if nd.SHUFFLE_DATA is False:
        return X, y
    SHUFFLE_FACTOR = nd.SHUFFLE_FACTOR
    seed(seed_no)

    # crop length to fit shuffle factor

    # print("Shuffling data...")
    x_length = len(X) - (len(X) % SHUFFLE_FACTOR)
    X = X[:x_length]
    y = y[:x_length]

    # Shuffle index of data

    r = np.arange(len(X))
    r = r.reshape(-1, SHUFFLE_FACTOR)
    s = np.arange(len(r))  # shuffling r directly doesnt work
    shuffle(s)
    r = r[s]
    r = r.reshape(-1)

    # shuffle data

    X = [X[j] for j in r]
    y = [y[j] for j in r]
    # print("Finished shuffling data")
    return X, y


def time_shift_positions(session, shift, nd):
    SLICE_SIZE = nd.SLICE_SIZE
    WIN_SIZE = nd.WIN_SIZE
    X_STEP = nd.X_STEP
    Y_STEP = nd.Y_STEP
    X_MAX = nd.X_MAX
    X_MIN = nd.X_MIN
    Y_MAX = nd.Y_MAX
    Y_MIN = nd.Y_MIN
    STRIDE = nd.STRIDE
    Y_SLICE_SIZE = nd.Y_SLICE_SIZE

    # Shift positions

    shift_in_filtered_spikes = int(shift / WIN_SIZE)
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

        posxy_list = [x_list, y_list]  # remove surrounding positional data and form average
        y.append(position_as_map(posxy_list, X_STEP, Y_STEP, X_MAX, X_MIN, Y_MAX, Y_MIN))
        pos_x = pos_x[BINS_IN_STRIDE * WIN_SIZE:]
        pos_y = pos_y[BINS_IN_STRIDE * WIN_SIZE:]
        # print(len(pos_x))

    return X, y


def lickwells_io(session, X, nd, allowed_distance, filter=None,normalize=False):
    """
    :param session:
    :param X:
    :param nd:
    :param allowed_distance:
    :param filter: starting well, is default None, set to 1 to exclude lick well 1
    :param normalize:
    :return:
    """
    slice_size = nd.SLICE_SIZE
    y_abs = []
    licks = []
    for i in range(len(session.licks)): # Remove all instances of well 1
        if session.licks[i]["lickwell"] != filter:
            licks.append(session.licks[i])
    if filter is not None:  # conditional filter, currently only well 1 implemented
        if filter == 1:
            filter_well_x = 30
            filter_well_y = 145

    session_length = len(X) * slice_size
    for i in range(0, session_length, slice_size):
        if len(licks) == 0:  # last samples have no corresponding well
            X.pop(-1)  # order does not matter

        else:
            if filter is not None:  # conditional filter, only keep samples close to well 1
                pos_x = session.position_x[i]
                pos_y = session.position_y[i]
                distance = np.sqrt(np.square(filter_well_x - pos_x) + np.square(filter_well_y - pos_y))
                if distance > allowed_distance:
                    X.pop(i // slice_size)
                else:
                    if licks[0]["lickwell"] == filter:
                        print(i,licks[1])
                        y_abs.append(licks[1]["lickwell"])
                    else:
                        y_abs.append(licks[0]["lickwell"])
            else:
                y_abs.append(licks[0]["lickwell"])
            if licks[0]["time"] <= i:
                licks.pop(0)
    print("Lickwell count:")
    unique, counts = np.unique(y_abs, return_counts=True)
    print(unique, counts)
    y = []
    if normalize is True:
        X,y_abs = filter_overrepresentation_discrete(X, y_abs, min(counts))
    for abs in y_abs:
        y_i = np.zeros(5)
        y_i[abs-1] = 1
        y.append(y_i)
    unique, counts = np.unique(y_abs, return_counts=True)
    print(unique, counts)

    return X, y
