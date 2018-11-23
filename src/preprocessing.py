from random import seed, shuffle, randint
import numpy as np


def position_as_map(pos_list, xstep, ystep, X_MAX, X_MIN, Y_MAX, Y_MIN):
    pos_list = np.asarray(pos_list)
    if np.isscalar(pos_list[0]):
        x_list = np.array(pos_list[0])
        y_list = np.array(pos_list[1])
    else:  # if more than one entry in pos_list (standard)
        x_list = pos_list[0, :]
        y_list = pos_list[1, :]
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





def filter_overrepresentation_map(x, y, max_occurrences, nd, axis=0):
    print("Filtering overrepresentation")
    x_return = []
    y_return = []
    y = np.array(y)
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


def shuffle_io(X, y, nd, seed_no, shuffle_batch_size=None):
    # Shuffle data
    if nd.SHUFFLE_DATA is False:
        return X, y
    if shuffle_batch_size is None:
        shuffle_batch_size = nd.SHUFFLE_FACTOR
    seed(seed_no)

    # crop length to fit shuffle factor

    # print("Shuffling data...")
    x_length = len(X) - (len(X) % shuffle_batch_size)
    X = X[:x_length]
    y = y[:x_length]

    # Shuffle index of data

    r = np.arange(len(X))
    r = r.reshape(-1, shuffle_batch_size)
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

        norm_x = pos_x[:SLICE_SIZE]
        norm_y = pos_y[:SLICE_SIZE]
        right_index_border = len(norm_x) - SURROUNDING_INDEX
        x_list = (((norm_x[SURROUNDING_INDEX:right_index_border]) - X_MIN) // X_STEP).astype(int)
        y_list = (((norm_y[SURROUNDING_INDEX:right_index_border]) - Y_MIN) // Y_STEP).astype(int)

        posxy_list = [x_list, y_list]  # remove surrounding positional data and form average
        y.append(position_as_map(posxy_list, X_STEP, Y_STEP, X_MAX, X_MIN, Y_MAX, Y_MIN))
        pos_x = pos_x[BINS_IN_STRIDE * WIN_SIZE:]
        pos_y = pos_y[BINS_IN_STRIDE * WIN_SIZE:]
    return X, y


def generate_counter(num_wells,excluded_wells):
    """
    :param num_wells:
    :param excluded_wells:
    :return: empty counter for licks per well with correct size
    """
    return np.zeros(num_wells-len(excluded_wells))

def normalize_well(well,num_wells,excluded_wells):
    return int((well - 1 - len(excluded_wells)) % (num_wells - len(excluded_wells)))


def get_all_valid_licks(session,start_well = 1,change_is_valid=False):
    """

    :param session: session object
    :param start_well: dominant well in training phase (usually well 1)
    :param change_is_valid: if True exclude licks without change in training phase, if False licks with change
    :return: a list of indizes of licks corresponding to filter
    """

    licks = session.licks
    current_phase_well = None
    filtered_licks = []
    for i,lick in enumerate(licks[0:-2]):
        well = lick["lickwell"]
        next_well = licks[i+1]["lickwell"]
        if current_phase_well is None and well != start_well: # set well that is currently being trained for
            current_phase_well = well
        if current_phase_well is not None and well == 1: # append lick if it fits valid_filter
            if (change_is_valid is True and next_well!= current_phase_well) or (change_is_valid is not True and next_well==current_phase_well):
                filtered_licks.append(lick)
            if next_well != current_phase_well: # change phase if applicable
                current_phase_well = next_well
    return filtered_licks




def lickwells_io(session, nd, excluded_wells=[1], shift=1, normalize=False,differentiate_false_licks = False,valid_licks=[]):
    # get all slices within n milliseconds around lick_well



    # Filter licks and spread them as evenly as possible

    licks = session.licks # np.zeros(len(session.licks))
    filtered_licks = []
    filtered_next_wells = []
    # create list of valid licks
    for i, lick in enumerate(licks):
        if lick["rewarded"] == 1:
            print(i,lick["lickwell"])
        if i > - shift and i < len(licks) - shift and lick["lickwell"] == np.any(excluded_wells) and (
                normalize is False or licks[i + shift][
            "lickwell"] != np.any(excluded_wells)) and (valid_licks == [] or i == np.any(valid_licks)):
            # exclude trailing samples to stay inside shift range
            filtered_licks.append(lick)
            next_well = int(licks[i+shift]["lickwell"])
            filtered_next_wells.append(next_well)
            # counter[next_well-len(excluded_wells)-1] += 1



    # distribute list of licks as evenly over well type as possible

    counter = generate_counter(nd.num_wells,excluded_wells)
    distributed_licks = [None]*(nd.num_wells*len(filtered_licks))
    distributed_next_well = np.zeros(nd.num_wells*len(filtered_licks))
    for i,well in enumerate(filtered_next_wells):
        lickwell_norm = normalize_well(well,nd.num_wells,excluded_wells)
        distributed_index = lickwell_norm + int((nd.num_wells-len(excluded_wells))*counter[lickwell_norm])
        distributed_licks[distributed_index] = filtered_licks[i]
        distributed_next_well[distributed_index] = well
        counter[lickwell_norm] += 1

    # remove zeros and convert back to list

    licks = []
    next_well = []
    for i,e in enumerate(distributed_licks):
        if e != None:
            licks.append(e)
            next_well.append(int(distributed_next_well[i]))
    y_abs = []
    X = []

    # Generate input and output
    for i, lick in enumerate(licks):
        lick_start = int(lick["time"] - 5000)
        lick_end = int(lick["time"] + 5000)
        slice = session[lick_start:lick_end]
        for j in range(0, 10000 // nd.WIN_SIZE - 11):
            bins_to_x = [c[j:j + 11] for c in slice.filtered_spikes]
            bins_to_x = np.reshape(bins_to_x, [len(bins_to_x), len(bins_to_x[0])])
            X.append(bins_to_x)
            if differentiate_false_licks is False:
                y_abs.append(next_well[i])
            # else:
                # TODO
                # y_abs.append(next_well +(licks[i + shift]["rewarded"]*nd.lw_classifications))

    print("Lickwell count:")
    unique, counts = np.unique(y_abs, return_counts=True)
    print(unique, counts)

    # remove instances of double licks at lick_well
    for i,y in enumerate(y_abs):
        if y == np.any(excluded_wells):
            y_abs.pop(i)
            X.pop(i)

    # Convert y to logits format

    y = []
    for val in y_abs:
        y_i = np.zeros(nd.num_wells)
        y_i[int(val) - 1] = 1
        y.append(y_i)
    return X, y
