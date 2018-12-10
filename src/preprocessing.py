from random import seed, shuffle, randint
import numpy as np


def position_as_map(pos_list, xstep, ystep, x_max, x_min, y_max, y_min):
    """

    :param pos_list: positions as tuple [[x,x,x,x,x],[y,y,y,y,y]]
    :param xstep:
    :param ystep:
    :param x_max:
    :param x_min:
    :param y_max:
    :param y_min:
    :return:
    """
    pos_list = np.asarray(pos_list)
    if np.isscalar(pos_list[0]):
        x_list = np.array(pos_list[0])
        y_list = np.array(pos_list[1])
    else:  # if more than one entry in pos_list (standard)
        x_list = pos_list[0, :]
        y_list = pos_list[1, :]
    pos_list = np.dstack((x_list, y_list))[0]
    pos_list = np.unique(pos_list, axis=0)
    ret = np.zeros(((x_max - x_min) // xstep, (y_max - y_min) // ystep))
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
        pos_counter = np.zeros((nd.x_max - nd.x_min) // nd.x_step)
    else:
        pos_counter = np.zeros((nd.y_max - nd.y_min) // nd.y_step)

    for i, e in enumerate(y):
        y_pos = np.unravel_index(e.argmax(), e.shape)[axis]  # TODO should not work 100%
        if pos_counter[y_pos] < max_occurrences:
            x_return.append(x[i])
            pos_counter[y_pos] += 1
            y_return.append(e)
    return x_return, y_return


def count_occurrences(y, net_dict, axis=0):
    if axis == 0:
        pos_counter = np.zeros((net_dict["x_max"] - net_dict["x_min"]) // net_dict["x_step"])
    else:
        pos_counter = np.zeros((net_dict["y_max"] - net_dict["y_min"]) // net_dict["y_step"])
    for i, e in enumerate(y):
        y_pos = np.unravel_index(e.argmax(), e.shape)[axis]  # TODO should not work 100%
        pos_counter[y_pos] += 1
    return pos_counter


def shuffle_io(X, y, nd, seed_no, shuffle_batch_size=None):
    # Shuffle data
    if nd.shuffle_data is False:
        return X, y
    if shuffle_batch_size is None:
        shuffle_batch_size = nd.shuffle_factor
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
    slice_size = nd.slice_size
    win_size = nd.win_size
    x_step = nd.x_step
    y_step = nd.y_step
    x_max = nd.x_max
    x_min = nd.x_min
    y_max = nd.y_max
    y_min = nd.y_min
    stride = nd.stride
    y_slice_size = nd.y_slice_size

    # Shift positions

    shift_in_filtered_spikes = int(shift / win_size)
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
    bins_in_sample = slice_size // win_size
    bins_in_stride = stride // win_size
    surrounding_index = int(0.5 * (slice_size - y_slice_size))
    X = []
    while len(filtered_spikes[0]) >= slice_size:

        # Input

        bins_to_X = []
        remaining_bins = []
        for spike in filtered_spikes:
            bins_to_X.append(spike[:bins_in_sample])
            remaining_bins.append(spike[bins_in_stride:])
        bins_to_X = np.reshape(bins_to_X, [len(bins_to_X), len(bins_to_X[0])])
        X.append(bins_to_X)
        filtered_spikes = remaining_bins

        # output

        norm_x = pos_x[:slice_size]
        norm_y = pos_y[:slice_size]
        right_index_border = len(norm_x) - surrounding_index
        x_list = (((norm_x[surrounding_index:right_index_border]) - x_min) // x_step).astype(int)
        y_list = (((norm_y[surrounding_index:right_index_border]) - y_min) // y_step).astype(int)

        posxy_list = [x_list, y_list]  # remove surrounding positional data and form average
        y.append(position_as_map(posxy_list, x_step, y_step, x_max, x_min, y_max, y_min))
        pos_x = pos_x[bins_in_stride * win_size:]
        pos_y = pos_y[bins_in_stride * win_size:]
    return X, y


def generate_counter(num_wells, excluded_wells):
    """
    :param num_wells:
    :param excluded_wells:
    :return: empty counter for licks per well with correct size
    """
    return np.zeros(num_wells - len(excluded_wells))


def fill_counter(num_wells, excluded_wells, licks):
    counter = generate_counter(num_wells=num_wells, excluded_wells=excluded_wells)
    for i, lick in enumerate(licks):
        if lick.target is None:
            well = lick.lickwell
        else:
            well = lick.target
        if well not in excluded_wells:
            index = normalize_well(well, num_wells, excluded_wells)
            counter[index] += 1
    return counter


def normalize_well(well, num_wells, excluded_wells):
    """

    :param well:
    :param num_wells:
    :param excluded_wells:
    :return: a normalized well id
    """
    return int((well - 1 - len(excluded_wells)) % (num_wells - len(excluded_wells)))


def abs_to_logits(y_abs, num_wells):
    """

    :param y_abs:
    :param num_wells:
    :return: cast of well_id to logit format
    """
    y_i = np.zeros(num_wells)
    y_i[int(y_abs) - 1] = 1
    return y_i


def lickwells_io(session, nd, excluded_wells=[1], shift=1, normalize=False, differentiate_false_licks=False,
                 valid_licks=[]):
    # get all slices within n milliseconds around lick_well

    # Filter licks and spread them as evenly as possible

    licks = session.licks  # np.zeros(len(session.licks))
    nd.get_all_valid_lick_ids(session, start_well=1)
    nd.get_all_phase_change_ids(session=session)
    # nd.valid_licks = session.licks
    filtered_licks = []
    filtered_next_wells = []
    # create list of valid licks
    for i, lick in enumerate(licks):
        if i > - shift and i < len(licks) - shift and lick.lickwell in excluded_wells and (
                normalize is False or licks[i + shift].lickwell not in excluded_wells) and (
                valid_licks == [] or i in valid_licks):
            # exclude trailing samples to stay inside shift range
            filtered_licks.append(lick)
            next_well = int(licks[i + shift].lickwell)
            filtered_next_wells.append(next_well)
            # counter[next_well-len(excluded_wells)-1] += 1

    # distribute list of licks as evenly over well type as possible

    counter = generate_counter(nd.num_wells, excluded_wells)
    distributed_licks = [None] * (nd.num_wells * len(filtered_licks))
    distributed_next_well = np.zeros(nd.num_wells * len(filtered_licks))
    for i, well in enumerate(filtered_next_wells):
        lickwell_norm = normalize_well(well, nd.num_wells, excluded_wells)
        distributed_index = lickwell_norm + int((nd.num_wells - len(excluded_wells)) * counter[lickwell_norm])
        distributed_licks[distributed_index] = filtered_licks[i]
        distributed_next_well[distributed_index] = well
        counter[lickwell_norm] += 1

    # remove zeros and convert back to list

    licks = []
    next_well = []
    for i, e in enumerate(distributed_licks):
        if e != None:
            licks.append(e)
            next_well.append(int(distributed_next_well[i]))
    y_abs = []
    X = []
    metadata = []
    # Generate input and output
    for i, lick in enumerate(licks):
        lick_start = int(lick.time - 5000)
        lick_end = int(lick.time + 5000)
        slice = session[lick_start:lick_end]
        for j in range(0, 10000 // nd.win_size - 11):
            bins_to_x = [c[j:j + 11] for c in slice.filtered_spikes]
            bins_to_x = np.reshape(bins_to_x, [len(bins_to_x), len(bins_to_x[0])])
            X.append(bins_to_x)
            if differentiate_false_licks is False:
                y_abs.append(next_well[i])
                lick.target = next_well[i]
                metadata.append(lick)
            # else:
            # TODO
            # y_abs.append(next_well +(licks[i + shift].rewarded*nd.lw_classifications))

    print("Lickwell count:")
    unique, counts = np.unique(y_abs, return_counts=True)
    print(unique, counts)

    # remove instances of double licks at lick_well
    for i, y in enumerate(y_abs):
        if y in excluded_wells:
            y_abs.pop(i)
            X.pop(i)

    # Convert y to logits format

    y = []
    for val in y_abs:
        y_i = np.zeros(nd.num_wells)
        y_i[int(val) - 1] = 1
        y.append(y_i)
    return X, y, metadata
