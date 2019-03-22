from random import seed, shuffle, randint
import numpy as np
import random


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


def shuffle_list_key(length,shuffle_batch_size=1,seed_no=1):
    """

    :param length:
    :param shuffle_batch_size:
    :param seed_no:
    :return: helpfunction: list of indizes so multiple lists can be shuffled together
    """
    seed(seed_no)
    r = np.arange(length)
    r = r.reshape(-1, shuffle_batch_size)
    s = np.arange(len(r))  # shuffling r directly doesnt work
    shuffle(s)
    r = r[s]
    r = [u for g in r for u in g]
    return r


def shuffle_io(X, y, nd, seed_no=2, shuffle_batch_size=1):
    if nd.shuffle_data is False:
        return X,y
    modulo = len(X) % shuffle_batch_size
    if modulo!=0:
        X = X[0:-modulo]
        y  = y[0:-modulo]
    r = shuffle_list_key(len(y), shuffle_batch_size=shuffle_batch_size, seed_no=seed_no)
    X = [X[i] for i in r]
    y = [y[i] for i in r]
    return X, y

def position_to_1d_map(positions,min,max,step):
    """
    :param positions: list of positions
    :param min: minimum value on map
    :param max: maximum value on map
    :return: numpy array containing ones where positions were and zero otherwise
    """

    positions = np.asarray(positions)
    positions = np.dstack(positions)[0]
    pos_list = np.unique(positions)
    ret = np.zeros((max - min) // step)
    for pos in pos_list:
        try:
            # ret[pos[0], pos[1]] = 1
            ret[int(pos//step)] = 1
        except IndexError:
            print("Warning, check if pos_list is formatted correctly ([[x,x,x,x,x]")
    return ret

def time_shift_positions(session, shift, nd):
    """

    :param session: Slice object containing session
    :param shift: time shift intended for training
    :param nd: Net data object
    :return: neural data and corresponding labels for each time shift
    """
    win_size = nd.win_size
    position_x = session.position_x

    # Shift position index and determine which range of filtered spikes contains valid samples
    if shift>=0:
        position_x = position_x[shift:]
        ystart = 0
        ystop = len(session.filtered_spikes[0])-shift//win_size
    if shift<0:
        position_x = position_x[:shift]
        ystart = - shift//win_size # how many samples are excluded from data set
        ystop = len(session.filtered_spikes[0])

    # map positions:

    # positions = position_to_1d_map(position_x,nd.x_min,nd.x_max,nd.x_step)
    X = []
    y = []
    xshape =  [len(session.filtered_spikes)] + [nd.number_of_bins]
    for i in range(ystart,ystop-nd.number_of_bins,nd.number_of_bins):
        # Append input data for given range
        # spikes = np.zeros(xshape)
        # for j, neuron_spikes in enumerate(session.filtered_spikes):
        #     spikes[j] = np.array(neuron_spikes[i:i+nd.number_of_bins])
        spikes = [neuron_spikes[i:i+nd.number_of_bins] for neuron_spikes in session.filtered_spikes]
        # spikes_as_array = np.zeros((len(spikes),len(spikes[0])))
        # for i,neuron_spikes in enumerate(spikes):
        #     for j,spike in enumerate(neuron_spikes):
        #         spikes_as_array[i][j] = spike
        X.append(spikes) # Input data for given range

        # determine which y range is used
        range_start = (i-ystart) * win_size
        range_stop = (i + nd.number_of_bins-ystart) * win_size
        y_raw = position_x[range_start:range_stop]
        y_raw = position_to_1d_map(positions=y_raw, min=nd.x_min, max=nd.x_max, step=nd.x_step)
        y.append(y_raw)

    return X, y


def return_well_positions(session):
    """

    :param session:
    :return: approximates the well positions by averaging over the positions at each lick event
    """
    position_by_well = []
    position_counter = np.zeros((10,len(session.licks)))
    for i, lick in enumerate(session.licks):
        position_counter[lick.lickwell-1][i] = session.position_x[int(lick.time)]
    for j,well in enumerate(position_counter):
        avg_pos = []
        for i,pos in enumerate(well):
            if pos!=0:
                avg_pos.append(pos)
        if avg_pos!= []:
            position_by_well.append(np.average(avg_pos))
    return position_by_well

def filter_behavior_component(X,y,nd,session,allowed_distance=3):
    """
    :param X:
    :param y:
    :param nd: net_data object
    :return: removes neural data not corresponding to nd.behavior_component_filter
    """
    filter = nd.behavior_component_filter
    if filter is None or filter == "None":
        return X,y
    X_return = []
    y_return = []

    if filter == "at lickwell":
        well_positions = return_well_positions(session)
        for i, position_as_map in enumerate(y):
            position = nd.x_step * median_position_bin(position_as_map)
            for well_position in well_positions:
                if abs(position-well_position)<allowed_distance:
                    X_return.append(X[i])
                    y_return.append(position_as_map)
                    break

    if filter == "not at lickwell":
        well_positions = return_well_positions(session)
        for i, position_as_map in enumerate(y):
            found_at_well = False
            position = nd.x_step * median_position_bin(position_as_map)
            for well_position in well_positions:
                if abs(position-well_position)<allowed_distance:
                    found_at_well = True
                    break
            if found_at_well is False:
                X_return.append(X[i])
                y_return.append(position_as_map)

    if filter == "correct trials" or filter == "false trials":
        start_index = 0
        for lick in session.licks:
            stop_index = int(lick.time - nd.time_shift) // (nd.win_size*nd.number_of_bins)
            if filter == "correct trials" and lick.rewarded == 1 or filter == "false trials" and lick.rewarded == 0:
                X_return.append(X[start_index:stop_index])
                y_return.append(y[start_index:stop_index])
            start_index = stop_index
        X_return = [a for li in X_return for a in li]
        y_return = [a for li in y_return for a in li]


    return X_return,y_return

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


def abs_to_logits(y_abs, num_classifiers, max_well_no):
    """

    :param y_abs:
    :param num_wells:
    :return: cast of well_id to logit format
    """
    y_i = np.zeros(num_classifiers)
    shift = num_classifiers-max_well_no-1
    y_i[int(y_abs)+shift] = 1
    return y_i


def lickwells_io(session, nd, excluded_wells=[1], shift=1,
                 valid_licks=[], lickstart=0, lickstop=5000, target_is_phase=False):
    """

    :param session: session object
    :param nd: net_data object
    :param excluded_wells: wells to be excluded from test. Note that by now well 1 is hard coded in some places, so changing this parameter would require further clean up elsewhere
    :param shift: if 1, target is set as next well, if -1 as last well
    :param valid_licks: predetermined list of valid lick_ids (TODO)
    :param lickstart: start of recording relative to lick_time
    :param lickstop: end of recording relative to lick_time
    :param target_is_phase: if True, sets target to current training phase rather than actually visited well
    :param start_time_by_lick_id: can contain list of tuples of (lick_id,time). times are set as starting time of lick for lick event(otherwise defaults to zero for each event)
    :return: samples in format x (list of slices) ,y (list of labels), updated nd, updated session
    """
    # get all slices within n milliseconds around lick_well

    # Filter licks and spread them as evenly as possible

    nd.get_all_valid_lick_ids(session, start_well=1, lickstart = lickstart, lickstop=lickstop, shift=shift)
    nd = session.add_lick_data_to_session_and_net_data(nd=nd)
    licks = session.licks
    # for i, lick in enumerate(licks):
    #     if lick.lickwell!=1:
    #         pass
    #         # lick.lickwell = random.randint(2,5)
    #     else:
    #         if i>=2:
    #             print(licks[i].lick_id,",",licks[i-2].lickwell)
    filtered_licks = []
    filtered_next_wells = []

    # create list of valid licks
    for i, lick in enumerate(licks):
        if  lick.lick_id in nd.valid_licks:
            # exclude trailing samples to stay inside shift range
            filtered_licks.append(lick)
            next_well = int(licks[i + shift].lickwell)
            if target_is_phase is True:
                next_well = lick.phase
            filtered_next_wells.append(next_well)

    # # shuffle
    #
    # r = shuffle_list_key(length=len(filtered_next_wells),shuffle_batch_size=1,seed_no=0)
    # filtered_next_wells = [filtered_next_wells[j] for j in r]
    # filtered_licks = [filtered_licks[j] for j in r]


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
    X = []
    y = []
    # Generate input and output

    for i,lick in enumerate(licks):
        if nd.start_time_by_lick_id is not None:
            offset = [i[1] for i in nd.start_time_by_lick_id if i[0]==lick.lick_id][0]
        else:
            offset = 0
        lick_start = int(lick.time + lickstart+offset)
        lick.target = next_well[i]
        lick_end = int(lick.time + lickstop+offset)
        X.append(session[lick_start+offset:lick_end+offset])
        y.append(lick)
    return X, y,nd,session


def median_position_bin(mapping):
    """

    :param mapping: list of positions in map format
    :return: median position bin over [x-coordinate,y-coordinate]
    """
    # sum_0 = np.sum(mapping, axis=0)
    # sum_0 = np.sum(np.sum(sum_0 * np.arange(mapping.shape[1]))) / np.sum(np.sum(sum_0))
    # sum_1 = np.sum(mapping, axis=1)
    # sum_1 = np.sum(np.sum(sum_1 * np.arange(mapping.shape[0]))) / np.sum(np.sum(sum_1))
    try:
        sum = np.sum(mapping * np.arange(len(mapping))) / np.sum(mapping)
        return int(sum)
    except ValueError:
        print("Value error, check network output")
        return "Value error, check network output"