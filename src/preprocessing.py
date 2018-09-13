from database_api_beta import Slice
from random import seed,shuffle
import numpy as np
from scipy import stats, spatial

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

def preprocess_raw_data(network_dict):
    RAW_DATA_PATH = network_dict["RAW_DATA_PATH"]
    INITIAL_TIMESHIFT = network_dict["INITIAL_TIMESHIFT"]
    TIME_SHIFT_STEPS = network_dict["TIME_SHIFT_STEPS"]
    TIME_SHIFT_ITER = network_dict["TIME_SHIFT_ITER"]
    SLICE_SIZE = network_dict["SLICE_SIZE"]
    X_STEP = network_dict["X_STEP"]
    Y_STEP = network_dict["Y_STEP"]
    X_MAX = network_dict["X_MAX"]
    Y_MAX = network_dict["Y_MAX"]
    X_MIN = network_dict["X_MIN"]
    Y_MIN = network_dict["Y_MIN"]
    BINS_AFTER = network_dict["BINS_AFTER"]
    BINS_BEFORE = network_dict["BINS_BEFORE"]
    SHUFFLE_DATA = network_dict["SHUFFLE_DATA"]
    SHUFFLE_FACTOR = network_dict["SHUFFLE_FACTOR"]


    # Load and filter session

    session = Slice.from_raw_data(RAW_DATA_PATH)
    session.neuron_filter(100)
    print("Convolving data...")
    session.set_filter(network_dict["session_filter"])
    print("Finished convolving data")
    session.filtered_spikes = stats.zscore(session.filtered_spikes,axis=1) # Z Score neural activity
    session.to_pickle("slice.pkl")
    session = Slice.from_pickle("slice.pkl")
    shifted_positions_list = []
    copy_session = None

    # Make list of outputs for all time shifts

    for z in range(0, TIME_SHIFT_STEPS):
        copy_session = session.timeshift_position(INITIAL_TIMESHIFT + z * TIME_SHIFT_ITER)
        shifted_positions_list.append(
            [copy_session.position_x, copy_session.position_y])

    X = []
    y_list = [[] for i in range(len(shifted_positions_list))]
    metadata = []

    # bin and normalize input and create metadata

    while len(copy_session.position_x) >= SLICE_SIZE:
        try:
            metadata.append(dict(lickwells=copy_session.licks[0], time=copy_session.absolute_time_offset,
                                 position=copy_session.position_x[0]))
        except:
            metadata.append(dict(rewarded=0, time=0, lickwell=0))  # no corresponding y-value

        data_slice = copy_session[0:SLICE_SIZE]
        copy_session = copy_session[SLICE_SIZE:]
        X.append(data_slice.filtered_spikes)
        # map of shifted positions_list

        for i in range(len(shifted_positions_list)):
            posxy_list = [shifted_positions_list[i][0][:SLICE_SIZE], shifted_positions_list[i][1][:SLICE_SIZE]]
            y_list[i].append(position_as_map(posxy_list, X_STEP, Y_STEP, X_MAX, X_MIN, Y_MAX, Y_MIN))
            shifted_positions_list[i] = [shifted_positions_list[i][0][SLICE_SIZE:],
                                         shifted_positions_list[i][1][SLICE_SIZE:]]
        print("slicing", len(X), "of", len(session.position_x) // SLICE_SIZE)
    print("Finished slicing data")

    # Increase range of X values

    if BINS_AFTER != 0 or BINS_BEFORE != 0:
        print("Adding surrounding neural activity to spike bins...")
        # crop unusable values
        X_c = X[BINS_BEFORE:-BINS_AFTER]
        y_list = [a[BINS_BEFORE:-BINS_AFTER] for a in y_list]
        metadata = metadata[BINS_BEFORE:-BINS_AFTER]
        # increase x-range

        for i, x in enumerate(X_c):
            X_c[i] = np.concatenate([a for a in X[i:i + BINS_BEFORE + BINS_AFTER + 1]], axis=1)
        X = X_c
        print("Finished adding surrounding neural activity to spike bins")

    # Shuffle data

    seed(2)
    if SHUFFLE_DATA is True:

        # crop length to fit shuffle factor
        print("Shuffling data...")
        x_length = len(X) - (len(X) % SHUFFLE_FACTOR)
        X = X[:x_length]
        metadata = metadata[:x_length]
        y_list = [y[:x_length] for y in y_list]

        # Shuffle index of data

        r = np.arange(len(X))
        r = r.reshape(-1, SHUFFLE_FACTOR)
        s = np.arange(len(r)) # shuffling r directly doesnt work
        shuffle(s)
        r = r[s]
        r = r.reshape(-1)

        # shuffle data

        X = [X[j] for j in r]
        metadata = [metadata[j] for j in r]
        li = []
        for y in y_list:
            li.append([y[j] for j in r])
        y_list = li
        print("Finished shuffling data")
    return X, y_list,metadata