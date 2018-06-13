from src.models import SimpleRNNDecoder, LSTMDecoder
from session_loader import make_dense_np_matrix
from database_api_beta import Slice
import numpy as np
from settings import save_as_pickle, load_pickle
from keras.models import load_model
from src.filters import bin_filter
import timeit
from session_loader import find_max_time, find_min_time

def get_R2(y_test, y_test_pred):
    R2_list = []  # Initialize a list that will contain the R2s for all the outputs
    for i in range(y_test.shape[1]):  # Loop through outputs
        # Compute R2 for each output
        y_mean = np.mean(y_test[:, i])
        R2 = 1 - np.sum((y_test_pred[:, i] - y_test[:, i]) ** 2) / np.sum((y_test[:, i] - y_mean) ** 2)
        R2_list.append(R2)  # Append R2 of this output to the list
    R2_array = np.array(R2_list)
    return R2_array  # Return an array of R2s


def process_input(X_train):
    return_list = []
    for i in range(0, len(X_train[0])):
        neuron_list = np.asarray([])
        for j in range(0, len(X_train)):
            neuron_list = np.append(neuron_list,X_train[j][i])
        neuron_list = np.expand_dims(neuron_list,axis=1)
        return_list.append(neuron_list)
    return return_list


def process_output(y,bin_size, max_x, max_y, min_x,min_y):
    """
    processes the output position to fit the input size by averaging over the rats position in each time and normalizes

    :param y: list of tuples (position_x, position_y)
    :param bin_size: size of each time-bin
    :return: binned list
    """
    n_bin_points = len(y) // bin_size
    return_list = np.zeros((n_bin_points+1, 2))
    counter = np.zeros((n_bin_points+1,2))
    for i, current_pos in enumerate(y):
        index = i // bin_size
        if (index) > n_bin_points:
            print(i // bin_size)
        return_list[index] += current_pos
        counter[i // bin_size ] += 1
    return_list = return_list / counter

    # normalize
    return_list = return_list - [min_x,min_y]
    return_list = return_list / [max_x,max_y]
    return return_list



def get_model_data(train_slice,test_slice,bin_size, file_name=None,save_as=None):
    if file_name is not None:
        return load_pickle(file_name)
    train_slice.set_filter(filter=bin_filter, window=bin_size, step_size=1)
    test_slice.set_filter(filter=bin_filter, window=1, step_size=bin_size)
    # test_slice.plot_filtered_spikes(filter=bin_filter, window=1, step_size=bin_size, max_range=None)
    y_train = list(zip(train_slice.position_x, train_slice.position_y))
    y_valid = list(zip(test_slice.position_x, test_slice.position_y))

    X_train = train_slice.filtered_spikes
    X_valid = test_slice.filtered_spikes
    # Fit model

    X_train = process_input(X_train)
    X_valid = process_input(X_valid)
    X_train = np.asarray(X_train)
    X_valid = np.asarray(X_valid)
    max_x_train = np.amax(train_slice.position_x)
    max_x_valid = np.amax(test_slice.position_x)
    max_y_train = np.amax(train_slice.position_y)
    max_y_valid = np.amax(test_slice.position_y)
    min_x_train = np.amin(train_slice.position_x)
    min_x_valid = np.amin(test_slice.position_x)
    min_y_train = np.amin(train_slice.position_y)
    min_y_valid = np.amin(test_slice.position_y)
    y_train = process_output(np.asarray(y_train), bin_size=bin_size, max_x=max_x_train, min_x=min_x_train,
                             max_y=max_y_train, min_y=min_y_train)
    y_valid = process_output(np.asarray(y_valid), bin_size=bin_size, max_x=max_x_valid, min_x=min_x_valid,
                             max_y=max_y_valid, min_y=min_y_valid)
    model_data = dict(
        X_train=X_train,
        X_valid=X_valid,
        y_train=y_train,
        y_valid=y_valid
    )
    if save_as is not None:
        save_as_pickle(save_as, model_data)
    return model_data

def make_predictions(model_lstm,X_train,X_valid,y_train,y_valid):
    # Get predictions
    y_valid_predicted_lstm = model_lstm.predict(X_valid)
    y_overfitting_lstm = model_lstm.predict(X_train)
    # Get metric of fit
    R2s_lstm = get_R2(y_train, y_overfitting_lstm)
    print('R2s of training set:', R2s_lstm)
    R2s_lstm = get_R2(y_valid, y_valid_predicted_lstm)
    print('R2s of validation set:', R2s_lstm)
    pass

def test_trials(bin_size,units,epochs,dropout=None):
    data_slice = Slice.from_path(load_from="data/pickle/slice.pkl")
    trials = data_slice.get_all_trials()
    trials = trials.filter_trials_by_well(
        start_well=1, end_well=3)
    trials_1 = trials[0:int(len(trials.get_all())/2)]
    trials_2 = trials[int(len(trials.get_all()) / 2):]
    model_lstm = LSTMDecoder(units=units, dropout=dropout, num_epochs=epochs)
    for i,data_slice in enumerate(trials_1):
        size = len(data_slice.position_x)
        train_slice = data_slice
        test_slice = trials_2[i]

        start = timeit.default_timer()
        model_data = get_model_data(train_slice=train_slice,test_slice=test_slice,bin_size=bin_size)
        X_train = model_data["X_train"]
        X_valid = model_data["X_valid"]
        y_train = model_data["y_train"]
        y_valid = model_data["y_valid"]
        # Fit model
        model_lstm.fit(X_train, y_train)
    print("Trial test:")
    make_predictions(model_lstm=model_lstm,X_train=X_train,X_valid=X_valid,y_train=y_train,y_valid=y_valid)
    stop = timeit.default_timer()
    print("Run time: ", str(stop - start))
    print("")
    pass

def test_full_session(bin_size,units,epochs,dropout=None):
    data_slice = Slice.from_path(load_from="data/pickle/slice.pkl")
    size = len(data_slice.position_x)
    train_slice = data_slice[0:int(size / 2)]
    test_slice = data_slice[int(size / 2):]
    model_data = get_model_data(train_slice=train_slice, test_slice=test_slice, bin_size=bin_size)
    X_train = model_data["X_train"]
    X_valid = model_data["X_valid"]
    y_train = model_data["y_train"]
    y_valid = model_data["y_valid"]
    print(X_train.shape)
    model_lstm = LSTMDecoder(units=units, dropout=dropout, num_epochs=epochs)
    start = timeit.default_timer()
    model_lstm.fit(X_train, y_train)
    print("Session test:")
    make_predictions(model_lstm=model_lstm, X_train=X_train, X_valid=X_valid, y_train=y_train, y_valid=y_valid)
    stop = timeit.default_timer()
    print("Run time: ", str(stop-start))
    print("")
    pass

def test_phase(bin_size,units,epochs,dropout = None):
    data_slice = Slice.from_path(load_from="data/pickle/slice.pkl")
    data_slice = data_slice.get_all_phases()[3]
    size = len(data_slice.position_x)
    train_slice = data_slice[0:int(size / 2)]
    test_slice = data_slice[int(size / 2):]
    model_data = get_model_data(train_slice=train_slice, test_slice=test_slice, bin_size=bin_size)
    X_train = model_data["X_train"]
    X_valid = model_data["X_valid"]
    y_train = model_data["y_train"]
    y_valid = model_data["y_valid"]
    start = timeit.default_timer()
    model_lstm = LSTMDecoder(units=units, dropout=dropout, num_epochs=epochs)
    model_lstm.fit(X_train, y_train)
    print("Phases test:")
    make_predictions(model_lstm=model_lstm, X_train=X_train, X_valid=X_valid, y_train=y_train, y_valid=y_valid)
    stop = timeit.default_timer()
    print("Run time: ", str(stop-start))
    print("")
    pass