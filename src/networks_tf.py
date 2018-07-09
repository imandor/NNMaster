import numpy as np
from src.settings import load_pickle,save_as_pickle
from src.filters import bin_filter
from src.multi_processing import bin_slices_spikes






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


def get_model_data(train_slice, test_slice, search_window_size, step_size, load_from=None, save_as=None):
    #get raw data
    if load_from is not None:
        return load_pickle(load_from)
    train_slice = bin_slices_spikes(train_slice, search_window_size=search_window_size, step_size=step_size, num_threads=20)
    test_slice = bin_slices_spikes(test_slice, search_window_size=search_window_size, step_size=step_size, num_threads=20)

    X_train = np.expand_dims(train_slice.filtered_spikes.T,axis=2)
    X_valid = np.expand_dims(test_slice.filtered_spikes.T,axis=2)
    max_x_train = np.amax(train_slice.position_x)
    max_x_valid = np.amax(test_slice.position_x)
    max_y_train = np.amax(train_slice.position_y)
    max_y_valid = np.amax(test_slice.position_y)
    min_x_train = np.amin(train_slice.position_x)
    min_x_valid = np.amin(test_slice.position_x)
    min_y_train = np.amin(train_slice.position_y)
    min_y_valid = np.amin(test_slice.position_y)

    model_data = dict(
        X_train=X_train,
        X_valid=X_valid,
        y_train=y_train,
        y_valid=y_valid
    )

    if save_as is not None:
        save_as_pickle(save_as, model_data)
    return model_data


def run_network_test(data_slice,bin_size,step_size):
    """

    :param bin_size:
    :param step_size:
    :param units:
    :param epochs:
    :param dropout:
    :return:
    """
    """
    runs a sample network in tensorflow
    
    :return: 
    """
    size = len(data_slice.position_x)
    train_slice = data_slice[0:int(size / 2)]
    test_slice = data_slice[int(size / 2):]
    model_data = get_model_data(train_slice=train_slice, test_slice=test_slice, bin_size=bin_size,step_size=step_size)
    X_train = model_data["X_train"]
    X_valid = model_data["X_valid"]
    y_train = model_data["y_train"]
    y_valid = model_data["y_valid"]
    # X_train =
    # X_valid =
    # y_train =
    # y_valid =


    pass
