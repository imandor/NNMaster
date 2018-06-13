import numpy as np
from settings import load_pickle,save_as_pickle



def get_model_data(train_slice, test_slice, bin_size, step_size, load_from=None, save_as=None):
    if load_from is not None:
        return load_pickle(load_from)
    train_slice.set_filter(filter=bin_filter, search_window_size=bin_size, step_size=step_size)
    test_slice.set_filter(filter=bin_filter, search_window_size=bin_size, step_size=step_size)
    y_train = list(zip(train_slice.position_x, train_slice.position_y))
    y_valid = list(zip(test_slice.position_x, test_slice.position_y))

    X_train = train_slice.filtered_spikes
    X_valid = test_slice.filtered_spikes
    # Fit model

    X_train = np.expand_dims(X_train.T,axis=2)
    X_valid = np.expand_dims(X_valid.T,axis=2)
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


def run_network_test(data_slice):
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
    X_train =
    X_valid =
    y_train =
    y_valid =


    pass
