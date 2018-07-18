import numpy as np
from src.settings import load_pickle,save_as_pickle
from src.filters import bin_filter
from database_api_beta import Slice
from tmp.estimators import cnn_model_fn
import tensorflow as tf



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


def get_next_lickwell_list(data_slice):
    y_list = np.zeros((len(data_slice.position_x),))
    k = 0
    current_lickwell = data_slice.licks[k]["lickwell"]
    current_time = data_slice.licks[k]["time"]

    for i in range(data_slice.start_time,data_slice.start_time + len(data_slice.position_x)):
        if current_time<i:
            if k == len(data_slice.licks): # last section doesnt have any lick data
                current_lickwell = 0
                current_time = np.inf
            else:
                current_lickwell = data_slice.licks[k]["lickwell"]
                current_time = data_slice.licks[k]["time"]
                k = k + 1
        y_list[i-data_slice.start_time] = current_lickwell
    return_array = np.array([])
    bin_size = int(len(data_slice.position_x) / len(data_slice.filtered_spikes))
    for bin_no in range(len(data_slice.filtered_spikes)):
        return_array = np.append(return_array,y_list[bin_size*bin_no])
    return return_array

def get_model_data(data_slice, search_window_size, step_size, load_from=None, save_as=None):
    #get raw data
    if load_from is not None:
        return load_pickle(load_from)
    size = len(data_slice.position_x)
    train_range = slice(0,int(size / 2))
    valid_range = slice(int(size / 2),size)
    train_slice = data_slice[train_range]
    valid_slice = data_slice[valid_range]
    train_slice.set_filter(filter=bin_filter, search_window_size=search_window_size, step_size= step_size, num_threads=20)
    valid_slice.set_filter(filter=bin_filter, search_window_size=search_window_size, step_size= step_size, num_threads=20)

    X_train = np.expand_dims(train_slice.filtered_spikes.T,axis=2)
    X_valid = np.expand_dims(valid_slice.filtered_spikes.T,axis=2)
    y_train = get_next_lickwell_list(train_slice)
    y_valid = get_next_lickwell_list(valid_slice)

    model_data = dict(
        X_train=X_train,
        X_valid=X_valid,
        y_train=y_train,
        y_valid=y_valid
    )

    if save_as is not None:
        save_as_pickle(save_as, model_data)
    return model_data


# Network Parameters

data_slice = Slice.from_path(load_from="slice.pkl")
search_window_size = 700
step_size = 700

# Load model data

model_data = get_model_data(data_slice, search_window_size, step_size, load_from="test_network_18-7-18.pkl", save_as=None)
train_data = model_data["X_train"]
train_labels = model_data["y_train"]
valid_data = model_data["X_valid"]
valid_labels = model_data["y_valid"]

# Create Estimator

network_classifier = tf.estimator.Estimator(model_fn=cnn_model_fn,model_dir="test_network_18-7_model")

# TODO logging hook
print("fin")

