import numpy as np
from src.settings import load_pickle, save_as_pickle
from src.filters import bin_filter
from database_api_beta import Slice
from networks.position_cnn import cnn_model_fn
import tensorflow as tf
import random
from src.settings import config
from networks.lick_cnn import lick_cnn_model
import os


def get_data_slices(data_slice, time_range,  search_window_size, step_size,train_validation_ratio ,
                             load_from=None, save_as=None):
    """
    Returns list of slices in the time_range around the licks
    """
    if load_from is not None:
        return load_pickle(load_from)

    # Slice data_slice into even parts

    data_slice_list,time_slice_list, lickwell_list, position_list, speed_list, bin_list = [], [], [], [], [],[]
    data_slice_c = data_slice
    while len(data_slice_c.position_x)>time_range:
        data_slice_list.append(data_slice_c[0:time_range])
        data_slice_c = data_slice_c[time_range:]

    # Get relevant data from slices for input/output

    for data_bit in data_slice_list:
        print("processing data_bit ",data_bit.start_time)
        time_of_lick = data_bit.start_time
        lickwell_list.append(0)
        position_list.append(data_slice.position_x[time_of_lick])
        speed_list.append(data_slice.speed[time_of_lick])
        data_bit.set_filter(filter=bin_filter, search_window_size=search_window_size, step_size=step_size,
                            num_threads=20)
        bin_list.append(data_bit.filtered_spikes)

    shuffle_list = list(zip(bin_list, lickwell_list, position_list, speed_list))

    # Assign list elements to training or validation

    random.shuffle(shuffle_list)

    bin_list, lickwell_list, position_list, speed_list = zip(*shuffle_list)

    train_validation_index = int(len(bin_list) * train_validation_ratio)
    return_dict = dict(
        X_train=np.float32(np.expand_dims(bin_list[:train_validation_index], axis=3)),
        # flattens neuron activity in bins
        X_valid=np.float32(np.expand_dims(bin_list[train_validation_index:], axis=3)),
        y_licks_train=np.int32(lickwell_list[:train_validation_index]),
        y_licks_valid=np.int32(lickwell_list[train_validation_index:]),
        y_position_train=np.int32(position_list[:train_validation_index]),
        y_position_valid=np.int32(position_list[train_validation_index:]),
        y_speed_train=speed_list[:train_validation_index],
        y_speed_valid=speed_list[train_validation_index:],
    )

    if save_as is not None:
        save_as_pickle(save_as, return_dict)
    return return_dict



data_slice = Slice.from_path(load_from="slice.pkl")[0:10000]
# data_slice.neuron_filter(300)
search_window_size = 25
step_size = 50
number_of_training_steps=20000
train_validation_ratio=0.8
time_range = 500
# lick_slices = get_slices_around_licks(data_slice=data_slice, time_range=1000, train_validation_ratio=train_validation_ratio, search_window_size=search_window_size, step_size=step_size,
#                             include_unsuccessful_licks=True,  normalize_stations=True,save_as="lick_slices.pkl")

lick_slices = get_data_slices(data_slice=data_slice, time_range=1000, train_validation_ratio=train_validation_ratio, search_window_size=search_window_size, step_size=step_size,
                            save_as="lick_slices.pkl")

X_train = (lick_slices["X_train"])
X_valid = lick_slices["X_valid"]
y_train = lick_slices["y_position_train"]
y_valid = lick_slices["y_position_valid"]

# Create Estimator

network_classifier = tf.estimator.Estimator(model_fn=cnn_model_fn,model_dir="position_cnn_24-07-18_4")

# Create logging hook

tensors_to_log = {}
logging_hook = tf.train.LoggingTensorHook(tensors=tensors_to_log,every_n_iter=50)


# Train model

train_input_fn = tf.estimator.inputs.numpy_input_fn(
    x={"x": X_train},
    y=y_train,
    batch_size=1,
    num_epochs=None,
    shuffle=True)

network_classifier.train(
    input_fn=train_input_fn,
    steps=number_of_training_steps,
    hooks=[logging_hook]
)

eval_hook = tf.train.LoggingTensorHook(tensors=tensors_to_log,every_n_iter=1)
eval_input_fn = tf.estimator.inputs.numpy_input_fn(
    x={"x":X_train},
    y=y_train,
    num_epochs=1,
    shuffle=False)


eval_results = network_classifier.evaluate(input_fn=eval_input_fn,hooks=[eval_hook])
print("Training set: ",eval_results)

eval_input_fn = tf.estimator.inputs.numpy_input_fn(
    x={"x":X_valid},
    y=y_valid,
    num_epochs=1,
    shuffle=False)

eval_results = network_classifier.evaluate(input_fn=eval_input_fn,hooks=[eval_hook])
print("Evaluation set: ",eval_results)


for i in range(0,6):

    X_single_lickwell = np.float32([e for j, e in enumerate(X_valid) if y_valid[j] == i])
    y_single_lickwell = np.int32([e for j, e in enumerate(y_valid) if y_valid[j] == i])

    if len(X_single_lickwell)>0:
        eval_input_fn = tf.estimator.inputs.numpy_input_fn(
            x={"x": X_single_lickwell},
            y=y_single_lickwell,
            num_epochs=1,
            shuffle=False)
        eval_results = network_classifier.evaluate(input_fn=eval_input_fn,hooks=[eval_hook])
        print("lickwell", i, ": sample size: ", len(y_single_lickwell))
        print("results: ", eval_results)


print("fin")