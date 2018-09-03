import numpy as np
from src.settings import load_pickle, save_as_pickle
from src.filters import bin_filter
from database_api_beta import Slice
from tmp.lickwell_position_cnn import lickwell_position_model_fn
import tensorflow as tf
import random
from src.settings import config
import os
from math import isclose

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
# tf.logging.set_verbosity(tf.logging.INFO)


def find_first(vec,item,index=None):
    if index is not None:
        vec = vec[:,index]
    for i in range(len(vec)):
        if item == vec[i]:
            return i
    return None

def get_data_slices(data_slice, time_range,  search_window_size, step_size,train_validation_ratio ,
                             load_from=None, normalize_stations= True,save_as=None,eval_set_minimum_size=5):
    """
    Returns list of slices in the time_range around the licks
    """
    if load_from is not None:
        return load_pickle(load_from)

    # Slice d'ata_slice into even parts

    data_slice_list,time_slice_list, lickwell_list, position_list, speed_list, bin_list = [], [], [], [], [],[]
    data_slice_c = data_slice
    c = 0
    abs_tol = 5
    while c < len(data_slice_c.position_x):
        current_lickwell = 0
        current_position = data_slice_c.position_x[c]
        if isclose(current_position,30,abs_tol=abs_tol):
            current_lickwell = 1
        if isclose(current_position,67,abs_tol=abs_tol):
            current_lickwell = 2
        if isclose(current_position,113,abs_tol=abs_tol):
            current_lickwell = 3
        if isclose(current_position,161,abs_tol=abs_tol):
            current_lickwell = 4
        if isclose(current_position,203,abs_tol=abs_tol):
            current_lickwell = 5

        if current_lickwell != 0:
            if c + time_range < len(data_slice_c.position_x): # exclude last slice if its size is smaller than one time_range
                lickwell_list.append(current_lickwell)
                data_slice_list.append(data_slice_c[c:c+time_range])
            c = c + time_range
        else:
            c = c + 1
    # Get relevant data from slices for input/output
    _,occurrence_counter = np.unique(lickwell_list, return_counts=True)
    for data_bit in data_slice_list:
        print("processing data_bit ",data_bit.start_time)
        time_of_lick = data_bit.start_time

        position_list.append(data_slice.position_x[time_of_lick])
        speed_list.append(data_slice.speed[time_of_lick])
        data_bit.set_filter(filter=bin_filter, search_window_size=search_window_size, step_size=step_size,
                            num_threads=1)
        bin_list.append(data_bit.filtered_spikes)



    shuffle_list = list(zip(bin_list, lickwell_list, position_list, speed_list))
    random.shuffle(shuffle_list)

    if normalize_stations is True:

        # Set up list

        shuffle_list = np.asarray([shuffle_list[i] for j in config["setup"]["lickwell_list"] for i, e in enumerate(shuffle_list) if
                        shuffle_list[i][1] == j])

        # Remove surplus entries from ordered list
        minimum_occurence = np.min(occurrence_counter)
        minimum_occurence= minimum_occurence - eval_set_minimum_size
        eval_list = []

        # Remove surplus samples from training set and add them to testing set

        for lickwell in config["setup"]["lickwell_list"]:
            max_val = int(occurrence_counter[lickwell - 1] - minimum_occurence)
            for j in range(0,max_val):

                # Find index
                item_index = find_first(shuffle_list,lickwell,1)
                if item_index is not None:
                    eval_list.append(shuffle_list[item_index])
                    shuffle_list = np.delete(shuffle_list,item_index,axis=0)
                else:
                    break
    shuffle_list = np.ndarray.tolist(shuffle_list)
    random.shuffle(shuffle_list)

    bin_list, lickwell_list, position_list, speed_list = zip(*shuffle_list)
    if train_validation_ratio is not None: # random assignment
        train_validation_index = int(len(bin_list) * train_validation_ratio)
        return_dict = dict(
            X_train=np.float32(np.expand_dims(bin_list[:train_validation_index], axis=3)),
            # flattens neuron activity in bins
            X_valid=np.float32(np.expand_dims(bin_list[train_validation_index:], axis=3)),
            y_licks_train=np.int32(lickwell_list[:train_validation_index]),
            y_licks_valid=np.int32(lickwell_list[train_validation_index:]),
            y_position_train=position_list[:train_validation_index],
            y_position_valid=position_list[train_validation_index:],
            y_speed_train=speed_list[:train_validation_index],
            y_speed_valid=speed_list[train_validation_index:],
        )
    else: # surplus is used as evaluation set
        bin_list_valid, lickwell_list_valid, position_list_valid, speed_list_valid = zip(*eval_list)

        return_dict = dict(
            X_train=np.float32(np.expand_dims(bin_list, axis=3)),
            # flattens neuron activity in bins
            X_valid=np.float32(np.expand_dims(bin_list_valid, axis=3)),
            y_licks_train=np.int32(lickwell_list),
            y_licks_valid=np.int32(lickwell_list_valid),
            y_position_train=position_list,
            y_position_valid=position_list_valid,
            y_speed_train=speed_list,
            y_speed_valid=speed_list_valid,
        )

    if save_as is not None:
        save_as_pickle(save_as, return_dict)
    return return_dict



data_slice = Slice.from_path(load_from="slice.pkl")
# data_slice.neuron_filter(300)
search_window_size = 20
step_size = 40
number_of_training_steps=25000
train_validation_ratio=None
time_range = 1000
load = False
train = True
if load is True:
    lick_slices = get_data_slices(data_slice=data_slice, time_range=time_range, train_validation_ratio=train_validation_ratio, search_window_size=search_window_size, step_size=step_size,
                            load_from="lick_slices_1.pkl")
else:
    lick_slices = get_data_slices(data_slice=data_slice, time_range=time_range, train_validation_ratio=train_validation_ratio, search_window_size=search_window_size, step_size=step_size,
                                save_as="lick_slices_1.pkl")

X_train = (lick_slices["X_train"])
X_valid = lick_slices["X_valid"]
y_train = lick_slices["y_licks_train"]
y_valid = lick_slices["y_licks_valid"]

#  Create Estimator
print(np.unique(y_train, return_counts=True))

network_classifier = tf.estimator.Estimator(model_fn=lickwell_position_model_fn,model_dir="lickwell_position_cnn_30-07-18_22")

# Create logging hook

tensors_to_log = {"probabilities": "softmax_tensor"}
logging_hook = tf.train.LoggingTensorHook(tensors=tensors_to_log,every_n_iter=50)


# Train model

if train is True:
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
    batch_size=1,
    shuffle=False)


eval_results = network_classifier.evaluate(input_fn=eval_input_fn,hooks=[eval_hook])
print("Training set: ",eval_results)

eval_input_fn = tf.estimator.inputs.numpy_input_fn(
    x={"x":X_valid},
    y=y_valid,
    num_epochs=1,
    batch_size = 1,
    shuffle=False)

eval_results = network_classifier.evaluate(input_fn=eval_input_fn,hooks=[eval_hook])
print("Evaluation set: ",eval_results)

run_opts = tf.RunOptions(report_tensor_allocations_upon_oom = True)
for i in range(0,6):

    X_single_lickwell = np.float32([e for j, e in enumerate(X_valid) if y_valid[j] == i])
    y_single_lickwell = np.int32([e for j, e in enumerate(y_valid) if y_valid[j] == i])

    if len(X_single_lickwell)>0:
        eval_input_fn = tf.estimator.inputs.numpy_input_fn(
            x={"x": X_single_lickwell},
            y=y_single_lickwell,
            num_epochs=1,
            batch_size=1,
            shuffle=False)
        eval_results = network_classifier.evaluate(input_fn=eval_input_fn,hooks=[eval_hook])
        print("lickwell", i, ": sample size: ", len(y_single_lickwell))
        print("results: ", eval_results)

    # sess = tf.InteractiveSession()
    # with sess.as_default():
    #     tf.initialize_all_variables().run()
    #     print("logits:",logits.eval())
    #     print("asd")
print("fin")