import numpy as np
from src.settings import load_pickle, save_as_pickle
from src.filters import bin_filter
from src.database_api_beta import Slice
from tmp.estimators import cnn_model_fn
import tensorflow as tf
import random
from src.settings import config
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
tf.logging.set_verbosity(tf.logging.INFO)








def get_slices_around_licks(data_slice, time_range,  search_window_size, step_size,train_validation_ratio = None,
                            include_unsuccessful_licks=True, normalize_stations=True, load_from=None, save_as=None):
    """
    Returns list of slices in the time_range around the licks
    """
    if load_from is not None:
        return load_pickle(load_from)

    # Find time slice corresponding to licks and generate feature lists

    licks = data_slice.licks

    if licks == []:
        raise IndexError("Error: No licks in data_slice")

    time_slice_list, lickwell_list, position_list, speed_list, bin_list = [], [], [], [], []
    for lick in licks:
        if include_unsuccessful_licks or lick["rewarded"] is True:
            time_of_lick = int(lick["time"])
            lickwell_list.append(int(lick["lickwell"]))
            position_list.append(data_slice.position_x[time_of_lick])
            speed_list.append(data_slice.speed[time_of_lick])

            data_bit = data_slice[slice(time_of_lick - time_range, time_of_lick + time_range)]
            data_bit.set_filter(filter=bin_filter, search_window_size=search_window_size, step_size=step_size,
                                num_threads=20)
            bin_list.append(data_bit.filtered_spikes)

    shuffle_list = list(zip(bin_list, lickwell_list, position_list, speed_list))

    # Normalize the amount of stations in training and testing set

    if normalize_stations is True:

        # Set up list

        shuffle_list = [shuffle_list[i] for j in config["setup"]["lickwell_list"] for i, e in enumerate(shuffle_list) if
                        shuffle_list[i][1] == j]
        occurrence_counter = np.zeros(len(config["setup"]["lickwell_list"]))

        # Determine minimum amount of occurrences

        for le in (shuffle_list):
            occurrence_counter[le[1] - 1] = occurrence_counter[le[1] - 1] + 1
        minimum_occurrence = np.inf
        for k in occurrence_counter:
            if minimum_occurrence > k and k != 0:
                minimum_occurrence = k

        # Remove surplus entries from ordered list

        eval_list = []
        for lickwell in config["setup"]["lickwell_list"]:
            # Remove surplus
            for j in range(0, int(occurrence_counter[lickwell - 1] - minimum_occurrence)):
                # Find index
                item_index = next((i for i, v in enumerate(shuffle_list) if lickwell == v[1]),
                                  None)  # Amount of licks add up to short runtimes
                if item_index is not None:
                    eval_list.append(shuffle_list[item_index])
                    del (shuffle_list[item_index])

    # Assign list elements to training or validation

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


# Network Parameters

data_slice = Slice.from_path(load_from="slice.pkl")
# data_slice.neuron_filter(300)
search_window_size = 25
step_size = 50
number_of_training_steps=200
train_validation_ratio=None
# lick_slices = get_slices_around_licks(data_slice=data_slice, time_range=1000, train_validation_ratio=train_validation_ratio, search_window_size=search_window_size, step_size=step_size,
#                             include_unsuccessful_licks=True,  normalize_stations=True,save_as="lick_slices.pkl")

lick_slices = get_slices_around_licks(data_slice=data_slice, time_range=1000, train_validation_ratio=train_validation_ratio, search_window_size=search_window_size, step_size=step_size,
                            include_unsuccessful_licks=True,  normalize_stations=True,load_from="lick_slices.pkl")

X_train = (lick_slices["X_train"])
X_valid = lick_slices["X_valid"]
y_train = lick_slices["y_licks_train"]
y_valid = lick_slices["y_licks_valid"]

# Create Estimator

network_classifier = tf.estimator.Estimator(model_fn=cnn_model_fn,model_dir="lick_cnn_24-07-18_6")

# Create logging hook

tensors_to_log = {"probabilities": "softmax_tensor"}
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

    # sess = tf.InteractiveSession()
    # with sess.as_default():
    #     tf.initialize_all_variables().run()
    #     print("logits:",logits.eval())
    #     print("asd")
print("fin")