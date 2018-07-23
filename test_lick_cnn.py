import numpy as np
from src.settings import load_pickle, save_as_pickle
from src.filters import bin_filter
from database_api_beta import Slice
from tmp.estimators import cnn_model_fn
import tensorflow as tf
import random
from src.settings import config
tf.logging.set_verbosity(tf.logging.INFO)

def get_slices_around_licks(data_slice, time_range, train_validation_ratio, search_window_size, step_size,
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

    time_slice_list, lickwell_list, position_list, speed_list,bin_list = [],[],[],[],[]
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

        shuffle_list = [shuffle_list[i] for j in config["setup"]["lickwell_list"] for i, e in enumerate(shuffle_list) if shuffle_list[i][1] == j]
        occurrence_counter = np.zeros(len(config["setup"]["lickwell_list"]))

        # Determine minimum amount of occurrences
        for le in (shuffle_list):
            occurrence_counter[le[1]-1] = occurrence_counter[le[1]-1] + 1
        minimum_occurrence = np.inf
        for k in occurrence_counter:
            if minimum_occurrence> k and k != 0:
                minimum_occurrence = k

        # Remove surplus entries from ordered list

        for lickwell in config["setup"]["lickwell_list"]:
            # Remove surplus
            for j in range(0, int(occurrence_counter[lickwell-1] - minimum_occurrence)):
                # Find index
                item_index = next((i for i, v in enumerate(shuffle_list) if lickwell == v[1]), None) # Amount of licks add up to short runtimes
                if item_index is not None:
                    del(shuffle_list[item_index])

    # Randomly assign list elements to training or validation

    random.shuffle(shuffle_list)

    bin_list, lickwell_list, position_list, speed_list = zip(*shuffle_list)
    train_validation_index = int(len(bin_list)* train_validation_ratio)
    return_dict = dict(
        X_train=np.float32(np.expand_dims(bin_list[:train_validation_index],axis=3)), # flattens neuron activity in bins
        X_valid=np.float32(np.expand_dims(bin_list[train_validation_index:],axis=3)),
        y_licks_train=np.int32(lickwell_list[:train_validation_index]),
        y_licks_valid=np.int32(lickwell_list[train_validation_index:]),
        y_position_train=position_list[:train_validation_index],
        y_position_valid=position_list[train_validation_index:],
        y_speed_train=speed_list[:train_validation_index],
        y_speed_valid=speed_list[train_validation_index:],
    )
    if save_as is not None:
        save_as_pickle(save_as, return_dict)
    return return_dict

# Network Parameters

data_slice = Slice.from_path(load_from="slice.pkl")
# data_slice.neuron_filter(300)
search_window_size = 50
step_size = 100

lick_slices = get_slices_around_licks(data_slice=data_slice, time_range=1000, train_validation_ratio=0.5, search_window_size=search_window_size, step_size=step_size,
                            include_unsuccessful_licks=True,  normalize_stations=True,load_from="lick_slices.pkl")


X_train = (lick_slices["X_train"])
X_valid = lick_slices["X_valid"]
y_train = lick_slices["y_licks_train"]
y_valid = lick_slices["y_licks_valid"]

# Create Estimator

network_classifier = tf.estimator.Estimator(model_fn=cnn_model_fn,model_dir="lick_cnn_23-07-18_15")

# Create logging hook

tensors_to_log = {"probabilities": "softmax_tensor"}
logging_hook = tf.train.LoggingTensorHook(tensors=tensors_to_log,every_n_iter=50)


# Train model
#
# train_input_fn = tf.estimator.inputs.numpy_input_fn(
#     x={"x": X_train},
#     y=y_train,
#     batch_size=1,
#     num_epochs=None,
#     shuffle=True)
#
# network_classifier.train(
#     input_fn=train_input_fn,
#     steps=7000,
#     hooks=[logging_hook]
# )


eval_input_fn = tf.estimator.inputs.numpy_input_fn(
    x={"x":X_train},
    y=y_train,
    num_epochs=1,
    shuffle=False)


eval_results = network_classifier.evaluate(input_fn=eval_input_fn)
print("Training set: ",eval_results)

eval_input_fn = tf.estimator.inputs.numpy_input_fn(
    x={"x":X_valid},
    y=y_valid,
    num_epochs=1,
    shuffle=False)

eval_results = network_classifier.evaluate(input_fn=eval_input_fn)
print("Evaluation set: ",eval_results)


for i in range(0,6):

    eval_data_1 = [e for j, e in enumerate(X_valid) if y_valid[j] == i]
    eval_labels_1 = [e for j, e in enumerate(y_valid) if y_valid[j] == i]
    eval_data_1 = np.float32(np.squeeze(eval_data_1,axis=2))
    eval_labels_1 = np.int32(eval_labels_1)
    eval_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={"x": eval_data_1},
        y=eval_labels_1,
        num_epochs=1,
        shuffle=False)

    if len(eval_data_1)>0:
        eval_results = network_classifier.evaluate(input_fn=eval_input_fn)
        print("lickwell", i, ": sample size: ", len(eval_data_1))
        print("results: ", eval_results)


print("fin")