import numpy as np
from src.settings import load_pickle, save_as_pickle
from src.filters import bin_filter
from src.database_api_beta import Slice
from tmp.position_cnn import cnn_model_fn
import tensorflow as tf
import random
import matplotlib.pyplot as plt
import os

# Setup network parameters

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
tf.logging.set_verbosity(tf.logging.INFO)

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
                            num_threads=1)
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
        y_position_train=np.float32(position_list[:train_validation_index]),
        y_position_valid=np.float32(position_list[train_validation_index:]),
        y_speed_train=speed_list[:train_validation_index],
        y_speed_valid=speed_list[train_validation_index:],
    )

    if save_as is not None:
        save_as_pickle(save_as, return_dict)
    return return_dict



data_slice = Slice.from_path(load_from="hippocampus_session.pkl")
# data_slice.neuron_filter(300)
search_window_size = 10
step_size = 20
number_of_training_steps=20000
train_validation_ratio=0.8
time_range = 2000
load = True
if load is True:
    lick_slices = get_data_slices(data_slice=data_slice, time_range=time_range, train_validation_ratio=train_validation_ratio, search_window_size=search_window_size, step_size=step_size,
                            load_from="lick_slices.pkl")
else:
    lick_slices = get_data_slices(data_slice=data_slice, time_range=time_range, train_validation_ratio=train_validation_ratio, search_window_size=search_window_size, step_size=step_size,
                                save_as="lick_slices.pkl")

X_train = (lick_slices["X_train"])
X_valid = lick_slices["X_valid"]
y_train = lick_slices["y_position_train"]
y_valid = lick_slices["y_position_valid"]

# Create Estimator

network_classifier = tf.estimator.Estimator(model_fn=cnn_model_fn,model_dir="/tmp/position_cnn_26-07-18_1")

# Create logging hook

tensors_to_log = {"distance":"logits"}
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

# eval_results = network_classifier.evaluate(input_fn=eval_input_fn,hooks=[eval_hook])
predict_results = network_classifier.predict(input_fn=eval_input_fn)
# print("Evaluation set: ",eval_results)


predicted_classes = [p["classes"] for p in predict_results]
x  = np.arange(0,len(predicted_classes))
plt.bar(x, predicted_classes, alpha=0.7)
plt.bar(x, y_valid, alpha=0.7)
plt.legend(['predicted position', 'actual position'], loc='upper left')
plt.show()
# print(
#     "New Samples, Class Predictions:    {}\n"
#     .format(predicted_classes))



print("fin")