import numpy as np
from src.settings import load_pickle, save_as_pickle
from src.filters import bin_filter
from database_api_beta import Slice
from networks.lickwell_position_cnn import lickwell_position_model_fn
import tensorflow as tf
import random
import matplotlib.pyplot as plt
from src.settings import config
import os
from math import isclose
from test_lickwell_position_cnn import get_data_slices


# Setup network parameters

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
tf.logging.set_verbosity(tf.logging.INFO)

data_slice = Slice.from_path(load_from="slice.pkl")

model = lickwell_position_model_fn
getter_function = get_data_slices

# data_slice.neuron_filter(300)
search_window_size = 50
step_size = 100
number_of_training_steps=20000
train_validation_ratio=None
time_range = 1500
load = False
train = True


# Load input and labels

if load is True:
    lick_slices = getter_function(data_slice=data_slice, time_range=time_range, train_validation_ratio=train_validation_ratio, search_window_size=search_window_size, step_size=step_size,
                            load_from="lick_slices.pkl")
else:
    lick_slices = getter_function(data_slice=data_slice, time_range=time_range, train_validation_ratio=train_validation_ratio, search_window_size=search_window_size, step_size=step_size,
                                save_as="lick_slices.pkl")

X_train = (lick_slices["X_train"])
X_valid = lick_slices["X_valid"]
y_train = lick_slices["y_licks_train"]
y_valid = lick_slices["y_licks_valid"]

#  Create Estimator

network_classifier = tf.estimator.Estimator(model_fn=model,model_dir="lickwell_position_cnn_24-07-18_10")

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

# Evaluate training set

eval_input_fn = tf.estimator.inputs.numpy_input_fn(
    x={"x":X_train},
    y=y_train,
    num_epochs=1,
    shuffle=False)


eval_results = network_classifier.evaluate(input_fn=eval_input_fn,hooks=[eval_hook])
print("Training set: ",eval_results)

# Evaluate testing set

eval_input_fn = tf.estimator.inputs.numpy_input_fn(
    x={"x":X_valid},
    y=y_valid,
    num_epochs=1,
    shuffle=False)

eval_results = network_classifier.evaluate(input_fn=eval_input_fn,hooks=[eval_hook])
print("Evaluation set: ",eval_results)

# Evaluate testing set while filtering for each well

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