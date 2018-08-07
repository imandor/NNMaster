import numpy as np
from database_api_beta import Slice
from tmp.lickwell_position_cnn import lickwell_position_model_fn
import tensorflow as tf
import os
from tmp.test_lickwell_position_cnn_beta import get_data_slices
from tmp.lickwell_position_cnn_beta import cnn_model,hidden_layer_output

# Setup network parameters

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
tf.logging.set_verbosity(tf.logging.INFO)

data_slice = Slice.from_path(load_from="slice.pkl")
model_filename = "lickwell_position_cnn_beta_01-08-18_3.h5"
model = lickwell_position_model_fn
getter_function = get_data_slices

# data_slice.neuron_filter(300)
search_window_size = 20
step_size = 40
epochs=100
train_validation_ratio=None
time_range = 1000
load = True
train = True
load_model = False

# Load input and labels

if load is True:
    lick_slices = getter_function(data_slice=data_slice, time_range=time_range, train_validation_ratio=train_validation_ratio, search_window_size=search_window_size, step_size=step_size,
                            load_from="lick_slices_1.pkl")
else:
    lick_slices = getter_function(data_slice=data_slice, time_range=time_range, train_validation_ratio=train_validation_ratio, search_window_size=search_window_size, step_size=step_size,
                            save_as="lick_slices_1.pkl")

X_train = (lick_slices["X_train"])
X_valid = lick_slices["X_valid"]
y_train = lick_slices["y_licks_train"]
y_valid = lick_slices["y_licks_valid"]

# Load model

if load_model is True:
    model= cnn_model(shape=X_train.shape, model_filename=model_filename,mode=tf.estimator.ModeKeys.TRAIN)
else:
    model= cnn_model(shape=X_train.shape, model_filename=None,mode=tf.estimator.ModeKeys.TRAIN)



if train is True:
    print("Training model...")
    model.fit(X_train, y_train, epochs=epochs)
    print("Finished training")
    model.save(model_filename)
# Evaluate training set
print("Training set:")
results = model.evaluate(X_train, y_train)
print("loss: ", results[0], ", accuracy: ", results[1])
output = hidden_layer_output(model, X_train)
print(output)


# Evaluate validation set
print("Validation set:")
results = model.evaluate(X_valid, y_valid)
print("loss: ", results[0], ", accuracy: ", results[1])

# Evaluate validation set by well
for i in range(0,6):

    X_single_lickwell = np.float32([e for j, e in enumerate(X_valid) if y_valid[j] == i])
    y_single_lickwell = np.int32([e for j, e in enumerate(y_valid) if y_valid[j] == i])

    if len(X_single_lickwell)>0:
        eval_input_fn = tf.estimator.inputs.numpy_input_fn(
            x={"x": X_single_lickwell},
            y=y_single_lickwell,
            num_epochs=1,
            shuffle=False)
        print("")
        print("")
        print("lickwell", i, ": sample size: ", len(y_single_lickwell))
        results = model.evaluate(X_single_lickwell, y_single_lickwell)
        print("loss: ", results[0], ", accuracy: ", results[1])
        print("hidden layer:")
        print(hidden_layer_output(model, X_single_lickwell))


print("fin")