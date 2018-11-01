import numpy as np
from src.settings import load_pickle,save_as_pickle
from src.filters import bin_filter
from src.database_api_beta import Slice
from tmp.estimators import cnn_model_fn
import tensorflow as tf

tf.logging.set_verbosity(tf.logging.INFO)



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
    bin_size = int(len(data_slice.position_x) / len(data_slice.filtered_spikes[0]))
    for bin_no in range(len(data_slice.filtered_spikes[0])):
        return_array = np.append(return_array,y_list[bin_size*bin_no])
    return return_array

def get_model_data(data_slice, search_window_size, step_size, load_from=None, save_as=None):
    #get raw data
    if load_from is not None:
        return load_pickle(load_from)
    size = len(data_slice.position_x)
    testing_ratio =  int(size - size/2)

    train_range = slice(0,testing_ratio)
    eval_range = slice(testing_ratio,size)

    # # Test: shuffle data
    # random.shuffle(data_slice)

    train_slice = data_slice[train_range]
    eval_slice = data_slice[eval_range]
    train_slice.set_filter(filter=bin_filter, search_window_size=search_window_size, step_size= step_size, num_threads=20)
    eval_slice.set_filter(filter=bin_filter, search_window_size=search_window_size, step_size= step_size, num_threads=20)

    train_data = np.expand_dims(train_slice.filtered_spikes.T,axis=2)
    eval_data = np.expand_dims(eval_slice.filtered_spikes.T,axis=2)
    train_labels = get_next_lickwell_list(train_slice)
    eval_labels = get_next_lickwell_list(eval_slice)
    eval_data = [e for i, e in enumerate(eval_data) if eval_labels[i] != 0]
    eval_labels = [e for i, e in enumerate(eval_labels) if eval_labels[i] != 0]


# Test: remove well 1
    eval_data = [e for i, e in enumerate(eval_data) if eval_labels[i] != 1]
    eval_labels = [e for i, e in enumerate(eval_labels) if eval_labels[i] != 1]
    train_data = [e for i, e in enumerate(train_data) if train_labels[i] != 1]
    train_labels = [e for i, e in enumerate(train_labels) if train_labels[i] != 1]
    model_data = dict(
        train_data=train_data,
        eval_data=eval_data,
        train_labels=train_labels,
        eval_labels=eval_labels
    )

    if save_as is not None:
        save_as_pickle(save_as, model_data)
    return model_data


# Network Parameters

data_slice = Slice.from_path(load_from="slice.pkl")
# data_slice.neuron_filter(300)

search_window_size = 300
step_size = 300

# Load model data

model_data = get_model_data(data_slice, search_window_size, step_size, load_from="test_network_18-7-18.pkl")
eval_labels = np.int32(model_data["eval_labels"])
train_labels = np.int32(model_data["train_labels"])
eval_data = model_data["eval_data"]
train_data = model_data["train_data"]
eval_data = np.float32(np.squeeze(eval_data,axis=2))
train_data = np.float32(np.squeeze(train_data,axis=2))


# Create Estimator

network_classifier = tf.estimator.Estimator(model_fn=cnn_model_fn,model_dir="test_network_19-7_model_11")

# Create logging hook

tensors_to_log = {"probabilities": "softmax_tensor"}
logging_hook = tf.train.LoggingTensorHook(tensors=tensors_to_log,every_n_iter=50)


# Train model

train_input_fn = tf.estimator.inputs.numpy_input_fn(
    x={"x": train_data},
    y=train_labels,
    batch_size=100,
    num_epochs=None,
    shuffle=True)

network_classifier.train(
    input_fn=train_input_fn,
    steps=10000,
    hooks=[logging_hook]
)


eval_input_fn = tf.estimator.inputs.numpy_input_fn(
    x={"x":train_data},
    y=train_labels,
    num_epochs=1,
    shuffle=False)


eval_results = network_classifier.evaluate(input_fn=eval_input_fn)
print("Training set: ",eval_results)

eval_input_fn = tf.estimator.inputs.numpy_input_fn(
    x={"x":eval_data},
    y=eval_labels,
    num_epochs=1,
    shuffle=False)


eval_results = network_classifier.evaluate(input_fn=eval_input_fn)
print("Evaluation set: ",eval_results)




for i in range(0,6):

    eval_data_1 = [e for j, e in enumerate(eval_data) if eval_labels[j] == i]
    eval_labels_1 = [e for j, e in enumerate(eval_labels) if eval_labels[j] == i]
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

