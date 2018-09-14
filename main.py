import tensorflow as tf
from database_api_beta import Slice, Filter, hann, bin
from src.nets import MultiLayerPerceptron, ConvolutionalNeuralNetwork1
from src.metrics import test_accuracy, print_network_dict
import numpy as np
from src.conf import mlp, sp1, cnn1
import os
import errno
from src.settings import save_as_pickle, load_pickle
from src.preprocessing import preprocess_raw_data
import datetime
import random
import multiprocessing
from functools import reduce


def hann_generator(width):
    def hann(x):
        if np.abs(x) < width:
            return (1 + np.cos(x / width * np.pi)) / 2
        else:
            return 0

    return hann


def bin_filter(x):
    return 1


def shift(li, n):
    return li[n:] + li[:n]


def time_shift_data(X, y, n):
    y = shift(y, n)[n:]
    X = X[:len(X) - n]
    return X, y


def run_network(network_dict):
    # S = ConvolutionalNeuralNetwork1([None, 166, 20, 1], cnn1)
    S = MultiLayerPerceptron([None, 56, 100, 1], mlp) # 56 147

    saver = tf.train.Saver()
    sess = tf.Session()
    r2_scores_train = []
    avg_scores_train = []
    r2_scores_eval = []
    avg_scores_eval = []
    acc_scores_train = []
    acc_scores_eval = []
    X_train = network_dict["X_train"]
    X_eval = network_dict["X_eval"]
    y_train = network_dict["y_train"]
    y_eval = network_dict["y_eval"]
    if network_dict["LOAD_MODEL"] is True:
        saver = tf.train.import_meta_graph(network_dict["MODEL_PATH"] + ".meta")
        saver.restore(sess, MODEL_PATH)
    else:
        S.initialize(sess)

    # Train model

    if network_dict["TRAIN_MODEL"] is True:
        sess.run(tf.local_variables_initializer())
        print("Training model...")
        xshape = [network_dict["BATCH_SIZE"]] + list(X_train[0].shape) + [1]
        yshape = [network_dict["BATCH_SIZE"]] + list(y_train[0].shape) + [1]
        metric_counter = 0
        metric_step_counter = []
        for i in range(0, network_dict["EPOCHS"]+1):
            # Train model
            for j in range(0, len(X_train) - network_dict["BATCH_SIZE"], network_dict["BATCH_SIZE"]):
                x = np.array([data_slice for data_slice in X_train[j:j + network_dict["BATCH_SIZE"]]])
                y = np.array(y_train[j:j + network_dict["BATCH_SIZE"]])
                x = np.reshape(x, xshape)
                y = np.reshape(y, yshape)
                t = np.max(S.train(sess, x, y, dropout=0.8))
                print(i, ", loss:", t)

                # Test accuracy and add evaluation to output

                if metric_counter == network_dict["METRIC_ITER"]:
                    metric_step_counter.append(j)
                    saver.save(sess, network_dict["MODEL_PATH"])
                    print("Step", i)
                    print("training data:")
                    r2_train, avg_train, accuracy_train = test_accuracy(sess, S, network_dict, i, is_training_data=True,
                                                                        show_plot=True, plot_after_iter=5000,
                                                                        print_distance=False)
                    r2_scores_train.append(r2_train)
                    avg_scores_train.append(avg_train)
                    acc_scores_train.append(accuracy_train)
                    print(r2_train, avg_train)
                    print("evaluation data:")
                    r2_eval, avg_eval, accuracy_eval = test_accuracy(sess, S, network_dict, i,is_training_data=False,
                                                                     show_plot=True, plot_after_iter=500,
                                                                     print_distance=False)
                    r2_scores_eval.append(r2_eval)
                    avg_scores_eval.append(avg_eval)
                    acc_scores_eval.append(accuracy_eval)
                    print(r2_eval, avg_eval)
                    # saver.save(sess, network_dict["MODEL_PATH"], global_step=i, write_meta_graph=False)
                    print("____________________")
                    metric_counter = 0
            metric_counter = metric_counter + 1
        saver.save(sess, network_dict["MODEL_PATH"])

    # Add performance to return dict

    network_dict["r2_scores_train"] = r2_scores_train
    network_dict["r2_scores_eval"] = r2_scores_eval
    network_dict["acc_scores_train"] = acc_scores_train
    network_dict["acc_scores_eval"] = acc_scores_eval
    network_dict["avg_scores_train"] = avg_scores_train
    network_dict["avg_scores_eval"] = avg_scores_eval
    network_dict["trained_steps"] = network_dict["trained_steps"] + network_dict["METRIC_ITER"]
    network_dict["metric_step_counter"] = metric_step_counter
    # Close session and add current time shift to network save file

    sess.close()
    return dict(network_dict)


if __name__ == '__main__':
    now = datetime.datetime.now().isoformat()

    # neo cortex

    # MODEL_PATH = "G:/master_datafiles/trained_networks/MLP_OFC_2018-09-13/"
    # RAW_DATA_PATH = "G:/master_datafiles/raw_data/2018-04-09_14-39-52/"
    # FILTERED_DATA_PATH = "G:/master_datafiles/filtered_data/neocortex_hann_win_size_100.pkl"

    # hippocampus

    MODEL_PATH = "G:/master_datafiles/trained_networks/MLP_hippocampus_2018-09-12/"
    RAW_DATA_PATH = "G:/master_datafiles/raw_data/2018-05-16_17-13-37/"
    FILTERED_DATA_PATH = "G:/master_datafiles/filtered_data/hippocampus_hann_win_size_25_09-5_7.pkl"

    # Program execution settings

    LOAD_RAW_DATA = True  # load source data from raw data path or load default model
    # LOAD_RAW_DATA = True # load source data from raw data path or load default model
    SAVE_MODEL = False
    LOAD_MODEL = False  # load model from model path
    TRAIN_MODEL = True  # train model or just show results
    EPOCHS = 500
    INITIAL_TIMESHIFT = 1000
    TIME_SHIFT_ITER = 200
    TIME_SHIFT_STEPS = 5
    METRIC_ITER = 100 # after how many epochs network is evaluated
    SHUFFLE_DATA = True # wether to randomly shuffle the data in big slices
    SHUFFLE_FACTOR = 20

    # Input data parameters

    SLICE_SIZE = 200
    BATCH_SIZE = 128
    WIN_SIZE = 20
    SEARCH_RADIUS = WIN_SIZE * 2
    EVAL_RATIO = 0.1
    BINS_BEFORE = 4
    BINS_AFTER = 5
    X_MAX = 240
    Y_MAX = 190
    X_MIN = 0
    Y_MIN = 100
    X_STEP = 3
    Y_STEP = 3
    session_filter = Filter(func=hann, search_radius=SEARCH_RADIUS, step_size=WIN_SIZE)

    # Create save file directories

    try:
        os.makedirs(os.path.dirname(MODEL_PATH))
        os.makedirs(os.path.dirname(MODEL_PATH + "output/"))
        os.makedirs(os.path.dirname(MODEL_PATH + "images/"))
    except OSError as exc:  # Guard against race condition
        if exc.errno != errno.EEXIST:
            raise


    # Update network dict values


    if LOAD_RAW_DATA is False:
        network_dict = load_pickle(FILTERED_DATA_PATH)
    else:
        network_dict = dict()
    network_dict["network_type"] = "Multi Layer Perceptron"
    network_dict["EPOCHS"] = EPOCHS
    network_dict["session_filter"] = session_filter
    network_dict["TIME_SHIFT_STEPS"] = TIME_SHIFT_STEPS
    network_dict["SHUFFLE_DATA"] = SHUFFLE_DATA
    network_dict["SHUFFLE_FACTOR"] = SHUFFLE_FACTOR
    network_dict["TIME_SHIFT_ITER"] = TIME_SHIFT_ITER
    network_dict["MODEL_PATH"] = MODEL_PATH
    network_dict["learning_rate"] = "placeholder"  # TODO
    network_dict["r2_scores_train"] = []
    network_dict["r2_scores_eval"] = []
    network_dict["acc_scores_train"] = []
    network_dict["acc_scores_eval"] = []
    network_dict["avg_scores_train"] = []
    network_dict["avg_scores_eval"] = []
    network_dict["LOAD_MODEL"] = LOAD_MODEL
    network_dict["INITIAL_TIMESHIFT"] = INITIAL_TIMESHIFT
    network_dict["TRAIN_MODEL"] = TRAIN_MODEL
    network_dict["METRIC_ITER"] = METRIC_ITER
    network_dict["BATCH_SIZE"] = BATCH_SIZE
    network_dict["BINS_BEFORE"] = BINS_BEFORE
    network_dict["BINS_AFTER"] = BINS_AFTER
    network_dict["SLICE_SIZE"] = SLICE_SIZE
    network_dict["trained_steps"] = 0
    network_dict["RAW_DATA_PATH"] = RAW_DATA_PATH
    network_dict["X_MAX"] = X_MAX
    network_dict["Y_MAX"] = Y_MAX
    network_dict["X_MIN"] = X_MIN
    network_dict["Y_MIN"] = Y_MIN
    network_dict["X_STEP"] = X_STEP
    network_dict["Y_STEP"] = Y_STEP
    network_dict["WIN_SIZE"] = WIN_SIZE
    network_dict["SEARCH_RADIUS"] = SEARCH_RADIUS

    # Preprocess data

    if LOAD_RAW_DATA is True:
        X, y_list, metadata = preprocess_raw_data(network_dict)
        # for i,x in enumerate(X): # random test for baseline
        #     for j,y in enumerate(x):
        #         X[i][j] = np.random.rand(100)
        network_dict["X"] = X
        network_dict["y_list"] = y_list
        network_dict["metadata"] = metadata
        network_dict["trained_steps"] = 0
        if SAVE_MODEL is True:
            save_as_pickle(FILTERED_DATA_PATH, network_dict)

    # Assign training and evaluation set

    y_list = network_dict["y_list"]
    eval_length = int(len(network_dict["X"]) * EVAL_RATIO)
    X_eval = network_dict["X"][:eval_length]
    metadata_eval = network_dict["metadata"][:eval_length]
    X_train = network_dict["X"][eval_length:]
    metadata_train = network_dict["metadata"][eval_length:]


    network_dict["X_train"] = X_train
    network_dict["X_eval"] = X_eval
    network_dict["y_train"] = []
    network_dict["y_eval"] = []
    network_dict["eval_length"] = eval_length
    network_dict["metadata_train"] = metadata_train
    network_dict["metadata_eval"] = metadata_eval

    if len(y_list) < TIME_SHIFT_STEPS:
        raise ValueError("Error: filtered data size does not match time-shift steps")

    # Start network with different time-shifts

    print_network_dict(network_dict) # show network parameters in console

    for z in range(INITIAL_TIMESHIFT, INITIAL_TIMESHIFT+ TIME_SHIFT_STEPS*TIME_SHIFT_ITER,TIME_SHIFT_ITER):
        iter = int(z-INITIAL_TIMESHIFT)//TIME_SHIFT_ITER
        print("Time shift is now", z)
        network_dict["TIME_SHIFT"] = z
        current_y = y_list[iter]
        network_dict["y_eval"] = current_y[:network_dict["eval_length"]]
        network_dict["y_train"] = current_y[network_dict["eval_length"]:]
        if TIME_SHIFT_STEPS == 1:
            save_dict = run_network(network_dict)
        else:
            with multiprocessing.Pool(
                    1) as p:  # keeping network inside process prevents memory issues when restarting session
                save_dict = p.map(run_network, [network_dict])[0]
                p.close()

        # Remove raw data for faster later loading and save to file

        save_dict.pop('X', None)
        save_dict.pop('X_train', None)
        save_dict.pop('X_eval', None)
        save_dict.pop('y', None)
        save_dict.pop('y_eval', None)
        save_dict.pop('y_train', None)
        save_dict.pop('metadata_train', None)
        save_dict.pop('metadata_eval', None)
        save_dict.pop('y_list', None)
        save_dict.pop('metadata', None)

        path = MODEL_PATH  + "output/" + chr(97 + iter+INITIAL_TIMESHIFT//TIME_SHIFT_ITER) + "_" + now[0:10] + "_network_output_timeshift=" + str(
            z) + ".pkl"
        save_as_pickle(path, save_dict)
    print("fin")
