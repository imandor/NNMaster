import tensorflow as tf
from database_api_beta import Slice, Filter, hann, bin
from src.nets import MultiLayerPerceptron, ConvolutionalNeuralNetwork1
from src.metrics import test_accuracy, print_net_dict, plot_histogram
import numpy as np
from src.conf import mlp, sp1, cnn1
import os
import errno
from src.settings import save_as_pickle, load_pickle, save_net_dict
from src.preprocessing import time_shift_io, shuffle_io, position_as_map, filter_overrepresentation, count_occurrences
import datetime
import pickle
import random
import multiprocessing
from functools import reduce
from scipy import stats, spatial
from external.preprocessing_funcs import get_spikes_with_history


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


def run_network(net_dict):
    # S = ConvolutionalNeuralNetwork1([None, 56, 10, 1], cnn1)
    S = MultiLayerPerceptron([None, net_dict["N_NEURONS"], 50, 1], mlp)  # 56 147

    # saver = tf.train.Saver()
    sess = tf.Session()
    r2_scores_train = []
    avg_scores_train = []
    r2_scores_valid = []
    avg_scores_valid = []
    acc_scores_train = []
    acc_scores_valid = []
    early_stop_max = -np.inf
    X_train = net_dict["X_train"]
    y_train = net_dict["y_train"]
    if net_dict["LOAD_MODEL"] is True:
        saver = tf.train.import_meta_graph(net_dict["MODEL_PATH"] + ".meta")
        saver.restore(sess, MODEL_PATH)
    else:
        S.initialize(sess)

    # Train model

    sess.run(tf.local_variables_initializer())
    print("Training model...")
    xshape = [net_dict["BATCH_SIZE"]] + list(X_train[0].shape) + [1]
    yshape = [net_dict["BATCH_SIZE"]] + list(y_train[0].shape) + [1]
    metric_counter = 0  # net_dict["METRIC_ITER"]
    metric_step_counter = []
    stop_early = False
    for i in range(0, net_dict["EPOCHS"] + 1):
        # Train model
        X_train, y_train = shuffle_io(X_train, y_train, net_dict, i + 1)
        if metric_counter == net_dict["METRIC_ITER"] and stop_early is False:
            metric_step_counter.append(i)
            # saver.save(sess, net_dict["MODEL_PATH"])
            print("\n_-_-_-_-_-_-_-_-_-_-Epoch", i,"_-_-_-_-_-_-_-_-_-_-\n")
            # print("Training results:")
            # if True:
            #     r2_train, avg_train, accuracy_train = test_accuracy(sess, S, net_dict, net_dict["X_train"], net_dict["y_train"],
            #                                                         print_distance=True)
            #     r2_scores_train.append(r2_train)
            #     avg_scores_train.append(avg_train)
            #     acc_scores_train.append(accuracy_train)
            #     print("R2-score:",r2_train)
            #     print("Avg-distance:",avg_train,"\n")

            print("Validation results:")
            r2_valid, avg_valid, acc_valid = test_accuracy(sess, S, net_dict, net_dict["X_valid"],net_dict["y_valid"],
                                                           print_distance=True)
            r2_scores_valid.append(r2_valid)
            avg_scores_valid.append(avg_valid)
            acc_scores_valid.append(acc_valid)
            print("R2-score:",r2_valid)
            print("Avg-distance:",avg_valid)
            # saver.save(sess, net_dict["MODEL_PATH"], global_step=i, write_meta_graph=False)
            metric_counter = 0
            if net_dict["EARLY_STOPPING"] is True and i >= 5:  # most likely overfitting instead of training
                if i % 1 == 0:

                    _,_,acc_train = test_accuracy(sess, S, net_dict, net_dict["X_test"],net_dict["y_test"],
                                                                   print_distance=False)
                    acc = acc_train[19]
                    if early_stop_max < acc:
                        early_stop_max = acc
                    else:
                        if acc < early_stop_max - 0.01:
                            stop_early = True
                            r2_scores_train = r2_scores_train[0:-1] # Remove latest result which was worse
                            avg_scores_train = avg_scores_train[0:-1]
                            acc_scores_train = acc_scores_train[0:-1]
                            r2_scores_valid = r2_scores_valid[0:-1]
                            avg_scores_valid = avg_scores_valid[0:-1]
                            acc_scores_valid = acc_scores_valid[0:-1]

                            # metric_counter = net_dict["METRIC_ITER"]  # one last calculation with local maximum return

            # a = get_radius_accuracy(prediction_list, actual_list, [network_dict["X_STEP"], network_dict["Y_STEP"]], 19)
        for j in range(0, len(X_train) - net_dict["BATCH_SIZE"], net_dict["BATCH_SIZE"]):
            x = np.array([data_slice for data_slice in X_train[j:j + net_dict["BATCH_SIZE"]]])
            y = np.array(y_train[j:j + net_dict["BATCH_SIZE"]])
            x = np.reshape(x, xshape)
            y = np.reshape(y, yshape)
            if net_dict["NAIVE_TEST"] is False or net_dict["TIME_SHIFT"] == 0:
                t = np.max(S.train(sess, x, y, dropout=0.65))
        metric_counter = metric_counter + 1
        net_dict["epochs_trained"] = i
        if stop_early is True:
            break
    # saver.save(sess, net_dict["MODEL_PATH"])

    # Add performance to return dict

    net_dict["r2_scores_train"] = r2_scores_train
    net_dict["r2_scores_valid"] = r2_scores_valid
    net_dict["acc_scores_train"] = acc_scores_train
    net_dict["acc_scores_valid"] = acc_scores_valid
    net_dict["avg_scores_train"] = avg_scores_train
    net_dict["avg_scores_valid"] = avg_scores_valid
    net_dict["metric_step_counter"] = metric_step_counter
    # Close session and add current time shift to network save file
    if net_dict["MAKE_HISTOGRAM"] is True:
        plot_histogram(sess, S, net_dict, net_dict["X_valid"], net_dict["y_valid"])
        plot_histogram(sess, S, net_dict, net_dict["X_train"], net_dict["y_train"])
    sess.close()
    return dict(net_dict)


if __name__ == '__main__':
    now = datetime.datetime.now().isoformat()

    # Glaser data set
    # MODEL_PATH = "G:/master_datafiles/trained_networks/MLP_Custom_2018-20-09/"
    # RAW_DATA_PATH = "C:/Users/NN\Desktop/Neural_Decoding-master/example_data_hc.pickle"
    # FILTERED_DATA_PATH = "G:/master_datafiles/filtered_data/neocortex_hann_win_size_100.pkl"

    # prefrontal cortex

    # MODEL_PATH = "G:/master_datafiles/trained_networks/MLP_PFC_2018-10-10_400_400_400/"
    # RAW_DATA_PATH = "G:/master_datafiles/raw_data/2018-04-09_14-39-52/"
    # FILTERED_DATA_PATH = "G:/master_datafiles/filtered_data/neocortex_hann_win_size_20.pkl"

    # hippocampus

    MODEL_PATH = "G:/master_datafiles/trained_networks/MLP_HC_2018-10-19_1000_200_100/"
    RAW_DATA_PATH = "G:/master_datafiles/raw_data/2018-05-16_17-13-37/"
    FILTERED_DATA_PATH = "G:/master_datafiles/filtered_data/hippocampus_hann_win_size_25_09-5_7.pkl"
    NEURONS_KEPT_FACTOR = 1.0

    # Program execution settings

    LOAD_RAW_DATA = True  # load source data from raw data path or load default model
    # LOAD_RAW_DATA = True # load source data from raw data path or load default model
    LOAD_GLASER_DATA = False
    SAVE_FILTERED_DATA = True
    MAKE_HISTOGRAM = False
    LOAD_MODEL = False  # load model from model path
    TRAIN_MODEL = True  # train model or just show results
    EPOCHS = 1000
    INITIAL_TIMESHIFT = 0
    TIME_SHIFT_ITER = 200
    TIME_SHIFT_STEPS = 1
    METRIC_ITER = 1  # after how many epochs network is validated <---
    SHUFFLE_DATA = True  # whether to randomly shuffle the data in big slices
    SHUFFLE_FACTOR = 500
    EARLY_STOPPING = True
    NAIVE_TEST = False
    K_CROSS_VALIDATION = 1
    # Input data parameters

    SLICE_SIZE = 1000
    Y_SLICE_SIZE = 200
    STRIDE = 100
    BATCH_SIZE = 50
    WIN_SIZE = 20
    SEARCH_RADIUS = WIN_SIZE * 2
    VALID_RATIO = 0.1
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
        net_dict = load_pickle(FILTERED_DATA_PATH)
    else:
        net_dict = dict()
    net_dict["MAKE_HISTOGRAM"] = MAKE_HISTOGRAM
    net_dict["STRIDE"] = STRIDE
    net_dict["Y_SLICE_SIZE"] = Y_SLICE_SIZE
    net_dict["network_type"] = "Multi Layer Perceptron"
    net_dict["EPOCHS"] = EPOCHS
    net_dict["session_filter"] = session_filter
    net_dict["TIME_SHIFT_STEPS"] = TIME_SHIFT_STEPS
    net_dict["SHUFFLE_DATA"] = SHUFFLE_DATA
    net_dict["SHUFFLE_FACTOR"] = SHUFFLE_FACTOR
    net_dict["TIME_SHIFT_ITER"] = TIME_SHIFT_ITER
    net_dict["MODEL_PATH"] = MODEL_PATH
    net_dict["learning_rate"] = "placeholder"  # TODO
    net_dict["r2_scores_train"] = []
    net_dict["r2_scores_valid"] = []
    net_dict["acc_scores_train"] = []
    net_dict["acc_scores_valid"] = []
    net_dict["avg_scores_train"] = []
    net_dict["avg_scores_valid"] = []
    net_dict["LOAD_MODEL"] = LOAD_MODEL
    net_dict["INITIAL_TIMESHIFT"] = INITIAL_TIMESHIFT
    net_dict["TRAIN_MODEL"] = TRAIN_MODEL
    net_dict["METRIC_ITER"] = METRIC_ITER
    net_dict["BATCH_SIZE"] = BATCH_SIZE
    net_dict["SLICE_SIZE"] = SLICE_SIZE
    net_dict["epochs_trained"] = 0
    net_dict["RAW_DATA_PATH"] = RAW_DATA_PATH
    net_dict["X_MAX"] = X_MAX
    net_dict["Y_MAX"] = Y_MAX
    net_dict["X_MIN"] = X_MIN
    net_dict["Y_MIN"] = Y_MIN
    net_dict["X_STEP"] = X_STEP
    net_dict["Y_STEP"] = Y_STEP
    net_dict["WIN_SIZE"] = WIN_SIZE
    net_dict["SEARCH_RADIUS"] = SEARCH_RADIUS
    net_dict["EARLY_STOPPING"] = EARLY_STOPPING
    net_dict["NAIVE_TEST"] = NAIVE_TEST
    # Preprocess data

    if LOAD_RAW_DATA is True and LOAD_GLASER_DATA is False:
        # TODO
        # session = Slice.from_raw_data(RAW_DATA_PATH)
        # session.filter_neurons(100)
        # session.print_details()
        # print("Convolving data...")
        # session.set_filter(net_dict["session_filter"])
        # print("Finished convolving data")
        # session.filtered_spikes = stats.zscore(session.filtered_spikes, axis=1)  # Z Score neural activity
        # session.to_pickle("slice_OFC.pkl")
        # TODO
        session = Slice.from_pickle("slice_OFC.pkl")
        # session.filter_neurons_randomly(ASD)
        session.print_details()
        if SAVE_FILTERED_DATA is True:
            save_as_pickle(FILTERED_DATA_PATH, net_dict)
            save_net_dict(MODEL_PATH, net_dict)
            net_dict["N_NEURONS"] = session.n_neurons

    # Start network with different time-shifts

    print_net_dict(net_dict)  # show network parameters in console

    for z in range(INITIAL_TIMESHIFT, INITIAL_TIMESHIFT + TIME_SHIFT_STEPS * TIME_SHIFT_ITER, TIME_SHIFT_ITER):
        for k in range (0,K_CROSS_VALIDATION):
            iter = int(z - INITIAL_TIMESHIFT) // TIME_SHIFT_ITER
            if LOAD_GLASER_DATA is False:
                print("Time shift is now", z)

                # Time-Shift input and output

                X, y = time_shift_io(session, z, net_dict)
                if len(X) != len(y):
                    raise ValueError("Error: Length of x and y are not identical")
                X, y = shuffle_io(X, y, net_dict, 3)

            if LOAD_GLASER_DATA is True:
                with open(RAW_DATA_PATH, 'rb') as f:
                    neural_data, y_raw = pickle.load(f, encoding='latin1')  # If using python 3
                # neural_data = neural_data[135:237]
                # y_raw = y_raw[135:237]
                bins_before = 4  # How many bins of neural data prior to the output are used for decoding
                bins_current = 1  # Whether to use concurrent time bin of neural data
                bins_after = 5  # How many bins of neural data after the output are used for decoding
                X = get_spikes_with_history(neural_data, bins_before, bins_after, bins_current)
                # X_mean = np.nanmean(X, axis=0)
                # X_std = np.nanstd(X, axis=0)
                # X = (X - X_mean) / X_std
                y = []
                for i, posxy_list in reversed(list(enumerate(y_raw))):
                    if np.isnan(posxy_list).any() or np.isnan(X[i].any()):
                        X = np.delete(X, i, axis=0)
                    else:
                        x_list = ((np.array(posxy_list[0]) - X_MIN) // X_STEP).astype(int)
                        y_list = ((np.array(posxy_list[1]) - Y_MIN) // Y_STEP).astype(int)
                        posxy_list = [y_list, y_list]
                        y.append(position_as_map(posxy_list, X_STEP, Y_STEP, 300, 0, 300, 0))
                X = X.transpose([0, 2, 1])
                y = list(reversed(y))
                X, y = shuffle_io(X, y, net_dict, 3)

            # Assign training and testing set

            if K_CROSS_VALIDATION == 1:
                valid_length = int(len(X) * VALID_RATIO)
            # net_dict["X_train"], net_dict["y_train"] = filter_overrepresentation(X[valid_length:], y[valid_length:], 70,
            #                                                                      net_dict, axis=0)
            # net_dict["X_valid"], net_dict["y_valid"] = filter_overrepresentation(X[:valid_length], y[:valid_length], 7,
            #                                                                      net_dict, axis=0)

                net_dict["X_train"] = X[valid_length:]
                net_dict["y_train"] = y[valid_length:]
                net_dict["X_valid"] = X[:valid_length//2]
                net_dict["y_valid"] = y[:valid_length//2]
                net_dict["X_test"] = X[valid_length//2:valid_length]
                net_dict["y_test"] = y[valid_length//2:valid_length]
                net_dict["TIME_SHIFT"] = z
            # train_oc = count_occurrences(net_dict["y_train"], net_dict)
            # valid_oc = count_occurrences(net_dict["y_valid"], net_dict)

            if TIME_SHIFT_STEPS == 1:
                save_dict = run_network(net_dict)
            else:
                with multiprocessing.Pool(
                        1) as p:  # keeping network inside process prevents memory issues when restarting session
                    save_dict = p.map(run_network, [net_dict])[0]
                    p.close()

        # Remove raw data for faster later loading and save to file

        save_dict.pop('X', None)
        save_dict.pop('X_train', None)
        save_dict.pop('X_valid', None)
        save_dict.pop('X_test', None)
        save_dict.pop('y', None)
        save_dict.pop('y_valid', None)
        save_dict.pop('y_train', None)
        save_dict.pop('y_test', None)
        save_dict.pop('y_list', None)

        path = MODEL_PATH + "output/" + chr(65 + iter) + "_" + now[0:10] + "_network_output_timeshift=" + str(
            z) + ".pkl"
        save_as_pickle(path, save_dict)
    print("fin")
