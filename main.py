import tensorflow as tf
from database_api_beta import Slice, Filter, hann, bin
from src.nets import MultiLayerPerceptron, ConvolutionalNeuralNetwork1
from src.metrics import test_accuracy, print_net_dict
import numpy as np
from src.conf import mlp, sp1, cnn1
import os
import errno
from src.settings import save_as_pickle, load_pickle
from src.preprocessing import preprocess_raw_data,time_shift_io,shuffle_io
import datetime
import random
import multiprocessing
from functools import reduce
from scipy import stats, spatial


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
    # S = ConvolutionalNeuralNetwork1([None, 56, 40, 1], cnn1)
    S = MultiLayerPerceptron([None, 56, 40, 1], mlp) # 56 147

    saver = tf.train.Saver()
    sess = tf.Session()
    r2_scores_train = []
    avg_scores_train = []
    r2_scores_valid = []
    avg_scores_valid = []
    acc_scores_train = []
    acc_scores_valid = []
    X_train = net_dict["X_train"]
    y_train = net_dict["y_train"]
    if net_dict["LOAD_MODEL"] is True:
        saver = tf.train.import_meta_graph(net_dict["MODEL_PATH"] + ".meta")
        saver.restore(sess, MODEL_PATH)
    else:
        S.initialize(sess)

    # Train model

    if net_dict["TRAIN_MODEL"] is True:
        sess.run(tf.local_variables_initializer())
        print("Training model...")
        xshape = [net_dict["BATCH_SIZE"]] + list(X_train[0].shape) + [1]
        yshape = [net_dict["BATCH_SIZE"]] + list(y_train[0].shape) + [1]
        metric_counter = 0
        metric_step_counter = []
        EARLY_STOP = False
        for i in range(0, net_dict["EPOCHS"]+1):
            # Train model
            for j in range(0, len(X_train) - net_dict["BATCH_SIZE"], net_dict["BATCH_SIZE"]):
                x = np.array([data_slice for data_slice in X_train[j:j + net_dict["BATCH_SIZE"]]])
                y = np.array(y_train[j:j + net_dict["BATCH_SIZE"]])
                x = np.reshape(x, xshape)
                y = np.reshape(y, yshape)
                t = np.max(S.train(sess, x, y, dropout=0.6))
                # print(i, ", loss:", t)

                # Test accuracy and add validation to output. Check if early stopping is necessary

                if metric_counter == net_dict["METRIC_ITER"]:
                    metric_step_counter.append(j)
                    saver.save(sess, net_dict["MODEL_PATH"])
                    print("Step", i)
                    print("training data:")
                    r2_train, avg_train, accuracy_train = test_accuracy(sess, S, net_dict, i, is_training_data=True,
                                                                        show_plot=False, plot_after_iter=5000,
                                                                        print_distance=False)
                    r2_scores_train.append(r2_train)
                    avg_scores_train.append(avg_train)
                    acc_scores_train.append(accuracy_train)
                    print(r2_train, avg_train)
                    print("validation data:")
                    r2_valid, avg_valid, accuracy_valid = test_accuracy(sess, S, net_dict, i,is_training_data=False,
                                                                     show_plot=False, plot_after_iter=500,
                                                                     print_distance=False)
                    r2_scores_valid.append(r2_valid)
                    avg_scores_valid.append(avg_valid)
                    acc_scores_valid.append(accuracy_valid)

                    # Check if early stopping applies
                    print(r2_valid, avg_valid)
                    if net_dict["EARLY_STOPPING"] is True and len(avg_scores_valid) > 1 and i > 200:
                        if avg_valid < avg_scores_valid[-2]:
                            EARLY_STOP = True
                            break


                    # saver.save(sess, net_dict["MODEL_PATH"], global_step=i, write_meta_graph=False)
                    print("____________________")
                    metric_counter = 0
            if EARLY_STOP is True:
                break
            metric_counter = metric_counter + 1
        saver.save(sess, net_dict["MODEL_PATH"])

    # Add performance to return dict

    net_dict["r2_scores_train"] = r2_scores_train
    net_dict["r2_scores_valid"] = r2_scores_valid
    net_dict["acc_scores_train"] = acc_scores_train
    net_dict["acc_scores_valid"] = acc_scores_valid
    net_dict["avg_scores_train"] = avg_scores_train
    net_dict["avg_scores_valid"] = avg_scores_valid
    net_dict["trained_steps"] = net_dict["trained_steps"] + net_dict["METRIC_ITER"]
    net_dict["metric_step_counter"] = metric_step_counter
    # Close session and add current time shift to network save file

    sess.close()
    return dict(net_dict)


if __name__ == '__main__':
    now = datetime.datetime.now().isoformat()

    # neo cortex

    # MODEL_PATH = "G:/master_datafiles/trained_networks/MLP_OFC_2018-09-13/"
    # RAW_DATA_PATH = "G:/master_datafiles/raw_data/2018-04-09_14-39-52/"
    # FILTERED_DATA_PATH = "G:/master_datafiles/filtered_data/neocortex_hann_win_size_100.pkl"

    # hippocampus

    MODEL_PATH = "G:/master_datafiles/trained_networks/MLP_hippocampus_2018-09-14/"
    RAW_DATA_PATH = "G:/master_datafiles/raw_data/2018-05-16_17-13-37/"
    FILTERED_DATA_PATH = "G:/master_datafiles/filtered_data/hippocampus_hann_win_size_25_09-5_7.pkl"

    # Program execution settings

    LOAD_RAW_DATA = True  # load source data from raw data path or load default model
    # LOAD_RAW_DATA = True # load source data from raw data path or load default model
    SAVE_FILTERED_DATA = True
    LOAD_MODEL = False  # load model from model path
    TRAIN_MODEL = True  # train model or just show results
    EPOCHS = 1000
    INITIAL_TIMESHIFT = 0
    TIME_SHIFT_ITER = 500
    TIME_SHIFT_STEPS = 1
    METRIC_ITER = 50 # after how many epochs network is validated
    SHUFFLE_DATA = True # wether to randomly shuffle the data in big slices
    SHUFFLE_FACTOR = 20
    EARLY_STOPPING = False

    # Input data parameters

    SLICE_SIZE = 800
    BATCH_SIZE = 50
    WIN_SIZE = 20
    SEARCH_RADIUS = WIN_SIZE * 2
    VALID_RATIO = 0.1
    BINS_BEFORE = 0
    BINS_AFTER = 0
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
    net_dict["BINS_BEFORE"] = BINS_BEFORE
    net_dict["BINS_AFTER"] = BINS_AFTER
    net_dict["SLICE_SIZE"] = SLICE_SIZE
    net_dict["trained_steps"] = 0
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
    
    # Preprocess data

    if LOAD_RAW_DATA is True:
        # session = Slice.from_raw_data(RAW_DATA_PATH)
        # session.neuron_filter(100)
        # print("Convolving data...")
        # session.set_filter(net_dict["session_filter"])
        # print("Finished convolving data")
        # session.filtered_spikes = stats.zscore(session.filtered_spikes, axis=1)  # Z Score neural activity
        # session.to_pickle("slice_2.pkl")
        session = Slice.from_pickle("slice_2.pkl")
        session.print_details()
        net_dict["X"] = preprocess_raw_data(session, net_dict)
        if SAVE_FILTERED_DATA is True:
            save_as_pickle(FILTERED_DATA_PATH, net_dict)

    # Start network with different time-shifts

    print_net_dict(net_dict) # show network parameters in console

    for z in range(INITIAL_TIMESHIFT, INITIAL_TIMESHIFT+ TIME_SHIFT_STEPS*TIME_SHIFT_ITER,TIME_SHIFT_ITER):
        iter = int(z-INITIAL_TIMESHIFT)//TIME_SHIFT_ITER
        print("Time shift is now", z)

        # Time-Shift input and output

        X, y = time_shift_io(session,net_dict["X"],z,net_dict)
        X, y = shuffle_io(X,y,net_dict)

        # Assign training and testing set

        valid_length = int(len(X) * VALID_RATIO)
        net_dict["X_valid"] = X[:valid_length]
        net_dict["X_train"] = X[valid_length:]
        net_dict["y_valid"] = y[:valid_length]
        net_dict["y_train"] = y[valid_length:]
        net_dict["TIME_SHIFT"] = z

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
        save_dict.pop('y', None)
        save_dict.pop('y_valid', None)
        save_dict.pop('y_train', None)
        save_dict.pop('y_list', None)

        path = MODEL_PATH  + "output/" + chr(97 + iter+INITIAL_TIMESHIFT//TIME_SHIFT_ITER) + "_" + now[0:10] + "_network_output_timeshift=" + str(
            z) + ".pkl"
        save_as_pickle(path, save_dict)
    print("fin")
