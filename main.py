import tensorflow as tf
from database_api_beta import Slice, Filter, hann, bin
from src.nets import MultiLayerPerceptron, ConvolutionalNeuralNetwork1
from src.metrics import test_accuracy, print_net_dict,get_radius_accuracy
import numpy as np
from src.conf import mlp, sp1, cnn1
import os
import errno
from src.settings import save_as_pickle, load_pickle
from src.preprocessing import time_shift_io,shuffle_io
import datetime
import pickle
import random
import multiprocessing
from functools import reduce
from scipy import stats, spatial
from external.preprocessing_funcs import  get_spikes_with_history


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
    early_stop_min = np.inf
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
    yshape = [net_dict["BATCH_SIZE"]]  + [2]
    metric_counter = 0
    metric_step_counter = []
    stop_early = False
    for i in range(0, net_dict["EPOCHS"]+1):
        # Train model
        X_train, y_train = shuffle_io(X_train, y_train, net_dict,i+1)
        t = None
        for j in range(0, len(X_train) - net_dict["BATCH_SIZE"], net_dict["BATCH_SIZE"]):
            x = np.array([data_slice for data_slice in X_train[j:j + net_dict["BATCH_SIZE"]]])
            y = np.array(y_train[j:j + net_dict["BATCH_SIZE"]])
            x = np.reshape(x, xshape)
            y = np.reshape(y, yshape)
            t = np.max(S.train(sess, x, y, dropout=0.8))

        # Test accuracy and add validation to output. Check if early stopping is necessary



        # Check if early stopping applies

        if net_dict["EARLY_STOPPING"] is True and i>200: # most likely overfitting
            if i % 20 == 0 and r2_scores_train[-1][0]>0.99:
                r2_valid, avg_valid, acc_valid = test_accuracy(sess, S, net_dict, i, is_training_data=False,
                                                           show_plot=False, plot_after_iter=500,
                                                           print_distance=False)
                if early_stop_min > acc_valid[19]:
                    early_stop_min = acc_valid[19]
                else:
                    if early_stop_min > acc_valid[19]  + 2:
                            stop_early = True
                            metric_counter = net_dict["METRIC_ITER"] # one last calculation with local maximum return

        if metric_counter == net_dict["METRIC_ITER"]:
            print(i, ", loss:", t)
            metric_step_counter.append(i)
            saver.save(sess, net_dict["MODEL_PATH"])
            print("Epoch", i)
            print("training data:")
            r2_train, avg_train, accuracy_train = test_accuracy(sess, S, net_dict, i, is_training_data=True,
                                                                show_plot=False, plot_after_iter=5000,
                                                                print_distance=True)
            r2_scores_train.append(r2_train)
            avg_scores_train.append(avg_train)
            acc_scores_train.append(accuracy_train)
            print(r2_train, avg_train)
            print("validation data:")
            r2_valid, avg_valid, acc_valid = test_accuracy(sess, S, net_dict, i,is_training_data=False,
                                                             show_plot=False, plot_after_iter=500,
                                                             print_distance=True)
            print(r2_valid, avg_valid)
            r2_scores_valid.append(r2_valid)
            avg_scores_valid.append(avg_valid)
            acc_scores_valid.append(acc_valid)



            # saver.save(sess, net_dict["MODEL_PATH"], global_step=i, write_meta_graph=False)
            print("____________________")
            metric_counter = 0
            # a = get_radius_accuracy(prediction_list, actual_list, [network_dict["X_STEP"], network_dict["Y_STEP"]], 19)

        metric_counter = metric_counter + 1
        if stop_early is True:
            break
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


    # Glaser data set
    # MODEL_PATH = "G:/master_datafiles/trained_networks/MLP_Custom_2018-20-09/"
    # RAW_DATA_PATH = "C:/Users/NN\Desktop/Neural_Decoding-master/example_data_hc.pickle"
    # FILTERED_DATA_PATH = "G:/master_datafiles/filtered_data/neocortex_hann_win_size_100.pkl"



    # neo cortex

    # MODEL_PATH = "G:/master_datafiles/trained_networks/MLP_OFC_2018-09-20_verification_set/"
    # RAW_DATA_PATH = "G:/master_datafiles/raw_data/2018-04-09_14-39-52/"
    # FILTERED_DATA_PATH = "G:/master_datafiles/filtered_data/neocortex_hann_win_size_100.pkl"

    # hippocampus

    MODEL_PATH = "G:/master_datafiles/trained_networks/MLP_hippocampus_2018-09-20/"
    RAW_DATA_PATH = "G:/master_datafiles/raw_data/2018-05-16_17-13-37/"
    FILTERED_DATA_PATH = "G:/master_datafiles/filtered_data/hippocampus_hann_win_size_25_09-5_7.pkl"

    # Program execution settings

    LOAD_RAW_DATA = True  # load source data from raw data path or load default model
    # LOAD_RAW_DATA = True # load source data from raw data path or load default model
    SAVE_FILTERED_DATA = True
    LOAD_MODEL = False  # load model from model path
    TRAIN_MODEL = True  # train model or just show results
    EPOCHS = 100000
    INITIAL_TIMESHIFT = 0
    TIME_SHIFT_ITER = 200
    TIME_SHIFT_STEPS = 1
    METRIC_ITER = 50 # after how many epochs network is validated
    SHUFFLE_DATA = False # wether to randomly shuffle the data in big slices
    SHUFFLE_FACTOR = 20
    EARLY_STOPPING = True

    # Input data parameters

    SLICE_SIZE = 800
    Y_SLICE_SIZE = 200
    STRIDE = int(SLICE_SIZE*0.1)
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
    net_dict["Y_SLICE_SIZE"] = Y_SLICE_SIZE
    net_dict["STRIDE"] = STRIDE

    # Preprocess data

    if LOAD_RAW_DATA is True :
        # session = Slice.from_raw_data(RAW_DATA_PATH)
        # session.neuron_filter(100)
        # session.print_details()
        # print("Convolving data...")
        # session.set_filter(net_dict["session_filter"])
        # print("Finished convolving data")
        # session.filtered_spikes = stats.zscore(session.filtered_spikes, axis=1)  # Z Score neural activity
        # session.to_pickle("slice_OFC.pkl")
        session = Slice.from_pickle("slice_OFC.pkl")
        session.print_details()
        if SAVE_FILTERED_DATA is True:
            save_as_pickle(FILTERED_DATA_PATH, net_dict)

    # Start network with different time-shifts

    print_net_dict(net_dict) # show network parameters in console

    for z in range(INITIAL_TIMESHIFT, INITIAL_TIMESHIFT+ TIME_SHIFT_STEPS*TIME_SHIFT_ITER,TIME_SHIFT_ITER):
        iter = int(z-INITIAL_TIMESHIFT)//TIME_SHIFT_ITER
        print("Time shift is now", z)

        # Time-Shift input and output

        X, y = time_shift_io(session,z,net_dict)
        if len(X)!= len(y):
            raise ValueError("Error: Length of x and y are not identical")
        X, y = shuffle_io(X,y,net_dict,3)




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

        path = MODEL_PATH  + "output/" + chr(65 + iter) + "_" + now[0:10] + "_network_output_timeshift=" + str(
            z) + ".pkl"
        save_as_pickle(path, save_dict)
    print("fin")
