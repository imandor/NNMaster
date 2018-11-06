
from src.database_api_beta import Slice, Filter, hann
from src.metrics import  print_net_dict
import numpy as np
import os
import errno
from src.settings import save_as_pickle, load_pickle, save_net_dict
from src.preprocessing import time_shift_io_positions, shuffle_io, position_as_map
import datetime
import pickle
import multiprocessing
from external.preprocessing_funcs import get_spikes_with_history
from src.network_functions import  run_network
from scipy import stats, spatial


now = datetime.datetime.now().isoformat()

# Glaser data set
# MODEL_PATH = "G:/master_datafiles/trained_networks/MLP_Custom_2018-20-09/"
# RAW_DATA_PATH = "C:/Users/NN\Desktop/Neural_Decoding-master/example_data_hc.pickle"
# FILTERED_DATA_PATH = "G:/master_datafiles/filtered_data/neocortex_hann_win_size_100.pkl"

# prefrontal cortex

MODEL_PATH = "G:/master_datafiles/trained_networks/MLP_PFC_2018-10-10_1000_200_100/"
RAW_DATA_PATH = "G:/master_datafiles/raw_data/2018-04-09_14-39-52/"
FILTERED_DATA_PATH = "G:/master_datafiles/filtered_data/neocortex_hann_win_size_20.pkl"

# hippocampus

# MODEL_PATH = "G:/master_datafiles/trained_networks/MLP_HC_2018-10-19_1000_200_100/"
# RAW_DATA_PATH = "G:/master_datafiles/raw_data/2018-05-16_17-13-37/"
# FILTERED_DATA_PATH = "G:/master_datafiles/filtered_data/hippocampus_hann_win_size_25_09-5_7.pkl"
NEURONS_KEPT_FACTOR = 1

# Program execution settings

LOAD_RAW_DATA = True  # load source data from raw data path or load default model
# LOAD_RAW_DATA = True # load source data from raw data path or load default model
LOAD_GLASER_DATA = False
SAVE_FILTERED_DATA = True
MAKE_HISTOGRAM = False
LOAD_MODEL = False  # load model from model path
TRAIN_MODEL = True  # train model or just show results
EPOCHS = 30
INITIAL_TIMESHIFT = 0
TIME_SHIFT_ITER = 200
TIME_SHIFT_STEPS = 100
METRIC_ITER = 1  # after how many epochs network is validated <---
SHUFFLE_DATA = True  # whether to randomly shuffle the data in big slices
SHUFFLE_FACTOR = 500
EARLY_STOPPING = False
NAIVE_TEST = False # TODO
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


if __name__ == '__main__':

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
        iter = int(z - INITIAL_TIMESHIFT) // TIME_SHIFT_ITER
        net_dict["TIME_SHIFT"] = z

        if LOAD_GLASER_DATA is False:
            print("Time shift is now", z)

            # Time-Shift input and output

            X, y = time_shift_io_positions(session, z, net_dict)
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

        # net_dict["X_train"], net_dict["y_train"] = filter_overrepresentation(X[valid_length:], y[valid_length:], 70,
        #                                                                      net_dict, axis=0)
        # net_dict["X_valid"], net_dict["y_valid"] = filter_overrepresentation(X[:valid_length], y[:valid_length], 7,
        #                                                                      net_dict, axis=0)
        # train_oc = count_occurrences(net_dict["y_train"], net_dict)
        # valid_oc = count_occurrences(net_dict["y_valid"], net_dict)
        r2_score_k_valid = []
        avg_score_k_valid = []
        acc_score_k_valid = []
        r2_score_k_train = []
        avg_score_k_train = []
        acc_score_k_train = []

        for k in range (0,K_CROSS_VALIDATION):
            print("cross validation step",str(k+1),"of",K_CROSS_VALIDATION)
            if K_CROSS_VALIDATION == 1:
                valid_length = int(len(X) * VALID_RATIO)
                net_dict["X_train"] = X[valid_length:]
                net_dict["y_train"] = y[valid_length:]
                net_dict["X_valid"] = X[:valid_length//2]
                net_dict["y_valid"] = y[:valid_length//2]
                net_dict["X_test"] = X[valid_length//2:valid_length]
                net_dict["y_test"] = y[valid_length//2:valid_length]
            else:
                k_len = int(len(X)//K_CROSS_VALIDATION)
                k_slice_test = slice(k_len*k,int(k_len*(k+0.5)))
                k_slice_valid = slice(int(k_len*(k+0.5)),k_len*(k+1))
                not_k_slice_1 = slice(0,k_len*k)
                not_k_slice_2 = slice(k_len*(k+1),len(X))
                net_dict["X_train"] = X[not_k_slice_1] + X[not_k_slice_2]
                net_dict["y_train"] = y[not_k_slice_1] + y[not_k_slice_2]
                net_dict["X_test"] = X[k_slice_test]
                net_dict["y_test"] = y[k_slice_test]
                net_dict["X_valid"] = X[k_slice_valid]
                net_dict["y_valid"] = y[k_slice_valid]

            if TIME_SHIFT_STEPS == 1:
                save_dict = run_network(net_dict)
            else:
                with multiprocessing.Pool(
                        1) as p:  # keeping network inside process prevents memory issues when restarting session
                    save_dict = p.map(run_network, [net_dict])[0]
                    p.close()
            r2_score_k_valid.append(save_dict["r2_scores_valid"])
            acc_score_k_valid.append(save_dict["acc_scores_valid"])
            avg_score_k_valid.append(save_dict["avg_scores_valid"])
            r2_score_k_train.append(save_dict["r2_scores_train"])
            acc_score_k_train.append(save_dict["acc_scores_train"])
            avg_score_k_train.append(save_dict["avg_scores_train"])
        save_dict["r2_scores_valid"] = np.average(np.array(r2_score_k_valid),axis=1)
        save_dict["acc_scores_valid"] = np.average(np.array(acc_score_k_valid),axis=1)
        save_dict["avg_scores_valid"] = np.average(np.array(avg_score_k_valid),axis=1)
        # save_dict["r2_scores_train"] = np.average(np.array(r2_score_k_train),axis=0)
        # save_dict["acc_scores_train"] = np.average(np.array(acc_score_k_train),axis=0)
        # save_dict["avg_scores_train"] = np.average(np.array(avg_score_k_train),axis=0)

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
