import tensorflow as tf
from database_api_beta import Slice, Filter, hann,bin
from src.nets import MultiLayerPerceptron, ConvolutionalNeuralNetwork1
from src.metrics import get_r2, get_avg_distance, bin_distance, get_accuracy, get_radius_accuracy
import numpy as np
from src.conf import mlp,sp1,cnn1
import matplotlib.pyplot as plt
from src.settings import save_as_pickle, load_pickle
from sklearn.preprocessing import normalize,scale
from random import shuffle, seed
import datetime
import multiprocessing
from scipy import stats
from functools import reduce

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def average_position(mapping):
    # mapping = np.reshape(mapping,[mapping.shape[0],mapping.shape[1]])
    # sum_0 = np.dot(mapping,np.arange(mapping.shape[1]))
    # sum_1 = np.dot(mapping.T,np.arange(mapping.shape[0]))

    sum_0 = np.sum(mapping,axis=0)
    sum_0 = np.sum(np.sum(sum_0*np.arange(mapping.shape[1])))/np.sum(np.sum(sum_0))
    sum_1 = np.sum(mapping,axis=1)
    sum_1  = np.sum(np.sum(sum_1*np.arange(mapping.shape[0])))/np.sum(np.sum(sum_1))
    try:
        return [int(sum_1),int(sum_0)]
    except ValueError:
        print("asd")
        return False



def test_accuracy(sess, S, network_dict, X_eval, y_eval, show_plot=False, plot_after_iter=1, print_distance=False):
    xshape = [1] + list(X_eval[0].shape) + [1]
    yshape = [1] + list(y_eval[0].shape) + [1]
    prediction_list = np.zeros([len(y_eval), 2])
    actual_list = np.zeros([len(y_eval), 2])
    for j in range(0, len(X_eval) - 1, 1):
        x = np.array([data_slice for data_slice in X_eval[j:j + 1]])
        y = np.array(y_eval[j:j + 1])
        x = np.reshape(x, xshape)
        y = np.reshape(y, yshape)
        a = S.eval(sess, x)
        # form softmax of output and remove nan values for columns with only zeros
        # exp_scores = np.exp(a)
        # a = np.nan_to_num(exp_scores / np.sum(exp_scores, axis=1, keepdims=True))
        a = sigmoid(a[0, :, :, 0])
        y = y[0,:,:,0]
        bin_1 = average_position(a)
        bin_2 = average_position(y)
        prediction_list[j][0] = bin_1[0]
        prediction_list[j][1] = bin_1[1]
        actual_list[j][0] = bin_2[0]
        actual_list[j][1] = bin_2[1]
        if print_distance is True:
            print("prediction:", bin_1)
            print("actual:    ", bin_2)
            print("distance:", bin_distance(bin_1, bin_2))
            print("_____________")
        if j % plot_after_iter == plot_after_iter - 1 and show_plot is True:
            print("plot")
            time = "{:.1f}".format(network_dict["metadata"][j]["lickwells"]["time"] / 1000)
            well = network_dict["metadata"][j]["lickwells"]
            title = "Next lickwell: " + str(well["lickwell"]) + " (" + str(
                well["rewarded"]) + ") in " + time + "s" + " at " + str(network_dict["metadata"][j]["position"]) + "cm" + " and " + str(
                network_dict["metadata"][j]["time"])
            y_prime = a
            # fig = plt.figure()
            fig = plt.gcf()
            fig.canvas.set_window_title(title)
            ax1 = fig.add_subplot(1, 2, 1)
            ax2 = fig.add_subplot(1, 2, 2)
            ax1.axis('off')
            ax2.axis('off')
            fig.subplots_adjust(left=0.01, right=0.99, bottom=0.01, top=0.99, hspace=0.01, wspace=0.01)
            Y = y
            # Y -= np.min(Y)
            # Y /= np.max(Y)
            Y *= 255

            Y_prime = a
            Y_prime -= np.min(Y_prime)
            Y_prime /= np.max(Y_prime)
            Y_prime *= 255
            # Y_prime[6, 9] = 30
            # Y_prime[14, 9] = 30
            # Y_prime[23, 9] = 30
            # Y_prime[32, 9] = 30
            # Y_prime[41, 9] = 30
            # Y[6, 9] = 30
            # Y[14, 9] = 30
            # Y[23, 9] = 30
            # Y[32, 9] = 30
            # Y[41, 9] = 30
            ax1.imshow(Y, cmap="gray")
            ax2.imshow(Y_prime, cmap="gray")

            plt.show()
            plt.close()

    r2 = get_r2(actual_list,prediction_list)
    distance = get_avg_distance(prediction_list, actual_list, [network_dict["X_STEP"], network_dict["Y_STEP"]])
    accuracy = []
    for i in range(0, 20):
        # print("accuracy",i,":",get_accuracy(prediction_list, actual_list,margin=i))
        acc = get_radius_accuracy(prediction_list, actual_list, [network_dict["X_STEP"], network_dict["Y_STEP"]],  i)
        accuracy.append(acc)
        print("accuracy", i, ":", acc)

    return r2, distance, accuracy


def hann_generator(width):
    def hann(x):
        if np.abs(x) < width:
            return (1 + np.cos(x / width * np.pi)) / 2
        else:
            return 0

    return hann


def bin_filter(x):
    return 1


def position_as_map(pos_list, xstep, ystep, X_MAX, X_MIN, Y_MAX, Y_MIN):
    pos_list = np.asarray(pos_list)
    x_list = pos_list[0, :]
    y_list = pos_list[1, :]
    x_list = ((x_list - X_MIN) // xstep).astype(int)
    y_list = ((y_list - Y_MIN) // ystep).astype(int)
    pos_list = np.dstack((x_list, y_list))[0]
    pos_list = np.unique(pos_list, axis=0)
    ret = np.zeros(((X_MAX - X_MIN) // xstep, (Y_MAX - Y_MIN) // ystep))
    for pos in pos_list:
        ret[pos[0], pos[1]] = 1
    return ret


def shift(li, n):
    return li[n:] + li[:n]


def time_shift_data(X, y, n):
    y = shift(y, n)[n:]
    X = X[:len(X) - n]
    return X, y


def run_network(network_dict):
    # S = ConvolutionalNeuralNetwork1([None, 166, 20, 1], cnn1)
    S = MultiLayerPerceptron([None, 56, 10, 1], mlp)

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
        for i in range(network_dict["TRAINING_STEPS"]):
            for j in range(0, len(X_train) - network_dict["BATCH_SIZE"], network_dict["BATCH_SIZE"]):
                x = np.array([data_slice for data_slice in X_train[j:j + network_dict["BATCH_SIZE"]]])
                y = np.array(y_train[j:j + network_dict["BATCH_SIZE"]])
                x = np.reshape(x, xshape)
                y = np.reshape(y, yshape)
                t = np.max(S.train(sess, x, y, dropout=0.8))
                print(i, ", loss:", t)

                # Test accuracy and add evaluation to output

                if metric_counter == network_dict["METRIC_ITER"]:
                    saver.save(sess, network_dict["MODEL_PATH"])
                    print("Step", i)
                    print("training data:")
                    r2_train, avg_train, accuracy_train = test_accuracy(sess, S, network_dict, X_train, y_train,
                                                                        show_plot=True, plot_after_iter=500,print_distance=False)
                    r2_scores_train.append(r2_train)
                    avg_scores_train.append(avg_train)
                    acc_scores_train.append(accuracy_train)
                    print(r2_train, avg_train)
                    print("evaluation data:")
                    r2_eval, avg_eval, accuracy_eval = test_accuracy(sess, S, network_dict, X_eval, y_eval,
                                                                     show_plot=True, plot_after_iter=500,print_distance=False)
                    r2_scores_eval.append(r2_eval)
                    avg_scores_eval.append(avg_eval)
                    acc_scores_eval.append(accuracy_eval)
                    print(r2_eval, avg_eval)
                    saver.save(sess, network_dict["MODEL_PATH"], global_step=i, write_meta_graph=False)
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
    network_dict["trained_steps"] = network_dict["trained_steps"] + network_dict["TRAINING_STEPS"]
    # Test model performance

    # print(test_accuracy(X_train, y_train, metadata_train, show_plot=True, plot_after_iter=2))
    print(test_accuracy(sess, S, network_dict, X_eval, y_eval, show_plot=False, plot_after_iter=2))

    # Close session and add time shift to training data

    sess.close()
    return dict(network_dict)


if __name__ == '__main__':
    now = datetime.datetime.now().isoformat()

    # neo cortex

    # MODEL_PATH = "G:/master_datafiles/trained_networks/Special_CNN_hippocampus"
    # RAW_DATA_PATH = "G:/master_datafiles/raw_data/2018-04-09_14-39-52/"
    # FILTERED_DATA_PATH = "G:/master_datafiles/filtered_data/neocortex_hann_win_size_100.pkl"

    # hippocampus

    MODEL_PATH = "G:/master_datafiles/trained_networks/MLP_hippocampus"
    RAW_DATA_PATH = "G:/master_datafiles/raw_data/2018-05-16_17-13-37/"
    FILTERED_DATA_PATH = "G:/master_datafiles/filtered_data/hippocampus_hann_win_size_25_09-5_6.pkl"

    # Program execution settings
    LOAD_RAW_DATA = False # load source data from raw data path or load default model
    # LOAD_RAW_DATA = True # load source data from raw data path or load default model
    LOAD_MODEL = False  # load model from model path
    TRAIN_MODEL = True  # train model or just show results
    TRAINING_STEPS = 50000
    INITIAL_TIMESHIFT = 0
    TIME_SHIFT_ITER = 200
    TIME_SHIFT_STEPS = 1
    METRIC_ITER = 100
    SHUFFLE_DATA = True
    SHUFFLE_FACTOR = 5
    # Network parameters

    SLICE_SIZE = 2000
    BATCH_SIZE = 128
    WIN_SIZE = 200
    SEARCH_RADIUS = WIN_SIZE * 2
    EVAL_RATIO = 0.1
    BINS_BEFORE = 0
    BINS_AFTER = 0
    X_MAX = 240
    Y_MAX = 190
    X_MIN = 0
    Y_MIN = 100
    X_STEP = 3
    Y_STEP = 3

    # Update network dict values

    if LOAD_RAW_DATA is False:
        network_dict = load_pickle(FILTERED_DATA_PATH)
    else:
        network_dict = dict()
    network_dict["network_type"] = "Multi Layer Perceptron"
    network_dict["TRAINING_STEPS"] = TRAINING_STEPS
    network_dict["MODEL_PATH"] = MODEL_PATH
    network_dict["learning_rate"] = "placeholder"  # TODO
    network_dict["r2_scores_train"] = []
    network_dict["r2_scores_eval"] = []
    network_dict["acc_scores_train"] = []
    network_dict["acc_scores_eval"] = []
    network_dict["avg_scores_train"] = []
    network_dict["avg_scores_eval"] = []
    network_dict["LOAD_MODEL"] = LOAD_MODEL
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

    session_filter = Filter(func=bin, search_radius=SEARCH_RADIUS, step_size=WIN_SIZE)

    # Preprocess data

    if LOAD_RAW_DATA is True:
        session = Slice.from_raw_data(RAW_DATA_PATH)
        session.neuron_filter(100)
        print("Convolving data...")
        session.set_filter(session_filter)
        print("Finished convolving data")
        # session.filtered_spikes = stats.zscore(session.filtered_spikes,axis=1)
        session.to_pickle("slice.pkl")
        session = Slice.from_pickle("slice.pkl")
        shifted_positions_list = []
        copy_session = None

        # list of outputs for all time shifts

        for z in range(0, TIME_SHIFT_STEPS):
            copy_session = session.timeshift_position(z * TIME_SHIFT_ITER)
            shifted_positions_list.append(
                [copy_session.position_x, copy_session.position_y])

        X = []
        y_list = [[] for i in range(len(shifted_positions_list))]
        metadata = []

        # bin and normalize input and create metadata

        while len(copy_session.position_x) >= SLICE_SIZE:
            try:
                metadata.append(dict(lickwells=copy_session.licks[0], time=copy_session.absolute_time_offset,
                                     position=copy_session.position_x[0]))
            except:
                metadata.append(dict(rewarded=0, time=0, lickwell=0))  # no corresponding y-value

            data_slice = copy_session[0:SLICE_SIZE]
            copy_session = copy_session[SLICE_SIZE:]
            X.append(data_slice.filtered_spikes)
            # map of shifted positions_list

            for i in range(len(shifted_positions_list)):
                posxy_list = [shifted_positions_list[i][0][:SLICE_SIZE], shifted_positions_list[i][1][:SLICE_SIZE]]
                y_list[i].append(position_as_map(posxy_list, X_STEP, Y_STEP, X_MAX, X_MIN, Y_MAX, Y_MIN))
                shifted_positions_list[i] = [shifted_positions_list[i][0][SLICE_SIZE:],
                                             shifted_positions_list[i][1][SLICE_SIZE:]]
            print("slicing", len(X), "of", len(session.position_x) // SLICE_SIZE)
        print("Finished slicing data")



        # Increase range of X values

        if BINS_AFTER != 0 or BINS_BEFORE != 0:

            # crop unusable values
            X_c = X[BINS_BEFORE:-BINS_AFTER]
            y_list = [a for a in (np.array(y_list)[:,BINS_BEFORE:-BINS_AFTER])]
            metadata = metadata[BINS_BEFORE:-BINS_AFTER]
            # increase x-range

            for i, x in enumerate(X_c):
                X_c[i] = np.concatenate([a for a in X[i:i + BINS_BEFORE + BINS_AFTER + 1]], axis=1)
            X = X_c

        # Shuffle data
        seed(2)
        if SHUFFLE_DATA is True:

            # crop length to fit shuffle factor

            x_length = len(X) - (len(X) % SHUFFLE_FACTOR)
            X = X[:x_length]
            metadata = metadata[:x_length]
            y_list = [y[:x_length] for y in y_list]

            # Shuffle index of data

            r = np.arange(len(X))
            r = r.reshape(-1, SHUFFLE_FACTOR)
            s = np.arange(len(r)) # shuffling r directly doesnt work
            shuffle(s)
            r = r[s]
            r = r.reshape(-1)

            # shuffle data

            X = [X[j] for j in r]
            metadata = [metadata[j] for j in r]
            li = []
            for y in y_list:
                li.append([y[j] for j in r])
            y_list = li


        network_dict["X"]  = X
        network_dict["y_list"] = y_list
        network_dict["metadata"] = metadata
        network_dict["trained_steps"] = 0
        save_as_pickle(FILTERED_DATA_PATH, network_dict)
    y_list = network_dict["y_list"]

    eval_length = int(len(network_dict["X"])*EVAL_RATIO)
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
        raise ValueError("Error: filtered data does not match amount of timeshift steps")

    # Start network with different time-shifts

    for z in range(INITIAL_TIMESHIFT // TIME_SHIFT_ITER, TIME_SHIFT_STEPS):
        print("Time shift is now",z*TIME_SHIFT_ITER)
        network_dict["TIME_SHIFT"] = TIME_SHIFT_ITER * z
        current_y = y_list[z]
        network_dict["y_eval"] = current_y[:network_dict["eval_length"]]
        network_dict["y_train"] = current_y[network_dict["eval_length"]:]
        if TIME_SHIFT_STEPS == 1:
            save_dict = run_network(network_dict)
        else:
            with multiprocessing.Pool(1) as p: # keeping network inside process prevents memory issues when restarting session
                save_dict = p.map(run_network, [network_dict])[0]
                p.close()

        # Save to file


        save_dict.pop('X',None)
        save_dict.pop('X_train',None)
        save_dict.pop('X_eval',None)
        save_dict.pop('y',None)
        save_dict.pop('y_eval',None)
        save_dict.pop('y_train', None)
        save_dict.pop('metadata_train',None)
        save_dict.pop('metadata_eval',None)
        save_dict.pop('y_list',None)
        save_dict.pop('metadata',None)

        path = MODEL_PATH + "/" + now[0:9] + "_network_output_timeshift=" + str(z * TIME_SHIFT_ITER) + ".pkl"
        save_as_pickle(path, save_dict)
    print("fin")
