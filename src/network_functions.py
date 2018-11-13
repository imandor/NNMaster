import tensorflow as tf
from src.nets import MultiLayerPerceptron
from src.metrics import test_accuracy, plot_histogram
from src.conf import mlp
from src.database_api_beta import Slice, Filter, hann
from src.metrics import print_Net_data
import numpy as np
from src.settings import save_as_pickle, load_pickle, save_net_dict
from src.preprocessing import time_shift_positions, shuffle_io, position_as_map,lickwells_io
import multiprocessing
import os
import errno
import datetime
from scipy import stats

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


def run_network_process(nd):
    # S = ConvolutionalNeuralNetwork1([None, 56, 10, 1], cnn1)
    S = MultiLayerPerceptron([None, nd.N_NEURONS, 50, 1], mlp)  # 56 147
    sess = tf.Session()
    r2_scores_train = []
    avg_scores_train = []
    r2_scores_valid = []
    avg_scores_valid = []
    acc_scores_train = []
    acc_scores_valid = []
    early_stop_max = -np.inf
    X_train = nd.X_train
    y_train = nd.y_train
    saver = tf.train.Saver()
    if nd.LOAD_MODEL is True:
        saver = tf.train.import_meta_graph(nd.MODEL_PATH + ".meta")
        saver.restore(sess, nd.MODEL_PATH)
    else:
        S.initialize(sess)

    # Train model

    sess.run(tf.local_variables_initializer())
    print("Training model...")
    xshape = [nd.BATCH_SIZE] + list(X_train[0].shape) + [1]
    yshape = [nd.BATCH_SIZE] + list(y_train[0].shape) + [1]
    metric_counter = 0  # net_dict["METRIC_ITER"]
    metric_step_counter = []
    stop_early = False
    for i in range(0, nd.EPOCHS + 1):
        X_train, y_train = shuffle_io(X_train, y_train, nd, i + 1)
        if metric_counter == nd.METRIC_ITER and stop_early is False:
            metric_step_counter.append(i)
            if nd.NAIVE_TEST is True:
                saver.save(sess, nd.MODEL_PATH)

            print("\n_-_-_-_-_-_-_-_-_-_-Epoch", i, "_-_-_-_-_-_-_-_-_-_-\n")

            print("Validation results:")
            r2_valid, avg_valid, acc_valid = test_accuracy(sess=sess, S=S, nd=nd, X=nd.X_valid, y=nd.y_valid, epoch=i,
                                                           print_distance=True)
            r2_scores_valid.append(r2_valid)
            avg_scores_valid.append(avg_valid)
            acc_scores_valid.append(acc_valid)
            print("R2-score:", r2_valid)
            print("Avg-distance:", avg_valid)
            # saver.save(sess, nd.MODEL_PATH, global_step=i, write_meta_graph=False)
            metric_counter = 0
            if nd.EARLY_STOPPING is True and i >= 5:  # most likely overfitting instead of training
                if i % 1 == 0:

                    _, _, acc_test = test_accuracy(sess=sess, S=S, nd=nd, X=nd.X_test, y=nd.y_test, epoch=i,
                                                           print_distance=False)
                    acc = acc_test[19]
                    if early_stop_max < acc:
                        early_stop_max = acc
                    else:
                        if acc < early_stop_max - 0.01:
                            stop_early = True
                            r2_scores_train = r2_scores_train[0:-1]  # Remove latest result which was worse
                            avg_scores_train = avg_scores_train[0:-1]
                            acc_scores_train = acc_scores_train[0:-1]
                            r2_scores_valid = r2_scores_valid[0:-1]
                            avg_scores_valid = avg_scores_valid[0:-1]
                            acc_scores_valid = acc_scores_valid[0:-1]

                            # metric_counter = net_dict["METRIC_ITER"]  # one last calculation with local maximum return

            # a = get_radius_accuracy(prediction_list, actual_list, [network_dict["X_STEP"], network_dict["Y_STEP"]], 19)
        for j in range(0, len(X_train) - nd.BATCH_SIZE, nd.BATCH_SIZE):
            x = np.array([data_slice for data_slice in X_train[j:j + nd.BATCH_SIZE]])
            y = np.array(y_train[j:j + nd.BATCH_SIZE])
            x = np.reshape(x, xshape)
            y = np.reshape(y, yshape)
            if nd.NAIVE_TEST is False or nd.TIME_SHIFT == 0:
                t = np.max(S.train(sess, x, y, dropout=0.65))

        metric_counter = metric_counter + 1
        nd.epochs_trained = i
        if stop_early is True:
            break
    # saver.save(sess, net_dict["MODEL_PATH"])

    # Add performance to return dict

    nd.r2_scores_train = r2_scores_train
    nd.r2_scores_valid = r2_scores_valid
    nd.acc_scores_train = acc_scores_train
    nd.acc_scores_valid = acc_scores_valid
    nd.avg_scores_train = avg_scores_train
    nd.avg_scores_valid = avg_scores_valid
    nd.metric_step_counter = metric_step_counter
    # Close session and add current time shift to network save file
    if nd.MAKE_HISTOGRAM is True:
        plot_histogram(sess, S, nd, nd.X_valid, nd.y_valid)
        plot_histogram(sess, S, nd, nd.X_train, nd.y_train)
    sess.close()
    return nd


def initiate_network(nd):
    try:
        os.makedirs(os.path.dirname(nd.MODEL_PATH))
        os.makedirs(os.path.dirname(nd.MODEL_PATH + "output/"))
        os.makedirs(os.path.dirname(nd.MODEL_PATH + "images/"))
    except OSError as exc:  # Guard against race condition
        if exc.errno != errno.EEXIST:
            raise
    # TODO
    # session = Slice.from_raw_data(nd.RAW_DATA_PATH)
    # session.filter_neurons(100)
    # session.print_details()
    # print("Convolving data...")
    # session.set_filter(nd.session_filter)
    # print("Finished convolving data")
    # session.filtered_spikes = stats.zscore(session.filtered_spikes, axis=1)  # Z Score neural activity
    # session.to_pickle("slice_OFC.pkl")
    # TODO
    session = Slice.from_pickle("slice_OFC.pkl")
    session.filter_neurons_randomly(nd.NEURONS_KEPT_FACTOR)
    session.print_details()
    nd.N_NEURONS = session.n_neurons

    # Start network with different time-shifts

    print_Net_data(nd)  # show network parameters in console
    X, y = time_shift_positions(session, 0, nd)
    return X, y, session


def run_network(X, y, nd, session):
    # Assign validation set.
    for z in range(nd.INITIAL_TIMESHIFT, nd.INITIAL_TIMESHIFT + nd.TIME_SHIFT_STEPS * nd.TIME_SHIFT_ITER,
                   nd.TIME_SHIFT_ITER):
        iter = int(z - nd.INITIAL_TIMESHIFT) // nd.TIME_SHIFT_ITER
        nd.TIME_SHIFT = z  # set current time shift
        print("Time shift is now", z)
        # Time-Shift input and output
        X, y = time_shift_positions(session, z, nd)
        if len(X) != len(y):
            raise ValueError("Error: Length of x and y are not identical")
        X, y = shuffle_io(X, y, nd, 3)

        r2_score_k_valid = []
        avg_score_k_valid = []
        acc_score_k_valid = []
        r2_score_k_train = []
        avg_score_k_train = []
        acc_score_k_train = []

        for k in range(0, nd.K_CROSS_VALIDATION):
            print("cross validation step", str(k + 1), "of", nd.K_CROSS_VALIDATION)
            nd.split_data(X, y, k)
            if nd.TIME_SHIFT_STEPS == 1:
                save_nd = run_network_process(nd)
            else:
                with multiprocessing.Pool(
                        1) as p:  # keeping network inside process prevents memory issues when restarting session
                    save_nd = p.map(run_network_process, [nd])[0]
                    p.close()
                if nd.NAIVE_TEST is True:
                    nd.LOAD_MODEL = True
                    nd.EPOCHS = 1
            r2_score_k_valid.append(save_nd.r2_scores_valid)
            acc_score_k_valid.append(save_nd.acc_scores_valid)
            avg_score_k_valid.append(save_nd.avg_scores_valid)
            r2_score_k_train.append(save_nd.r2_scores_train)
            acc_score_k_train.append(save_nd.acc_scores_train)
            avg_score_k_train.append(save_nd.avg_scores_train)

        save_dict = create_save_dict(save_nd,z)
        if k == 1:
            save_dict["r2_scores_valid"] = np.average(np.array(r2_score_k_valid), axis=1)
            save_dict["acc_scores_valid"] = np.average(np.array(acc_score_k_valid), axis=1)
            save_dict["avg_scores_valid"] = np.average(np.array(avg_score_k_valid), axis=1)
        now = datetime.datetime.now().isoformat()
        path = nd.MODEL_PATH + "output/" + chr(65 + iter) + "_" + now[0:10] + "_network_output_timeshift=" + str(
            z) + ".pkl"
        save_as_pickle(path, save_dict)
    print("fin")


def run_lickwell_network(X, y, nd, session):
    # Assign validation set.
    for z in range(nd.INITIAL_TIMESHIFT, nd.INITIAL_TIMESHIFT + nd.TIME_SHIFT_STEPS * nd.TIME_SHIFT_ITER,
                   nd.TIME_SHIFT_ITER):
        iter = int(z - nd.INITIAL_TIMESHIFT) // nd.TIME_SHIFT_ITER
        nd.TIME_SHIFT = z  # set current time shift
        print("Time shift is now", z)
        # Time-Shift input and output
        X, y = lickwells_io(session, z, nd)
        if len(X) != len(y):
            raise ValueError("Error: Length of x and y are not identical")
        X, y = shuffle_io(X, y, nd, 3)

        r2_score_k_valid = []
        avg_score_k_valid = []
        acc_score_k_valid = []
        r2_score_k_train = []
        avg_score_k_train = []
        acc_score_k_train = []

        for k in range(0, nd.K_CROSS_VALIDATION):
            print("cross validation step", str(k + 1), "of", nd.K_CROSS_VALIDATION)
            nd.split_data(X, y, k)
            if nd.TIME_SHIFT_STEPS == 1:
                save_nd = run_network_process(nd)
            else:
                with multiprocessing.Pool(
                        1) as p:  # keeping network inside process prevents memory issues when restarting session
                    save_nd = p.map(run_lickwell_network_process, [nd])[0]
                    p.close()
                if nd.NAIVE_TEST is True:
                    nd.LOAD_MODEL = True
                    nd.EPOCHS = 1
            r2_score_k_valid.append(save_nd.r2_scores_valid)
            acc_score_k_valid.append(save_nd.acc_scores_valid)
            avg_score_k_valid.append(save_nd.avg_scores_valid)
            r2_score_k_train.append(save_nd.r2_scores_train)
            acc_score_k_train.append(save_nd.acc_scores_train)
            avg_score_k_train.append(save_nd.avg_scores_train)

        save_dict = create_save_dict(save_nd, z)
        if k == 1:
            save_dict["r2_scores_valid"] = np.average(np.array(r2_score_k_valid), axis=1)
            save_dict["acc_scores_valid"] = np.average(np.array(acc_score_k_valid), axis=1)
            save_dict["avg_scores_valid"] = np.average(np.array(avg_score_k_valid), axis=1)
        now = datetime.datetime.now().isoformat()
        path = nd.MODEL_PATH + "output/" + chr(65 + iter) + "_" + now[0:10] + "_network_output_timeshift=" + str(
            z) + ".pkl"
        save_as_pickle(path, save_dict)
    print("fin")
def create_save_dict(save_nd,z):
    save_dict = dict()
    save_dict["MAKE_HISTOGRAM"] = save_nd.MAKE_HISTOGRAM
    save_dict["STRIDE"] = save_nd.STRIDE
    save_dict["Y_SLICE_SIZE"] = save_nd.Y_SLICE_SIZE
    save_dict["network_type"] = save_nd.network_type
    save_dict["EPOCHS"] = save_nd.EPOCHS
    save_dict["session_filter"] = save_nd.session_filter
    save_dict["TIME_SHIFT_STEPS"] = save_nd.TIME_SHIFT_STEPS
    save_dict["SHUFFLE_DATA"] = save_nd.SHUFFLE_DATA
    save_dict["SHUFFLE_FACTOR"] = save_nd.SHUFFLE_FACTOR
    save_dict["TIME_SHIFT_ITER"] = save_nd.TIME_SHIFT_ITER
    save_dict["MODEL_PATH"] = save_nd.MODEL_PATH
    save_dict["learning_rate"] = "placeholder"  # TODO
    save_dict["r2_scores_train"] = save_nd.r2_scores_train
    save_dict["r2_scores_valid"] = save_nd.r2_scores_valid
    save_dict["acc_scores_train"] = save_nd.acc_scores_train
    save_dict["acc_scores_valid"] = save_nd.acc_scores_valid
    save_dict["avg_scores_train"] = save_nd.avg_scores_train
    save_dict["avg_scores_valid"] = save_nd.avg_scores_valid
    save_dict["LOAD_MODEL"] = save_nd.LOAD_MODEL
    save_dict["INITIAL_TIMESHIFT"] = save_nd.INITIAL_TIMESHIFT
    save_dict["TRAIN_MODEL"] = save_nd.TRAIN_MODEL
    save_dict["METRIC_ITER"] = save_nd.METRIC_ITER
    save_dict["BATCH_SIZE"] = save_nd.BATCH_SIZE
    save_dict["SLICE_SIZE"] = save_nd.SLICE_SIZE
    save_dict["RAW_DATA_PATH"] = save_nd.RAW_DATA_PATH
    save_dict["X_MAX"] = save_nd.X_MAX
    save_dict["Y_MAX"] = save_nd.Y_MAX
    save_dict["X_MIN"] = save_nd.X_MIN
    save_dict["Y_MIN"] = save_nd.Y_MIN
    save_dict["X_STEP"] = save_nd.X_STEP
    save_dict["Y_STEP"] = save_nd.Y_STEP
    save_dict["WIN_SIZE"] = save_nd.WIN_SIZE
    save_dict["SEARCH_RADIUS"] = save_nd.SEARCH_RADIUS
    save_dict["EARLY_STOPPING"] = save_nd.EARLY_STOPPING
    save_dict["NAIVE_TEST"] = save_nd.NAIVE_TEST
    save_dict["TIME_SHIFT"] = z
    return save_dict