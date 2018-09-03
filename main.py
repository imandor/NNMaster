import tensorflow as tf
from database_api_beta import Slice, Filter, hann
from src.nets import MultiLayerPerceptron, ConvolutionalNeuralNetwork1
from src.metrics import get_r2, get_avg_distance, bin_distance, get_accuracy, get_radius_accuracy
import numpy as np
from src.conf import mlp, sp2, cnn1
import matplotlib.pyplot as plt
from src.settings import save_as_pickle, load_pickle
from sklearn.preprocessing import normalize, scale
from random import shuffle
import datetime
import multiprocessing


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def test_accuracy(sess, S, network_dict, x, y, show_plot=False, plot_after_iter=1, print_distance=False):
    X_eval = x
    y_eval = y
    xshape = [1] + list(X_eval[0].shape) + [1]
    yshape = [1] + list(y_eval[0].shape) + [1]
    prediction_list = np.zeros([len(y_eval), 2])
    actual_list = np.zeros([len(y_eval), 2])
    for j in range(0, len(X_eval) - 1, 1):
        x = np.array([data_slice for data_slice in X_eval[j:j + 1]])
        y = np.array(y_eval[j:j + 1])
        x = np.reshape(x, xshape)
        y = np.reshape(y, yshape)
        # print(np.max(S.train(sess, x, y)))
        a = S.eval(sess, x)
        bin_1 = np.unravel_index(a.argmax(), a.shape)
        bin_2 = np.unravel_index(y.argmax(), y.shape)
        prediction_list[j][0] = bin_1[1]
        prediction_list[j][1] = bin_1[2]
        actual_list[j][0] = bin_2[1]
        actual_list[j][1] = bin_2[2]
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
            y_prime = S.eval(sess, x)
            # fig = plt.figure()
            fig = plt.gcf()
            fig.canvas.set_window_title(title)
            ax1 = fig.add_subplot(1, 2, 1)
            ax2 = fig.add_subplot(1, 2, 2)
            ax1.axis('off')
            ax2.axis('off')
            fig.subplots_adjust(left=0.01, right=0.99, bottom=0.01, top=0.99, hspace=0.01, wspace=0.01)
            Y = y[0, :, :, 0]
            Y -= np.min(Y)
            Y /= np.max(Y)
            Y *= 255

            Y_prime = sigmoid(y_prime[0, :, :, 0])
            Y_prime -= np.min(Y_prime)
            Y_prime /= np.max(Y_prime)
            Y_prime *= 255
            Y_prime[6, 9] = 30
            Y_prime[14, 9] = 30
            Y_prime[23, 9] = 30
            Y_prime[32, 9] = 30
            Y_prime[41, 9] = 30
            Y[6, 9] = 30
            Y[14, 9] = 30
            Y[23, 9] = 30
            Y[32, 9] = 30
            Y[41, 9] = 30
            ax1.imshow(Y, cmap="gray")
            ax2.imshow(Y_prime, cmap="gray")

            plt.show(block=True)
            plt.close()

    r2 = get_r2(prediction_list, actual_list)
    distance = get_avg_distance(prediction_list, actual_list, [network_dict["X_STEP"], network_dict["Y_STEP"]])
    accuracy = []
    for i in range(0, 5):
        # print("accuracy",i,":",get_accuracy(prediction_list, actual_list,margin=i))
        acc = get_radius_accuracy(prediction_list, actual_list, [network_dict["X_STEP"], network_dict["Y_STEP"]], 5 * i)
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
        ret[pos[0], pos[1]] += 1
    ret_shape = ret.shape
    ret = scale(ret.flatten(), axis=0).reshape(ret_shape)
    return ret


def shift(li, n):
    return li[n:] + li[:n]


def time_shift_data(X, y, n):
    y = shift(y, n)[n:]
    X = X[:len(X) - n]
    return X, y


def run_network(network_dict):
    # S = ConvolutionalNeuralNetwork1([None, 166, 20, 1], cnn1)
    S = MultiLayerPerceptron([None, 68, 20, 1], mlp)

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
                t = np.max(S.train(sess, x, y, dropout=1))
                # print(i, ", loss:", t)

                # Test accuracy and add evaluation to output

                if metric_counter == network_dict["METRIC_ITER"]:
                    print("Step", i)
                    print("training data:")
                    r2_train, avg_train, accuracy_train = test_accuracy(sess, S, network_dict, X_train, y_train,
                                                                        show_plot=False, plot_after_iter=100)
                    r2_scores_train.append(r2_train)
                    avg_scores_train.append(avg_train)
                    acc_scores_train.append(accuracy_train)
                    print(r2_train, avg_train)
                    print("evaluation data:")
                    r2_eval, avg_eval, accuracy_eval = test_accuracy(sess, S, network_dict, X_eval, y_eval,
                                                                     show_plot=False, plot_after_iter=1000)
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

    MODEL_PATH = "G:/master_datafiles/trained_networks/Special_CNN"
    RAW_DATA_PATH = "G:/master_datafiles/raw_data/2018-05-16_17-13-37/"
    FILTERED_DATA_PATH = "G:/master_datafiles/filtered_data/hippocampus_hann_win_size_10.pkl"

    # Program execution settings

    LOAD_RAW_DATA = False # load source data from raw data path or load default model
    LOAD_MODEL = False  # load model from model path if True
    TRAIN_MODEL = True  # train model if True or just show results if False
    SHUFFLE_DATA = False
    TRAINING_STEPS = 5000
    INITIAL_TIMESHIFT = 0
    TIME_SHIFT_ITER = 200
    TIME_SHIFT_STEPS = 1
    METRIC_ITER = 1

    # Network parameters

    SLICE_SIZE = 500
    BATCH_SIZE = 128
    WIN_SIZE = 25
    SEARCH_RADIUS = WIN_SIZE * 2
    X_MAX = 240
    Y_MAX = 190
    X_MIN = 0
    Y_MIN = 100
    X_STEP = 3
    Y_STEP = 3
    session_filter = Filter(func=hann, search_radius=SEARCH_RADIUS, step_size=WIN_SIZE)

    # Preprocess data

    if LOAD_RAW_DATA is True:
        # session = Slice.from_raw_data(RAW_DATA_PATH)
        # print("Convolving data...")
        # session.set_filter(session_filter)
        # print("Finished convolving data")
        # session.to_pickle("slice.pkl")  # TODO delete again
        session = Slice.from_pickle("slice.pkl")  # TODO delete again
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
            X.append(normalize(scale(data_slice.filtered_spikes,axis=1), norm='l2', axis=1))

            # map of shifted positions_list

            for i in range(len(shifted_positions_list)):
                posxy_list = [shifted_positions_list[i][0][:SLICE_SIZE], shifted_positions_list[i][1][:SLICE_SIZE]]
                y_list[i].append(position_as_map(posxy_list, X_STEP, Y_STEP, X_MAX, X_MIN, Y_MAX, Y_MIN))
                shifted_positions_list[i] = [shifted_positions_list[i][0][SLICE_SIZE:],
                                             shifted_positions_list[i][1][SLICE_SIZE:]]
            print("slicing", len(X), "of", len(session.position_x) // SLICE_SIZE)
        print("Finished slicing data")

        # Create dict for saving network data to file

        network_dict = dict(
            X = X,
            metadata = metadata,
            y_list = y_list,
            WIN_SIZE=WIN_SIZE,
            SEARCH_RADIUS=SEARCH_RADIUS,
            X_MAX=X_MAX,
            Y_MAX=Y_MAX,
            X_MIN=X_MIN,
            Y_MIN=Y_MIN,
            X_STEP=X_STEP,
            Y_STEP=Y_STEP,
            BATCH_SIZE=BATCH_SIZE,
            network_type="Multi Layer Perceptron",
            TRAINING_STEPS=TRAINING_STEPS,
            trained_steps=0,
            MODEL_PATH=MODEL_PATH,
            RAW_DATA_PATH=RAW_DATA_PATH,
            learning_rate="placeholder",  # TODO
            TIME_SHIFT=TIME_SHIFT_ITER * z,  # TODO
            r2_scores_train=[],
            r2_scores_eval=[],
            acc_scores_train=[],
            acc_scores_eval=[],
            avg_scores_train=[],
            avg_scores_eval=[],
            SLICE_SIZE=SLICE_SIZE,
            LOAD_MODEL=LOAD_MODEL,
            TRAIN_MODEL=TRAIN_MODEL,
            METRIC_ITER=METRIC_ITER,
        )
        save_as_pickle(FILTERED_DATA_PATH, network_dict)
    else:
        # Update network parameters dictionary

        network_dict = load_pickle(FILTERED_DATA_PATH)
        network_dict["network_type"] = "Multi Layer Perceptron"
        network_dict["TRAINING_STEPS"] = TRAINING_STEPS
        network_dict["MODEL_PATH"] = MODEL_PATH
        network_dict["BATCH_SIZE"] = BATCH_SIZE
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
        save_as_pickle(FILTERED_DATA_PATH, network_dict)

    X = network_dict["X"]
    y_list = network_dict["y_list"]
    metadata = network_dict["metadata"]

    if len(y_list) < TIME_SHIFT_STEPS:
        raise ValueError("Error: filtered data does not match amount of timeshift steps")

    # Shuffle data
    if SHUFFLE_DATA is True:
        shuffle_list = np.asarray(list(zip(X, metadata, zip(*y_list))))
        shuffle_list = shuffle_list.reshape(-1,3,55)
        shuffle(shuffle_list)
        shuffle_list = shuffle_list.reshape(-1,3)
        X, metadata, y_list = zip(*shuffle_list)
        for i,y in enumerate(y_list):
            y_list[i] = np.ndarray.tolist(y)
    eval_length = int(len(X) / 2)
    network_dict["X_eval"] = X[:eval_length]
    network_dict["metadata_eval"] = metadata[:eval_length]
    network_dict["X_train"] = X[eval_length:]
    network_dict["metadata_train"] = metadata[eval_length:]
    network_dict["eval_length"] = eval_length
    X_eval = X[:eval_length]
    metadata_eval = metadata[:eval_length]
    X_train = X[eval_length:]
    metadata_train = metadata[eval_length:]

    # Start network with different time-shifts


    for z in range(INITIAL_TIMESHIFT // TIME_SHIFT_ITER, TIME_SHIFT_STEPS):
        print("Time shift is now",z*TIME_SHIFT_ITER)
        network_dict["TIME_SHIFT"] = TIME_SHIFT_ITER * z  # TODO
        network_dict["y_eval"] = [a for a in y_list[z][network_dict["eval_length"]:]]
        network_dict["y_train"] = [a for a in y_list[z][:network_dict["eval_length"]]]
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
