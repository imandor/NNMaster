import tensorflow as tf
from src.nets import MultiLayerPerceptron
from src.metrics import test_accuracy, plot_histogram
import numpy as np
from src.conf import mlp
from src.preprocessing import shuffle_io


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
        saver.restore(sess, net_dict["MODEL_PATH"])
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
            print("Training results:")
            if False:
                r2_train, avg_train, accuracy_train = test_accuracy(sess, S, net_dict, net_dict["X_train"], net_dict["y_train"],
                                                                    print_distance=True)
                r2_scores_train.append(r2_train)
                avg_scores_train.append(avg_train)
                acc_scores_train.append(accuracy_train)
                print("R2-score:",r2_train)
                print("Avg-distance:",avg_train,"\n")

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

                    _,_,acc_test = test_accuracy(sess, S, net_dict, net_dict["X_test"],net_dict["y_test"],
                                                                   print_distance=False)
                    acc = acc_test[19]
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


def predict_lickwell(net_dict):
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
        saver.restore(sess, net_dict["MODEL_PATH"])
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
            print("Training results:")
            if False:
                r2_train, avg_train, accuracy_train = test_accuracy(sess, S, net_dict, net_dict["X_train"], net_dict["y_train"],
                                                                    print_distance=True)
                r2_scores_train.append(r2_train)
                avg_scores_train.append(avg_train)
                acc_scores_train.append(accuracy_train)
                print("R2-score:",r2_train)
                print("Avg-distance:",avg_train,"\n")

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

                    _,_,acc_test = test_accuracy(sess, S, net_dict, net_dict["X_test"],net_dict["y_test"],
                                                                   print_distance=False)
                    acc = acc_test[19]
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