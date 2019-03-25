import tensorflow as tf
from src.nets import MultiLayerPerceptron
from src.metrics import  plot_histogram, Metric, Network_output,print_metric_details
from src.conf import mlp, mlp_discrete
from src.database_api_beta import Slice, Filter, hann
from src.metrics import print_Net_data, cross_validate_lickwell_data, Evaluated_Samples_By_Epoch,print_lickwell_metrics
import numpy as np
from src.settings import save_as_pickle, load_pickle, save_net_dict
from src.preprocessing import time_shift_positions, shuffle_io,shuffle_list_key, position_as_map, lickwells_io, generate_counter, \
    abs_to_logits,filter_behavior_component
import multiprocessing
import os
import errno
import datetime
import copy
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


def licks_to_logits(licks, num_classifiers,max_well_no):
    return_list = []
    for lick in licks:
        return_list.append(abs_to_logits(lick.target, num_classifiers,max_well_no))
    return return_list


def run_lickwell_network_process(nd):
    # Initialize session, parameters and network
    ns = mlp_discrete
    ns.fc1.weights = tf.truncated_normal(shape=(nd.number_of_bins*nd.n_neurons,100), stddev=0.01) # 36, 56, 75, 111, 147
    S = MultiLayerPerceptron([None, nd.n_neurons, nd.number_of_bins, 1], ns)
    sess = tf.Session()
    acc_scores_valid = []
    X_train = nd.X_train
    logits_train = licks_to_logits(nd.y_train, nd.lw_classifications,nd.num_wells)
    X_valid = nd.X_valid
    logits_valid = licks_to_logits(nd.y_valid, nd.lw_classifications,nd.num_wells)
    nd.x_shape = [nd.batch_size] + list(X_train[0].shape) + [1]
    nd.y_shape = [nd.batch_size] + [nd.lw_classifications]
    if nd.load_model is True:
        saver = tf.train.import_meta_graph(nd.model_path + ".meta")
        saver.restore(sess, nd.model_path)
    else:
        S.initialize(sess)
    # saver = tf.train.Saver()
    sess.run(tf.local_variables_initializer())
    print("Training model...")
    metric_counter = 0
    metric_step_counter = []
    for i in range(0, nd.epochs + 1):
        X_train, logits_train = shuffle_io(X_train, logits_train, nd, seed_no=i + 1)

        if metric_counter == nd.metric_iter:
            metric_step_counter.append(i)

            # Evaluate

            # print("Epoch", i, "of",nd.epochs)
            if nd.evaluate_training is True: # print training performance
                epoch_metric = Evaluated_Samples_By_Epoch.test_accuracy(sess=sess, S=S, nd=nd, X=X_train, y=logits_train,
                                                                        metadata=nd.y_train, epoch=i)

            epoch_metric = Evaluated_Samples_By_Epoch.test_accuracy(sess=sess, S=S, nd=nd, X=X_valid, y=logits_valid,
                                                                    metadata=nd.y_valid, epoch=i)
            acc_scores_valid.append(epoch_metric)
            metric_counter = 0

        # Train network

        for j in range(0, len(X_train) - nd.batch_size, nd.batch_size):
            x = np.array([data_slice for data_slice in X_train[j:j + nd.batch_size]])
            y = np.array(logits_train[j:j + nd.batch_size])
            x = np.reshape(x, nd.x_shape)
            y = np.reshape(y, nd.y_shape)
            if nd.naive_test is False or nd.time_shift == 0:
                t = np.max(S.train(sess, x, y, dropout=nd.dropout))

        metric_counter = metric_counter + 1
        nd.epochs_trained = i

    # Add performance to return dict
    nd.acc_scores_valid = acc_scores_valid
    nd.metric_step_counter = metric_step_counter
    # Close session and add current time shift to network save file

    sess.close()
    return acc_scores_valid


def run_network_process(nd):
    # S = ConvolutionalNeuralNetwork1([None, 56, 10, 1], cnn1)
    ns = mlp
    ns.fc1.weights = tf.truncated_normal(shape=(nd.number_of_bins*nd.n_neurons,100), stddev=0.01) # 36, 56, 75, 111, 147
    S = MultiLayerPerceptron([None, nd.n_neurons, 10, 1], ns)  # 56 147
    sess = tf.Session()
    metric_eval = Metric(r2_by_epoch=[], ape_by_epoch=[], acc20_by_epoch=[], r2_best=None,
                         ape_best=None, acc20_best=None)
    metric_test = Metric(r2_by_epoch=[], ape_by_epoch=[], acc20_by_epoch=[], r2_best=None,
                         ape_best=None, acc20_best=None)
    r2_max = -np.inf
    X_train = nd.X_train
    y_train = nd.y_train
    saver = tf.train.Saver()
    if nd.load_model is True:
        saver = tf.train.import_meta_graph(nd.model_path + ".meta")
        saver.restore(sess, nd.model_path)
        # init_op = tf.initialize_all_variables()
        # sess.run(init_op)
    else:
        S.initialize(sess)

    # Train model

    sess.run(tf.local_variables_initializer())
    print("Training model...")
    xshape = [nd.batch_size] + [len(X_train[0])] + [len(X_train[0][0])] + [1]
    yshape = [nd.batch_size] + list(y_train[0].shape) + [1]
    metric_counter = 0  # net_dict["metric_iter"]
    metric_step_counter = []
    stop_early = False
    for i in range(0, nd.epochs + 1):
        X_train, y_train = shuffle_io(X_train, y_train, nd,shuffle_batch_size=nd.shuffle_factor)
        if metric_counter == nd.metric_iter and stop_early is False:
            metric_step_counter.append(i)
            if nd.naive_test is True and nd.load_model is False:
                saver.save(sess, nd.model_path)

            print("\n_-_-_-_-_-_-_-_-_-_-Epoch", i, "_-_-_-_-_-_-_-_-_-_-\n")

            print("Validation results:")
            metric_eval.set_metrics( sess=sess, S=S, nd=nd, X=nd.X_valid, y=nd.y_valid)
            print("R2-score:", metric_eval.r2_by_epoch[-1])
            print("Avg-distance:", metric_eval.ape_by_epoch[-1])
            # saver.save(sess, nd.model_path, global_step=i, write_meta_graph=False)
            metric_counter = 0
            if nd.early_stopping is True:  # most likely overfitting instead of training
                metric_test.set_metrics(sess=sess, S=S, nd=nd, X=nd.X_test, y=nd.y_test)
                r2 = metric_test.r2_by_epoch[-1]
                if r2_max < r2:
                    r2_max = r2
                else:
                    if r2 < r2_max and r2>=0: # network is at least random and performs worse than before
                        stop_early = True
                        metric_eval.r2_by_epoch = metric_eval.r2_by_epoch[0:-1]
                        metric_eval.ape_by_epoch = metric_eval.ape_by_epoch[0:-1]
                        metric_eval.acc20_by_epoch = metric_eval.acc20_by_epoch[0:-1]

        # Train network
        if nd.naive_test is False or nd.time_shift == 0:

            for j in range(0, len(X_train) - nd.batch_size, nd.batch_size):
                x = (X_train[j:j + nd.batch_size])
                y = np.array(y_train[j:j + nd.batch_size])
                try:
                    x = np.reshape(x, xshape)
                except ValueError:
                    print("Warning: Check training data")
                y = np.reshape(y, yshape)
                t = np.max(S.train(sess, x, y, dropout=0.65))

        metric_counter = metric_counter + 1
        nd.epochs_trained = i
        if stop_early is True:
            break
    if nd.naive_test is False or nd.time_shift==0:
        saver.save(sess, nd.model_path)
    # Close session and add current time shift to network save file
    sess.close()
    metric_eval.set_bests(nd.early_stopping)  # find best (or newest) value in metric object

    return metric_eval


def initiate_network(nd):
    try:
        os.makedirs(os.path.dirname(nd.model_path))
        os.makedirs(os.path.dirname(nd.model_path + "output/"))
        os.makedirs(os.path.dirname(nd.model_path + "images/"))
    except OSError as exc:  # Guard against race condition
        if exc.errno != errno.EEXIST:
            raise
    if nd.session_from_raw is True:
        session = Slice.from_raw_data(nd.raw_data_path, filter_tetrodes=nd.filter_tetrodes)
        session.filter_neurons(100)
        session.print_details()
        print("Convolving data...")
        session.set_filter(nd.session_filter)
        print("Finished convolving data")
        session.filtered_spikes = stats.zscore(session.filtered_spikes, axis=1)  # Z Score neural activity
        session.to_pickle(nd.filtered_data_path)
    session = Slice.from_pickle(nd.filtered_data_path)
    session.filter_neurons_randomly(nd.neurons_kept_factor,nd.keep_neurons)
    session.print_details()
    nd.n_neurons = session.n_neurons
    if nd.switch_x_y is True:
        copy_pos_x = session.position_x
        session.position_x = session.position_y
        session.position_y = copy_pos_x

    # Start network with different time-shifts

    print_Net_data(nd)  # show network parameters in console
    return session


def initiate_lickwell_network(nd):
    try:
        os.makedirs(os.path.dirname(nd.model_path))
        os.makedirs(os.path.dirname(nd.model_path + "output/"))
        os.makedirs(os.path.dirname(nd.model_path + "images/"))
    except OSError as exc:  # Guard against race condition
        if exc.errno != errno.EEXIST:
            raise
    if nd.session_from_raw is True:
        session = Slice.from_raw_data(nd.raw_data_path, filter_tetrodes=nd.filter_tetrodes)
        session.filter_neurons(100)
        session.print_details()
        print("Convolving data...")
        session.set_filter(nd.session_filter)
        print("Finished convolving data")
        session.filtered_spikes = stats.zscore(session.filtered_spikes, axis=1)  # Z Score neural activity
        session.to_pickle(nd.filtered_data_path)
    session = Slice.from_pickle(nd.filtered_data_path)
    # session = Slice.from_pickle("slice_PFC_200.pkl")

    session.filter_neurons_randomly(nd.neurons_kept_factor)
    session.print_details()
    nd.n_neurons = session.n_neurons

    # Start network with different time-shifts
    print_Net_data(nd)  # show network parameters in console
    return session


def run_network(nd, session):
    # Assign validation set.
    for z in range(nd.initial_timeshift, nd.initial_timeshift + nd.time_shift_steps * nd.time_shift_iter,
                   nd.time_shift_iter):
        iter = int(z - nd.initial_timeshift) // nd.time_shift_iter
        # nd.network_shape = mlp
        nd.time_shift = z  # set current time shift
        # Time-Shift input and output
        X, y = time_shift_positions(session, z, nd)
        X,y = filter_behavior_component(X, y, nd, session)
        if len(X) != len(y):
            raise ValueError("Error: Length of x and y are not identical")
        if len(X) == 0:
            raise ValueError("Error: No samples. Check behavior component filter")

        # X, y = shuffle_io(X, y, nd, seed_no=2,shuffle_batch_size=nd.shuffle_factor)

        metric_list = []

        for k in range(0, nd.k_cross_validation):
            print("Timeshift", nd.time_shift)
            print("cross validation step", str(k + 1), "of", nd.k_cross_validation)
            if nd.naive_test is True and nd.time_shift != 0:
                nd.load_model = True
                nd.epochs=1
            nd.assign_training_testing(X, y, k)
            # nd.plot_validation_position_histogram(k)
            if nd.time_shift_steps == 1:
                metric = run_network_process(nd)
            else:
                with multiprocessing.Pool(
                        1) as p:  # keeping network inside process prevents tensorflow memory issues when restarting training session
                    metric = p.map(run_network_process, [nd])[0]
                    p.close()
            metric_list.append(metric)
        save_nd = copy.deepcopy(nd)
        save_nd.clear_io()
        save_metric = Network_output(net_data=save_nd, metric_by_cvs=metric_list)
        path = save_nd.model_path + "output/" + "network_output_timeshift=" + str(z) + ".pkl"
        save_as_pickle(path, save_metric)
    print("fin")


def run_lickwell_network(nd, session, X, y,pathname_metadata=""):
    nd.time_shift = nd.initial_timeshift  # set current shift
    # Shift input and output
    if len(X) != len(y):
        raise ValueError("Error: Length of x and y are not identical")
    metrics_k = []
    for k in range(0, nd.k_cross_validation):
        for j in range(0,int(1/nd.valid_ratio)):
            print("cross validation step", str(j + 1), "of",str(int(1/nd.valid_ratio)) )
            nd.assign_training_testing_lickwell(X, y, j, excluded_wells=[1], normalize=nd.lw_normalize)
            # save_nd = run_lickwell_network_process(nd)
            with multiprocessing.Pool(
                    1) as p:  # keeping network inside process prevents memory issues when restarting session
                save_nd = p.map(run_lickwell_network_process, [nd])[0]
                counter = 0
                for guess in save_nd[-1].guesses:
                    if guess.guess_is_correct is True:
                        counter +=1
                print("Accuracy",counter/len(save_nd[-1].guesses))
                p.close()
            metrics_k.append(save_nd)


    metrics,all_guesses = cross_validate_lickwell_data(metrics=metrics_k,licks=session.licks,epoch=-1,nd=nd)
    metrics_k_obj = []
    for metric in metrics_k:
        metrics_k_obj.append(cross_validate_lickwell_data(metrics=[metric],licks=session.licks,epoch=-1,nd=nd)[0])

    nd.clear_io()
    save_as_pickle(nd.model_path + "output/all_guesses_timeshift="+ str(nd.time_shift) + pathname_metadata + ".pkl", all_guesses)
    save_as_pickle(nd.model_path + "output/metrics_timeshift=" + str(nd.time_shift) + pathname_metadata+ ".pkl", metrics)
    save_as_pickle(nd.model_path + "output/nd_timeshift="+str(nd.time_shift)+ pathname_metadata +".pkl",nd)
    save_as_pickle(nd.model_path + "output/licks_timeshift="+str(nd.time_shift)+ pathname_metadata +".pkl",session.licks)
    save_as_pickle(nd.model_path + "output/metrics_k_timeshift="+str(nd.time_shift)+ pathname_metadata +".pkl",metrics_k_obj)
    # print_metric_details(nd.model_path, nd.initial_timeshift, pathname_metadata=pathname_metadata)
    pass

