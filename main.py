import tensorflow as tf
from database_api_beta import Slice,Filter,hann
from src.nets import MultiLayerPerceptron,ConvolutionalNeuralNetwork1
from src.metrics import get_r2,get_avg_distance,bin_distance,get_accuracy,get_radius_accuracy
import numpy as np
from src.conf import mlp,sp2,cnn1
import matplotlib.pyplot as plt
from src.settings import save_as_pickle, load_pickle
from sklearn.preprocessing import normalize
from random import shuffle
from itertools import zip_longest
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def test_accuracy(X_eval,y_eval,metadata,show_plot=False,plot_after_iter= 1,print_distance=False):
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
        if j % plot_after_iter == plot_after_iter-1 and show_plot is True:
            print("plot")
            time ="{:.1f}".format(metadata[j]["lickwells"]["time"]/1000)
            well = metadata[j]["lickwells"]
            title = "Next lickwell: " + str(well["lickwell"]) +" ("+ str(well["rewarded"])+") in " + time +"s" + " at " + str(metadata[j]["position"]) + "cm" + " and " + str(metadata[j]["time"])
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
            Y_prime[6,9] = 30
            Y_prime[14,9] = 30
            Y_prime[23,9] = 30
            Y_prime[32,9] = 30
            Y_prime[41,9] = 30
            Y[6,9] = 30
            Y[14,9] = 30
            Y[23,9] = 30
            Y[32,9] = 30
            Y[41,9] = 30
            ax1.imshow(Y, cmap="gray")
            ax2.imshow(Y_prime, cmap="gray")

            plt.show(block=True)
            plt.close()

    r2 = get_r2(prediction_list, actual_list)
    distance = get_avg_distance(prediction_list, actual_list,[X_STEP,Y_STEP])
    for i in range(0,11):
        print("accuracy",i,":",get_accuracy(prediction_list, actual_list,margin=i))
        # print("accuracy",i,":",get_radius_accuracy(prediction_list, actual_list,[X_STEP,Y_STEP],5))


    return r2,distance


def hann_generator(width):
    def hann(x):
        if np.abs(x) < width:
            return (1 + np.cos(x / width * np.pi)) / 2
        else:
            return 0
    return hann


def bin_filter(x):
    return 1

def position_as_map(y_list, xstep, ystep):
    ret = np.zeros(((X_MAX - X_MIN) // xstep, (Y_MAX - Y_MIN) // ystep))
    for x, y in list(zip(*y_list)):
        X = int((x - X_MIN) // xstep)
        Y = int((y - Y_MIN) // ystep)
        ret[X, Y] = 1
    return ret


def shift(li,n):
    return li[n:] + li[:n]


def time_shift_data(X,y,n):
    y = shift(y,n)[n:]
    X = X[:len(X)-n]
    return X,y


if __name__ == '__main__':

    # File settings settings

    # MODEL_PATH = "data/models/model_neocortex"
    # RAW_DATA_PATH = "G:/master_datafiles/raw_data/2018-04-09_14-39-52/" # neo cortex
    # FILTERED_DATA_PATH = "G:/master_datafiles/filtered_data/neocortex_hann_win_size_100.pkl"
    MODEL_PATH = "data/models/model_hippocampus"
    RAW_DATA_PATH = "G:/master_datafiles/raw_data/2018-05-16_17-13-37/" # hippocampus
    FILTERED_DATA_PATH = "G:/master_datafiles/filtered_data/hippocampus_hann_win_size_50.pkl"

    # Program execution settings

    LOAD_RAW_DATA = True# load source data from raw data path or load default model
    LOAD_MODEL = False# load model from model path
    TRAIN_MODEL = True # train model or just show results
    TRAINING_STEPS = 1000
    TIME_SHIFT = 2000
    TIME_SHIFT_STEPS = 2
    METRIC_ITER = 1000
    # Network parameters

    SLICE_SIZE = 1000
    BATCH_SIZE = 128
    WIN_SIZE = 50
    SEARCH_RADIUS = WIN_SIZE*2
    X_MAX = 240
    Y_MAX = 190
    X_MIN = 0
    Y_MIN = 100
    X_STEP = 5
    Y_STEP = 5
    session_filter = Filter(func=hann, search_radius=SEARCH_RADIUS, step_size=WIN_SIZE)
    # S = MultiLayerPerceptron([None, 68, 20, 1], mlp)
    S = ConvolutionalNeuralNetwork1([None, 68, 20, 1], cnn1)
    # Preprocess data

    if LOAD_RAW_DATA is True:
        # session = Slice.from_raw_data(RAW_DATA_PATH)
        # print("Convolving data...")
        # session.set_filter(session_filter)
        # print("Finished convolving data")
        # session.to_pickle("slice.pkl") # TODO delete again
        session = Slice.from_pickle("slice.pkl") # TODO delete again
        shifted_session_list = []# # time shifted y's
        copy_session = None
        for z in range(0, TIME_SHIFT_STEPS):
                copy_session = session.timeshift_position(z*TIME_SHIFT)
                shifted_session_list.append([copy_session.position_x,copy_session.position_y])
        X = []
        y_list = [[] for i in range(len(shifted_session_list))]
        metadata = []
        while len(copy_session.position_x)>=SLICE_SIZE:
            try:
                metadata.append(dict(lickwells=copy_session.licks[0],time=copy_session.absolute_time_offset,position = copy_session.position_x[0]))
            except:
                metadata.append(dict(rewarded=0,time=0,lickwell=0))
                print("deleteme")
            data_slice = copy_session[0:SLICE_SIZE]
            copy_session = copy_session[SLICE_SIZE:]
            X.append(normalize(data_slice.filtered_spikes,norm='l2',axis=0))
            for i in range(len(shifted_session_list)):
                y_list[i].append(position_as_map(shifted_session_list[i], X_STEP, Y_STEP))
            print("slicing", len(X),"of",len(session.position_x) // 1000)
        print("Finished slicing data")



        pickled_dict = dict(
                X=X,
                y_list=y_list,
                win_size=WIN_SIZE,
                search_radius=SEARCH_RADIUS,
                x_max=X_MAX,
                y_max=Y_MAX,
                x_min=X_MIN,
                y_min=Y_MIN,
                x_step=X_STEP,
                y_step=Y_STEP,
                metadata=metadata
            )
        save_as_pickle(FILTERED_DATA_PATH, pickled_dict)
    else:
        pickled_dict = load_pickle(FILTERED_DATA_PATH)
        X = pickled_dict["X"]
        y_list = pickled_dict["y_list"]
        metadata=pickled_dict["metadata"]


        shuffle_list = (list(zip(X, y_list, metadata)))
        shuffle(shuffle_list)
        X, y_list, metadata = zip(*shuffle_list)

        # ziplist = zip(X,y,metadata)
        # ziplist
    eval_length = int(len(X)/2)
    X_eval = X[eval_length:]
    metadata_eval=metadata[eval_length:]
    X_train = X[:eval_length]
    metadata_train = metadata[:eval_length]

    for z in range(0,TIME_SHIFT_STEPS):
        y_eval = y_list[z][eval_length:]
        y_train = y_list[z][:eval_length]

        # Train model

        saver = tf.train.Saver()
        sess = tf.Session()
        if LOAD_MODEL is True:
            saver = tf.train.import_meta_graph(MODEL_PATH+".meta")
            saver.restore(sess, MODEL_PATH)
        else:
            S.initialize(sess)

        if TRAIN_MODEL is True:
            sess.run(tf.local_variables_initializer())

            # print("__________________________________________")
            # print("Time shift is", z * TIME_SHIFT)
            print("Training model...")
            xshape = [BATCH_SIZE] + list(X_train[0].shape) + [1]
            yshape = [BATCH_SIZE] + list(y_train[0].shape) + [1]

            metric_counter = 0
            r2_scores_train = []
            avg_scores_train = []
            r2_scores_eval = []
            avg_scores_eval = []

            initial_eval = True
            for i in range(TRAINING_STEPS):
                # for j, (data_slice, position_map) in enumerate(zip(all_slices, all_position_maps)):
                for j in range(0, len(X_train) - BATCH_SIZE, BATCH_SIZE):

                    x =np.array([data_slice for data_slice in X_train[j:j + BATCH_SIZE]])
                    y = np.array(y_train[j:j + BATCH_SIZE])
                    x = np.reshape(x, xshape)
                    y = np.reshape(y, yshape)

                    # Print first model
                    if initial_eval is True:
                        print("Untrained model:")
                        test_accuracy(X_train, y_train, metadata_train)
                        initial_eval = False

                    t = np.max(S.train(sess, x, y,dropout=0.6))
                    print("loss:",t)
                    # Test accuracy
                    # TODO: schauen wo samples hingehen, time shift, schauen wo das aktuelle Ziel der Ratte ist bei jedem sample
                    if metric_counter == METRIC_ITER:
                        print("Step",i)
                        print("training data:")
                        r2_train, avg_train = test_accuracy(X_train,y_train,metadata_train,show_plot=False,plot_after_iter=100)
                        r2_scores_train.append(r2_train)
                        avg_scores_train.append(avg_train)
                        print(r2_train,avg_train)
                        print("evaluation data:")
                        r2_eval, avg_eval = test_accuracy(X_eval,y_eval,metadata_eval,show_plot=False,plot_after_iter=1)
                        r2_scores_eval.append(r2_eval)
                        avg_scores_train.append(avg_eval)
                        print(r2_eval,avg_eval)
                        saver.save(sess, MODEL_PATH, global_step=i, write_meta_graph=False)
                        print("____________________")
                        metric_counter = 0
                metric_counter = metric_counter + 1
            save_path = saver.save(sess, MODEL_PATH)

            # Save performance to file

            file_dict = dict(
                batch_size = BATCH_SIZE,
                network_type = "Multi Layer Perceptron",
                output_shape = y_list[0][0].shape,
                training_steps = TRAINING_STEPS,
                win_size = WIN_SIZE,
                x_max = X_MAX,
                y_max = Y_MAX,
                x_min = X_MIN,
                y_min = Y_MIN,
                x_step = X_STEP,
                y_step = Y_STEP,
                model_path = MODEL_PATH,
                raw_data_path = RAW_DATA_PATH,
                learning_rate = "placeholder", # TODO
                time_shift = TIME_SHIFT*z, # TODO
                search_radius=SEARCH_RADIUS,
                r2_scores_train = r2_scores_train,
                r2_scores_eval=r2_scores_eval,
                slice_size = SLICE_SIZE
                )

        # Test model performance
        # print(test_accuracy(X_train, y_train, metadata_train, show_plot=True, plot_after_iter=2))
        print(test_accuracy(X_eval, y_eval, metadata_eval, show_plot=True, plot_after_iter=2))

        # Close session and add time shift to training data

        sess.close()
        X_train,y_train = time_shift_data(X_train,y_train,TIME_SHIFT)

    print("fin")
