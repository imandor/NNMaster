import tensorflow as tf
from database_api_beta import Slice,Filter,hann
from src.nets import MultiLayerPerceptron
from src.metrics import get_r2,get_avg_distance,bin_distance,get_accuracy
import numpy as np
from src.conf import mlp
import matplotlib.pyplot as plt
from src.settings import save_as_pickle, load_pickle
from sklearn.preprocessing import normalize

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def test_accuracy(X_eval,y_eval,show_plot=False,print_distance=False):
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
        if j % 100 == 99 and show_plot is True:
            print("plot")
            y_prime = S.eval(sess, x)
            fig = plt.figure()
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
            ax1.imshow(Y, cmap="gray")
            ax2.imshow(Y_prime, cmap="gray")
            plt.show()

    r2 = get_r2(prediction_list, actual_list)
    distance = get_avg_distance(prediction_list, actual_list,[X_STEP,Y_STEP])
    for i in range(0,20):
        print("accuracy",i,":",get_accuracy(prediction_list, actual_list,margin=i))



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

def position_as_map(slice, xstep, ystep):
    ret = np.zeros(((X_MAX - X_MIN) // xstep, (Y_MAX - Y_MIN) // ystep))
    for x, y in zip(slice.position_x, slice.position_y):
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

    # MODEL_PATH = "data/models/model_neocortex.ckpt"
    MODEL_PATH = "data/models/model_hippocampus.ckpt"
    RAW_DATA_PATH = "G:/master_datafiles/raw_data/2018-05-16_17-13-37/" # hippocampus
    # RAW_DATA_PATH = "G:/master_datafiles/raw_data/2018-04-09_14-39-52/" # neo cortex
    # FILTERED_DATA_PATH = "G:/master_datafiles/filtered_data/neocortex_hann_win_size_100.pkl"
    FILTERED_DATA_PATH = "G:/master_datafiles/filtered_data/hippocampus_hann_win_size_100.pkl"

    # Program execution settings

    LOAD_RAW_DATA = False # load source data from raw data path or load default model
    LOAD_MODEL = True # load model from model path
    TRAIN_MODEL = True # train model or just show results
    TRAINING_STEPS = 2001
    TIME_SHIFT = 500
    TIME_SHIFT_STEPS = 10

    # Network parameters

    SLICE_SIZE = 1000
    BATCH_SIZE = 128
    WIN_SIZE = 100
    SEARCH_RADIUS = WIN_SIZE*2
    X_MAX = 240
    Y_MAX = 190
    X_MIN = 0
    Y_MIN = 100
    X_STEP = 5
    Y_STEP = 5
    session_filter = Filter(func=hann, search_radius=SEARCH_RADIUS, step_size=WIN_SIZE)
    S = MultiLayerPerceptron([None, 68, 10, 1], mlp)

    # Preprocess data

    if LOAD_RAW_DATA is True:
        session = Slice.from_raw_data(RAW_DATA_PATH)
        print("Convolving data...")
        session.set_filter(session_filter)
        print("Finished convolving data")
        X = []
        y = []
        for frm, to in zip(range(0, len(session.position_x), SLICE_SIZE), range(SLICE_SIZE, len(session.position_x), SLICE_SIZE)):
            # print(" " * 50 + "{}/{}".format(frm, len(session.position_x)), end="\r")
            # print("Convolving",frm,"to",to,"...")
            print("Slicing data",frm,to)
            data_slice = session[frm:to]
            # data_slice.set_filter(session_filter)
            X.append(normalize(data_slice.filtered_spikes,norm='l2',axis=0))
            y.append(position_as_map(data_slice, X_STEP, Y_STEP))
        print("Finished slicing data")
        pickled_dict = dict(
            X=X,
            y=y,
            win_size=WIN_SIZE,
            search_radius=SEARCH_RADIUS,
            x_max=X_MAX,
            y_max=Y_MAX,
            x_min=X_MIN,
            y_min=Y_MIN,
            x_step=X_STEP,
            y_step=Y_STEP
        )
        save_as_pickle(FILTERED_DATA_PATH, pickled_dict)
    else:
        pickled_dict = load_pickle(FILTERED_DATA_PATH)
        X = pickled_dict["X"]
        y = pickled_dict["y"]

    eval_length = int(len(X)/2)
    X_eval = X[eval_length:]
    y_eval = y[eval_length:]
    X_train = X[:eval_length]
    y_train = y[:eval_length]

    # Train model

    saver = tf.train.Saver()
    for z in range(0, TIME_SHIFT_STEPS):
        print("__________________________________________")
        print("Time shift is",z*TIME_SHIFT)
        sess = tf.Session()
        if LOAD_MODEL is True:
            saver.restore(sess, MODEL_PATH)
        if TRAIN_MODEL is True:
            print("Training model...")
            S.initialize(sess)
            xshape = [BATCH_SIZE] + list(X_train[0].shape) + [1]
            yshape = [BATCH_SIZE] + list(y_train[0].shape) + [1]

            metric_counter = 0
            r2_scores_train = []
            avg_scores_train = []
            r2_scores_eval = []
            avg_scores_eval = []
            for i in range(TRAINING_STEPS):
                # for j, (data_slice, position_map) in enumerate(zip(all_slices, all_position_maps)):
                for j in range(0, len(X_train) - BATCH_SIZE, BATCH_SIZE):

                    x =np.array([data_slice for data_slice in X_train[j:j + BATCH_SIZE]])
                    y = np.array(y_train[j:j + BATCH_SIZE])
                    x = np.reshape(x, xshape)
                    y = np.reshape(y, yshape)
                    t = (S.train(sess, x, y))

                    # Test accuracy

                    if metric_counter == 1000:
                        print("Step",i)
                        r2_train, avg_train = test_accuracy(X_train,y_train)
                        r2_scores_train.append(r2_train)
                        avg_scores_train.append(avg_train)
                        print("training data:",r2_train,avg_train)
                        r2_eval, avg_eval = test_accuracy(X_eval,y_eval)
                        r2_scores_eval.append(r2_eval)
                        avg_scores_train.append(avg_eval)
                        print("evaluation data:",r2_eval,avg_eval)
                        print("____________________")
                        save_path = saver.save(sess, MODEL_PATH)
                        metric_counter = 0
                metric_counter = metric_counter + 1
            save_path = saver.save(sess, MODEL_PATH)

            # Save performance to file

            file_dict = dict(
                batch_size = BATCH_SIZE,
                network_type = "Multi Layer Perceptron",
                output_shape = y.shape,
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

        print(test_accuracy(X_eval, y_eval, show_plot=False, print_distance=True))

        # Close session and add time shift to training data

        sess.close()
        X_train,y_train = time_shift_data(X_train,y_train,TIME_SHIFT)

    print("fin")




