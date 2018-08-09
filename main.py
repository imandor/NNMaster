import tensorflow as tf
from database_api_beta import Slice,Filter,hann
from src.nets import SpecialNetwork1, SpecialNetwork2, MultiLayerPerceptron
import numpy as np
from src.conf import sp1, sp2, mlp
import matplotlib.pyplot as plt
from numpy.random import seed
from multiprocessing import Pool
from src.settings import save_as_pickle, load_pickle

def sigmoid(x):
    return 1 / (1 + np.exp(-x))


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

def bin_distance(bin_1,bin_2):
    return [abs(bin_1[1]-bin_2[1]),abs(bin_1[2]-bin_2[2])]


if __name__ == '__main__':
    WIN_SIZE = 57
    X_MAX = 240
    Y_MAX = 180
    X_MIN = 0
    Y_MIN = 100
    # session = Slice.from_raw_data("G:/raw data/16-05-2018/2018-05-16_17-13-37/")#hippocampus
    # session = Slice.from_raw_data("G:/raw data/2018-04-09_14-39-52/") # neo cortex
    # session.to_pickle("data/pickle/hippocampus.pkl")
    session = Slice.from_pickle("data/pickle/neofrontal_cortex_hann_win_size_57.pkl") # load a data slice containing entire session
    # # S = SpecialNetwork1([None, 166, 18, 1], sp1)
    S = MultiLayerPerceptron([None, 166, 17, 1], mlp)
    # # S = SpecialNetwork2([None, 166, 18, 1], sp2)
    # # hann = hann_generator(WIN_SIZE)
    session_filter = Filter(func=hann, search_radius=2 * WIN_SIZE, step_size=WIN_SIZE)
    # session.set_filter(session_filter)
    # session.to_pickle("data/pickle/neofrontal_cortex_hann_win_size_57.pkl") # load a data slice containing entire session

    # session.to_pickle("data/pickle/neofrontal_cortex_deleteme.pkl")
    # session = Slice.from_pickle("data/pickle/neofrontal_cortex_deleteme.pkl")
    # spikes = np.divide(session.spikes[0],1000)
    # filtered_spikes = session.filtered_spikes[0]
    # f, ax = plt.subplots()
    # session[0:1000].plot_filtered_spikes(ax)
    #
    # f.show()







    print(np.max(session.position_x), np.min(session.position_x))
    print(np.max(session.position_y), np.min(session.position_y))

    print(S.output.shape.as_list())
    # session = session[0:500000]
    # session = session[0:200000]

    # all_slices = []
    # all_position_maps = []
    # for frm, to in zip(range(0, len(session.position_x), 1000), range(1000, len(session.position_x), 1000)):
    #     # print(" " * 50 + "{}/{}".format(frm, len(session.position_x)), end="\r")
    #     print("Convolving",frm,"to",to,"...")
    #     data_slice = session[frm:to]
    #     data_slice.set_filter(session_filter)
    #     all_slices.append(data_slice)
    #     all_position_maps.append(position_as_map(data_slice, 1, 1))
    # save_as_pickle("data/pickle/all_slices.pkl",all_slices)
    # save_as_pickle("data/pickle/all_position_maps.pkl",all_position_maps)
    #
    all_slices = load_pickle("data/pickle/all_slices.pkl")
    all_position_maps = load_pickle("data/pickle/all_position_maps.pkl")
    eval_length = int(len(all_slices)/2)
    eval_slices = all_slices[eval_length:]
    eval_positions = all_position_maps[eval_length:]
    all_slices = all_slices[:eval_length]
    all_position_maps = all_position_maps[:eval_length]
    BATCH_SIZE = 128

    with tf.Session() as sess:
        S.initialize(sess)
        xshape = [BATCH_SIZE] + list(all_slices[0].filtered_spikes.shape) + [1]
        yshape = [BATCH_SIZE] + list(all_position_maps[0].shape) + [1]
        for i in range(20000):
            # for j, (data_slice, position_map) in enumerate(zip(all_slices, all_position_maps)):
            for j in range(0, len(all_slices) - BATCH_SIZE, BATCH_SIZE):
                x = np.array([data_slice.filtered_spikes for data_slice in all_slices[j:j + BATCH_SIZE]])
                y = np.array(all_position_maps[j:j + BATCH_SIZE])
                x = np.reshape(x, xshape)
                y = np.reshape(y, yshape)
                print(i, ":", np.max(S.train(sess, x, y)))

                # if j // BATCH_SIZE in [0, 1, 2, 3, 4, 5, 10, 20] and i % 1000 == 999:
                #     print("plot")
                #     y_prime = S.eval(sess, x)
                #     fig = plt.figure()
                #     ax1 = fig.add_subplot(1, 2, 1)
                #     ax2 = fig.add_subplot(1, 2, 2)
                #     ax1.axis('off')
                #     ax2.axis('off')
                #     fig.subplots_adjust(left=0.01, right=0.99, bottom=0.01, top=0.99, hspace=0.01, wspace=0.01)
                #     Y = y[0, :, :, 0]
                #     Y -= np.min(Y)
                #     Y /= np.max(Y)
                #     Y *= 255
                #     Y_prime = sigmoid(y_prime[0, :, :, 0])
                #     Y_prime -= np.min(Y_prime)
                #     Y_prime /= np.max(Y_prime)
                #     Y_prime *= 255
                #     ax1.imshow(Y, cmap="gray")
                #     ax2.imshow(Y_prime, cmap="gray")
                #     plt.show()
        # save_as_pickle("data/pickle/perceptron_08-08-18_1",S)


        BATCH_SIZE=1
        xshape = [BATCH_SIZE] + list(eval_slices[0].filtered_spikes.shape) + [1]
        yshape = [BATCH_SIZE] + list(eval_positions[0].shape) + [1]
        for j in range(0, len(all_slices) - BATCH_SIZE, BATCH_SIZE):
            x = np.array([data_slice.filtered_spikes for data_slice in eval_slices[j:j + BATCH_SIZE]])
            y = np.array(eval_positions[j:j + BATCH_SIZE])
            x = np.reshape(x, xshape)
            y = np.reshape(y, yshape)
            # print(np.max(S.train(sess, x, y)))
            a = S.eval(sess, x)
            bin_1 = np.unravel_index(a.argmax(), a.shape)
            bin_2 = np.unravel_index(y.argmax(), y.shape)
            print("prediction:",bin_1)
            print("actual:    ",bin_2)
            print("distance:",bin_distance(bin_1,bin_2))
            print("_____________")
    print("fin")






    seed(1)
    bin_size = 700
    step_size=700
    units = 500
    epochs = 10


    data_slice = Slice.from_path(load_from="data/pickle/hippocampus_session.pkl")  # load a data slice containing entire session

    smaller_data_slice = data_slice[0:2000000]  # slices first 200 seconds of session

    phases = data_slice.get_all_phases()  # gets all training phases as list of data slices

    trial = phases[0].get_nth_trial(0)  # gets first trial in first training phase

    list_of_trials = data_slice.get_trials(
        slice(0, 200000))  # returns a list of trials in the first 200000 ms of session

    sub_list_of_trials = list_of_trials[0:10]  # slices first 10 entries in list of trials
    # for trial_i in sub_list_of_trials:
    #     trial_i.plot_filtered_spikes(filter=bin_filter, window=1, step_size=100)

    nth_trial_list = phases.get_nth_trial_in_each_phase(1)  # gets list of all second trials in list of phases

    all_trials_in_second_phase = phases[2].get_all_trials()  # gets all trials in second phase
    all_trials_starting_with_well_1 = all_trials_in_second_phase.filter_trials_by_well(
        start_well=1)

    # data_slice.set_filter(filter=bin_filter, search_window_size=700, step_size=700, num_threads=20)


    for i in range(0, 3):  # for neuron no 0 to 9:
        all_trials_starting_with_well_1.plot_time_x_trials(neuron_no=i)
        all_trials_starting_with_well_1.plot_positionx_x_trials(neuron_no=i)

    print("fin")
