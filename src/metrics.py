import numpy as np
import matplotlib.pyplot as plt
from src.preprocessing import position_as_map
from src.settings import save_as_pickle


def get_r2(y_predicted, y_target, step_size):
    R2_list = []
    y_predicted = y_predicted * step_size
    y_target = y_target * step_size

    # begin test
    # y_mean = 90 # np.mean(y_target[:, 0])
    # y_pred = y_predicted[:,0]
    # y_tar = y_target[:,0]
    # y_p = []
    # y_t = []
    # R2_count_pos = 0
    # R2_count_neg = 0
    # R2_list = []
    # R2_min = 1
    # index = 0
    # target_90_counter = 0
    # terrible_counter = 0
    # for i in range(y_tar.shape[0]):
    #     y_p.append(y_pred[i])
    #     y_t.append(y_tar[i])
    #     if y_tar[i] == 90:
    #         target_90_counter += 1
    #     y_P = np.array(y_p)
    #     y_T = np.array(y_t)
    #     R2_tot = 1-np.sum((y_P-y_T)**2)/np.sum((y_T-y_mean)**2)
    #     R2_current = 1 - (y_pred[i]-y_tar[i])**2 / (y_tar[i]-y_mean)**2
    #     if R2_current < -1000:
    #         terrible_counter += 1
    #     if R2_current < R2_min:
    #         R2_min = R2_current
    #         index = i
    #     if i == 2961:
    #         print("asd")
    #     R2_list.append(R2_current)
    #     if R2_current >= 0:
    #         R2_count_pos +=1
    #     else:
    #         R2_count_neg+=1
    #
    #     dist_curr = np.abs(y_pred[i]-y_tar[i])
    #     random_curr = np.abs(y_T[i]-y_mean)
    #     dist_total = np.average(np.abs(y_P-y_T))
    #     random_total = np.average(np.abs(y_T-y_mean))
    #     print("Y_pred",y_pred[i])
    #     print("Y_tar",y_tar[i])
    #     print("R2 total:",R2_tot)
    #     print("R2 current:",R2_current)
    #     print("dist curr:",dist_curr)
    #     print("dist total:",dist_total)
    #     print("dist r current:",random_curr)
    #     print("dist r total:",random_total)
    #
    #     print("fin")
    # R2_list = np.array(R2_list)
    # R2_min = np.min(R2_list)
    # R2_max = np.max(R2_list)
    # R2_list_avg = np.average(R2_list)

    # end test

    for i in range(y_target.shape[1]):
        y_mean = np.mean(y_target[:, i])

        R2 = 1 - np.sum((y_predicted[:, i] - y_target[:, i]) ** 2) / np.sum((y_target[:, i] - y_mean) ** 2)
        R2_list.append(R2)
    R2_array = np.array(R2_list)
    return R2_array  # Return an array of R2s


def get_distances(y_predicted, y_target, step_size):
    return np.sqrt(np.square(step_size[0] * (y_predicted[:, 0] - y_target[:, 0])) + np.square(
        step_size[1] * (y_predicted[:, 1] - y_target[:, 1])))


def get_avg_distance(y_predicted, y_target, step_size):
    distance_list = get_distances(y_predicted, y_target, step_size)
    return np.average(distance_list, axis=0)


def get_radius_accuracy(y_predicted, y_target, step_size, absolute_margin=0):
    """ returns percentage of valid vs predicted with absolute distance <= margin in cm"""
    distance_list = get_radius_distance_list(y_predicted, y_target, step_size, absolute_margin)
    return np.average(distance_list, axis=0)


def get_radius_distance_list(y_predicted, y_target, step_size, absolute_margin=0):
    return np.sqrt(np.square(step_size[0] * (y_predicted[:, 0] - y_target[:, 0])) + np.square(
        step_size[1] * (y_predicted[:, 1] - y_target[:, 1]))) <= absolute_margin


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def average_position(mapping):
    # mapping = np.reshape(mapping,[mapping.shape[0],mapping.shape[1]])
    # sum_0 = np.dot(mapping,np.arange(mapping.shape[1]))
    # sum_1 = np.dot(mapping.T,np.arange(mapping.shape[0]))

    sum_0 = np.sum(mapping, axis=0)
    sum_0 = np.sum(np.sum(sum_0 * np.arange(mapping.shape[1]))) / np.sum(np.sum(sum_0))
    sum_1 = np.sum(mapping, axis=1)
    sum_1 = np.sum(np.sum(sum_1 * np.arange(mapping.shape[0]))) / np.sum(np.sum(sum_1))
    try:
        return [int(sum_1), int(sum_0)]
    except ValueError:
        print("Value error, check validation output")
        return "Value error, check validation output"


def plot_histogram(sess, S, nd, X, y):
    y_predicted, y_target = predict_map(S, sess, X, y)
    distances = get_distances(y_predicted, y_target, [nd.X_STEP, nd.Y_STEP])
    fig, ax = plt.subplots()
    x_axis_labels = np.linspace(0, 150, 15)
    # ax.plot(time_shift_list,distance_scores_train,label='Training set',color='r')
    ax.hist(distances, x_axis_labels, density=True, cumulative=True, rwidth=0.9, color='b')
    ax.grid(c='k', ls='-', alpha=0.3)
    # ax.set_title(r'$\varnothing$distance of validation wrt time-shift')
    ax.set_xlabel("Prediction error [cm]")
    ax.set_ylabel('Fraction of instances with prediction error')
    fig.tight_layout()

    x_axis_labels = np.linspace(0, 150, 150)
    fig2, ax2 = plt.subplots()
    # ax.plot(time_shift_list,distance_scores_train,label='Training set',color='r')
    ax2.hist(distances, x_axis_labels, density=False, rwidth=0.9, color='r')
    ax2.grid(c='k', ls='-', alpha=0.3)
    # ax.set_title(r'$\varnothing$distance of validation wrt time-shift')
    ax2.set_xlabel('Prediction error [cm]')
    ax2.set_ylabel("Instances with given prediction error")
    ax2.set_ylim([0, 31])
    fig.tight_layout()
    plt.show()
    print("plotted histograms")

    # plt.savefig(PATH + "images/avg_dist" + "_epoch=" + str(training_step_list[-i]) + ".pdf")


def predict_map(S, sess, X, Y):
    """ returns list of x/y tuples for predicted and actual positions"""
    xshape = [1] + list(X[0].shape) + [1]
    yshape = [1] + list(Y[0].shape) + [1]
    y_predicted = np.zeros([len(Y), 2])
    y_target = np.zeros([len(Y), 2])
    for j in range(0, len(X), 1):
        x = np.array([data_slice for data_slice in X[j:j + 1]])
        y = np.array(Y[j:j + 1])
        x = np.reshape(x, xshape)
        y = np.reshape(y, yshape)
        prediction = S.valid(sess, x)[0, :, :, 0]
        prediction_a = sigmoid(prediction)
        y = y[0, :, :, 0]
        # bin_1 = average_position(a)
        bin_1 = np.unravel_index(prediction_a.argmax(), prediction_a.shape)
        bin_2 = average_position(y)
        y_predicted[j][0] = bin_1[0]
        y_predicted[j][1] = bin_1[1]
        y_target[j][0] = bin_2[0]
        y_target[j][1] = bin_2[1]
    return y_predicted, y_target


def predict_discrete(S, sess, X, Y, nd):
    y_predicted = np.zeros((len(Y), 5))
    y_target = np.zeros((len(Y), 5))
    for j in range(0, len(X), 1):
        x = np.array([data_slice for data_slice in X[j:j + 1]])
        y = np.array(Y[j:j + 1])
        x = np.reshape(x, [1] + [nd.x_shape[1]] + [nd.x_shape[2]] + [nd.x_shape[3]])
        y_predicted[j] = S.valid(sess, x)[0]
        y_target[j] = y

    return y_predicted, y_target


def test_accuracy(sess, S, nd, X, y, epoch, print_distance=False):
    if nd.metric == "map":
        return test_map_accuracy(sess, S, nd, X, y, epoch, print_distance=False)
    elif nd.metric == "discrete":
        return test_discrete_accuracy(sess, S, nd, X, y, epoch, print_distance=False)


def get_discrete_accuracy(y_predicted, y_target):
    correct_count = 0
    list_1 = []
    list_2 = []
    for i in range(len(y_predicted)):
        y_1 = int(np.argmax(y_predicted[i]))
        y_2 = np.argmax(y_target[i])
        list_1.append(y_1)
        list_2.append(y_2)
        if y_1 == y_2:
            correct_count += 1
    return correct_count/ len(y_predicted)


def test_discrete_accuracy(sess, S, nd, X, y, epoch, print_distance=False):
    y_predicted, y_target = predict_discrete(S, sess, X, y, nd)
    r2 = None
    distance = None
    accuracy = get_discrete_accuracy(y_predicted,y_target)
    print(y_predicted[0])
    return r2, distance, accuracy


def test_map_accuracy(sess, S, nd, X, y, epoch, print_distance=False):
    y_predicted, y_target = predict_map(S, sess, X, y)
    r2 = get_r2(y_predicted, y_target, [nd.X_STEP, nd.Y_STEP])
    distance = get_avg_distance(y_predicted, y_target, [nd.X_STEP, nd.Y_STEP])
    accuracy = []
    for i in range(0, 20):
        acc = get_radius_accuracy(y_predicted, y_target, [nd.X_STEP, nd.Y_STEP], i)
        accuracy.append(acc)
        if i == 19 and print_distance is True: print("Fraction pos error less than", i, ":", acc)
    if False:  # plot planes
        plot_all_planes(X, y, y_predicted, y_target, nd)
    if False:
        save_as_pickle("C:/Users/NN/AppData/Local/Temp/animation/predicted/step=" + str(nd.epochs_trained),
                       y_predicted[0])
        save_as_pickle("C:/Users/NN/AppData/Local/Temp/animation/target/step=" + str(nd.epochs_trained), y_target[0])
    if False:
        plot_results_as_map(sess, S, X, epoch, nd)
    return r2, distance, accuracy


def plot_all_planes(X, y, is_training_data, y_predicted, y_target, nd):
    map_list = y
    distance_list = get_radius_distance_list(y_predicted, y_target, [nd.X_STEP, nd.Y_STEP],
                                             absolute_margin=50)
    # y_target_f = []
    # y_prediction_f = []
    # for i, dist in enumerate(distance_list):
    #     if True:
    #         y_target_f.append(y_target[i])
    #         y_prediction_f.append(y_predicted[i])
    # plot_plane(y_prediction_f, nd, "C:/Users/NN/Desktop/Master/" + is_train + "_all" + "_prediction")
    # plot_plane(y_target_f, nd, "C:/Users/NN/Desktop/Master/" + is_train + "_all" + "_target")
    y_target_f = []
    y_prediction_f = []
    x_list = []
    for i, dist in enumerate(distance_list):
        if dist:
            y_target_f.append(y_target[i])
            y_prediction_f.append(y_predicted[i])
            x_list.append(X[i])
    plot_axis_representation_1d(y_target, "C:/Users/NN/Desktop/Master/experiments/Histogram of positions/all_positions",
                                is_training_data)
    plot_axis_representation_1d(np.array(y_target_f),
                                "C:/Users/NN/Desktop/Master/experiments/Histogram of positions/all_better_40_target",
                                is_training_data)
    plot_axis_representation_1d(np.array(y_prediction_f),
                                "C:/Users/NN/Desktop/Master/experiments/Histogram of positions/all_better_40_prediction",
                                is_training_data)

    y_target_t = np.array(y_target_f)
    y_prediction_t = np.array(y_prediction_f)
    y_distance = np.abs(y_target_t - y_prediction_t)
    spikes_per_second = np.mean([len(x) for x in x_list]) / 200

    # plot_plane(y_prediction_f, nd, "C:/Users/NN/Desktop/Master/" + is_train + "_better" + "_prediction")
    # plot_plane(y_target_f, nd, "C:/Users/NN/Desktop/Master/" + is_train + "_better" + "_target")
    y_target_f = []
    y_prediction_f = []
    x_list = []
    maps = []
    for i, dist in enumerate(distance_list):
        if not dist:
            y_target_f.append(y_target[i])
            y_prediction_f.append(y_predicted[i])
            x_list.append(X[i])
            maps.append(map_list[i])
    plot_axis_representation_1d(np.array(y_target_f),
                                "C:/Users/NN/Desktop/Master/experiments/Histogram of positions/all_worse_40_target",
                                is_training_data)
    plot_axis_representation_1d(np.array(y_prediction_f),
                                "C:/Users/NN/Desktop/Master/experiments/Histogram of positions/all_worse_40_prediction",
                                is_training_data)

    # y_target_f = np.array(y_target_f)
    # y_prediction_f = np.array(y_prediction_f)
    # y_distance = np.abs(y_target_f-y_prediction_f)
    # spikes_per_second = np.mean([len(x) for x in x_list]) /200
    # y_avg_distance = np.average(y_distance)
    plot_plane(y_prediction_f, nd, "C:/Users/NN/Desktop/Master/" + is_training_data + "_worse" + "_prediction")
    plot_plane(y_target_f, nd, "C:/Users/NN/Desktop/Master/" + is_training_data + "_worse" + "_target")


def plot_results_as_map(sess, S, x, epoch, nd):
    x = nd.X_valid[1000]
    x = np.reshape(x, [1] + list(x.shape) + [1])
    y = nd.y_valid[1000]
    # y = S.valid(sess, x)[0, :, :, 0]
    # y = sigmoid(y)
    print("plot")
    # fig = plt.figure()
    fig, ax = plt.subplots()
    ax.axis('off')
    # fig.subplots_adjust(left=0.01, right=0.99, bottom=0.01, top=0.99, hspace=0.01, wspace=0.01)
    y -= np.min(y)
    y /= np.max(y)
    y *= 255
    ax.imshow(y, cmap="gray")
    plt.savefig("C:/Users/NN/Desktop/presentation_group/plane_original" + str(epoch))
    plt.close()


def plot_axis_representation_2d(y_values, path):
    # plots representation over one axis
    fig, ax = plt.subplots()
    ax.hist2d(x=y_values[:, 0], y=y_values[:, 1], bins=[80, 30])
    plt.savefig(path + "_2d")
    plt.close('all')


def plot_axis_representation_1d(y_values, path, is_training):
    # plots representation over one axis
    fig, ax = plt.subplots()
    ind = np.arange(0, 80)
    if is_training:
        classification = "_train_"
        c = "r"
    else:
        classification = "_valid_"
        c = "k"
    ax.hist(y_values[:, 0], ind, density=True, color=c)
    plt.savefig(path + classification + "_1d")
    plt.close('all')


def print_Net_data(nd):
    print("session_filter", nd.session_filter)
    print("SLICE_SIZE", nd.SLICE_SIZE)
    print("Y_SLICE_SIZE", nd.Y_SLICE_SIZE)
    print("STRIDE", nd.STRIDE)
    print("WIN_SIZE", nd.WIN_SIZE)
    print("EPOCHS", nd.EPOCHS)
    print("TIME_SHIFT_STEPS", nd.TIME_SHIFT_STEPS)
    print("SHUFFLE_DATA", nd.SHUFFLE_DATA)
    print("SHUFFLE_FACTOR", nd.SHUFFLE_FACTOR)
    print("TIME_SHIFT_ITER", nd.TIME_SHIFT_ITER)
    print("METRIC_ITER", nd.METRIC_ITER)
    print("BATCH_SIZE", nd.BATCH_SIZE)
    print("= SEARCH_RADIUS", nd.SEARCH_RADIUS)


def plot_plane(y, nd, path):
    pos_x_list = []
    pos_y_list = []
    for i, posxy in enumerate(y):
        pos_x_list.append(posxy[0])
        pos_y_list.append(posxy[1])
    a = position_as_map([pos_x_list, pos_y_list], nd.X_STEP, nd.Y_STEP, nd.X_MAX,
                        nd.X_MIN, nd.Y_MAX, nd.Y_MIN)
    Y = a
    # print("plot:", "sample size is", len(y))
    fig, ax = plt.subplots()
    # fig.suptitle("Samples: "+str(len(y)))
    ax.axis('off')
    fig.subplots_adjust(left=0.01, right=0.99, bottom=0.01, top=0.99, hspace=0.01, wspace=0.01)
    Y -= np.min(y)
    # Y /= np.max(1,np.max(y))
    Y *= 255
    Y[30 // nd.X_STEP, 50 // nd.Y_STEP] = 30
    Y[70 // nd.X_STEP, 50 // nd.Y_STEP] = 30
    Y[115 // nd.X_STEP, 50 // nd.Y_STEP] = 30
    Y[160 // nd.X_STEP, 50 // nd.Y_STEP] = 30
    Y[205 // nd.X_STEP, 50 // nd.Y_STEP] = 30

    ax.imshow(Y, cmap="gray")
    plt.savefig(path)
    # plt.close()
