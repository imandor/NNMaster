import numpy as np
import matplotlib.pyplot as plt
from src.preprocessing import position_as_map
from src.settings import save_as_pickle


def get_r2(y_predicted, y_target, step_size):
    R2_list = []
    y_predicted = y_predicted * step_size
    y_target = y_target * step_size
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
    y_predicted = np.zeros((len(Y), nd.lw_classifications))
    y_target = np.zeros((len(Y), nd.lw_classifications))
    for j in range(0, len(X), 1):
        x = np.array([data_slice for data_slice in X[j:j + 1]])
        y = np.array(Y[j:j + 1])
        x = np.reshape(x, [1] + [nd.x_shape[1]] + [nd.x_shape[2]] + [nd.x_shape[3]])
        prediction = S.valid(sess, x)[0]
        y_predicted[j] = np.where(prediction == np.max(prediction), 1, 0)  # int(np.argmax(S.valid(sess, x)[0]))
        if np.max(prediction) == 0:  # TODO
            y_predicted[j] = np.array([1, 0, 0, 0, 0])
        y_target[j] = y
    return y_predicted, y_target


def test_accuracy(sess, S, nd, X, y, epoch, print_distance=False):
    return test_map_accuracy(sess, S, nd, X, y, epoch, print_distance=print_distance)


def get_label_accuracy(y_predicted, y_target):
    correct_count = 0
    for i in range(len(y_predicted)):
        if np.argmax(y_predicted[i]) == np.argmax(y_target[i]):
            correct_count += 1
    return correct_count / len(y_predicted)


def get_label_correct_count(y_predicted, y_target):
    counter = np.multiply(y_predicted, y_target)
    return np.sum(counter, axis=0)  # / np.sum(y_target,axis=0)


def get_label_total_count(y):
    return np.sum(y, axis=0)


class Lickwell_metric:
    def __init__(self, correct_guesses, guesses_by_well, total_samples, accuracy, discriminate_count,
                 discriminate_correct_count):
        self.correct_guesses = correct_guesses
        self.guesses_by_well = guesses_by_well
        self.total_samples = total_samples
        self.accuracy = accuracy
        self.discriminate_count = discriminate_count
        self.discriminate_correct_count = discriminate_correct_count


def cross_validate_lickwell_data(metrics):
    metrics_average = []
    for i, metrics_by_epoch in enumerate(metrics[0]):
        accuracy = []  # fraction
        correct_guesses = []  # array of length number of lickwells
        discriminate_correct_count = []  # array of length number of discriminate licks
        discriminate_count = []  # array of length number of discriminate licks
        guesses_by_well = []  # array of length number of lickwells
        total_samples = []
        for j, metrics_k in enumerate(metrics):
            accuracy.append(metrics[j][i].accuracy)
            correct_guesses.append(metrics[j][i].correct_guesses)
            discriminate_correct_count.append(metrics[j][i].discriminate_correct_count)
            discriminate_count.append(metrics[j][i].discriminate_count)
            guesses_by_well.append(metrics[j][i].guesses_by_well)
            total_samples.append(metrics[j][i].total_samples)
        accuracy = np.average(accuracy, axis=0)
        correct_guesses = np.average(correct_guesses, axis=0)
        discriminate_correct_count = np.sum(discriminate_correct_count, axis=0)
        discriminate_count = np.sum(discriminate_count, axis=0)
        guesses_by_well = np.sum(guesses_by_well, axis=0)
        total_samples = np.sum(total_samples, axis=0)
        metrics_average.append(Lickwell_metric(correct_guesses=correct_guesses, guesses_by_well=guesses_by_well,
                                               total_samples=total_samples, accuracy=accuracy,
                                               discriminate_count=discriminate_count,
                                               discriminate_correct_count=discriminate_correct_count))
    return metrics_average


def get_lickwell_accuracy(metrics_average):
    accuracy = []
    for metrics in metrics_average:
        accuracy.append(metrics.accuracy)
    return accuracy


def test_discrete_accuracy(sess, S, nd, X, y, metadata):
    y_predicted, y_target = predict_discrete(S, sess, X, y, nd)
    correct_guesses = get_label_correct_count(y_predicted, y_target)
    accuracy = get_label_accuracy(y_predicted, y_target)
    guesses_by_well = get_label_total_count(y_predicted)
    total_samples = get_label_total_count(y_target)
    discriminate_count, discriminate_correct_count = test_discriminate_discrete_accuracy(y_predicted, y_target,
                                                                                         metadata, nd)
    print("correct guesses", correct_guesses)
    print("guesses by well", guesses_by_well)
    print("total samples", total_samples)
    print("accuracy", accuracy)
    # y_predicted, y_target = predict_discrete(S, sess, X, y, nd)
    return Lickwell_metric(correct_guesses=correct_guesses, guesses_by_well=guesses_by_well,
                           total_samples=total_samples,
                           accuracy=accuracy, discriminate_count=discriminate_count,
                           discriminate_correct_count=discriminate_correct_count)


def test_discriminate_discrete_accuracy(y_predicted, y_target, metadata, nd):
    unique = np.unique(nd.valid_licks, return_counts=False)
    correct_counter = np.zeros(unique.shape)
    counts = np.zeros(unique.shape)
    for i, e in enumerate(y_predicted):
        if metadata[i].lick_id in nd.valid_licks:
            index = np.where(unique == metadata[i].lick_id)
            counts[index] += 1
            if np.array_equal(e, y_target[i]):
                correct_counter[index] += 1
    return counts, correct_counter


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
