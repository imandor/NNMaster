import numpy as np
import matplotlib.pyplot as plt

from src.preprocessing import position_as_map
from src.database_api_beta import Evaluated_Lick, get_lick_from_id
from src.settings import load_pickle


def get_r2(y_predicted, y_target):
    """
    :param y_predicted: list of predicted outputs
    :param y_target: list of target outputs
    :return: R2 score, format [R2_x-axis,R2_y-axis]
    """
    R2_list = []
    y_predicted = y_predicted
    y_target = y_target
    for i in range(y_target.shape[1]):
        y_mean = np.mean(y_target[:, i])
        R2 = 1 - np.sum((y_predicted[:, i] - y_target[:, i]) ** 2) / np.sum((y_target[:, i] - y_mean) ** 2)
        R2_list.append(R2)
    R2_array = np.array(R2_list)
    return R2_array  # Return an array of R2s


def get_distances(y_predicted, y_target, step_size):
    """

    :param y_predicted: list of predicted outputs
    :param y_target: list of target outputs
    :param step_size: length of output bins, format [len_x-bins,len_y-bins]
    :return: list of distances in bins between all target and prediction bins
    """
    return np.sqrt(np.square(step_size[0] * (y_predicted[:, 0] - y_target[:, 0])) + np.square(
        step_size[1] * (y_predicted[:, 1] - y_target[:, 1])))


def get_avg_distance(y_predicted, y_target, step_size):
    """
    :param y_predicted: list of predicted outputs
    :param y_target: list of target outputs
    :param step_size: length of output bins, format [len_x-bins,len_y-bins]
    :return: list of distances in bins between all target and prediction bins
    """
    distance_list = get_distances(y_predicted, y_target, step_size)
    return np.average(distance_list, axis=0)


def get_radius_accuracy(y_predicted, y_target, step_size, absolute_margin=0):
    """
    :param y_predicted: list of predicted outputs
    :param y_target: list of target outputs
    :param step_size: length of output bins, format [len_x-bins,len_y-bins]
    :return: returns percentage of valid vs predicted with absolute distance <= margin in cm
    """
    distance_list = get_radius_distance_list(y_predicted, y_target, step_size, absolute_margin)
    return np.average(distance_list, axis=0)


def get_radius_distance_list(y_predicted, y_target, step_size, absolute_margin=0):
    """

    :param y_predicted: list of predicted outputs
    :param y_target: list of target outputs
    :param step_size: length of output bins, format [len_x-bins,len_y-bins]
    :param absolute_margin:
    :return:
    """
    return np.sqrt(np.square(step_size[0] * (y_predicted[:, 0] - y_target[:, 0])) + np.square(
        step_size[1] * (y_predicted[:, 1] - y_target[:, 1]))) <= absolute_margin


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def average_position(mapping):
    """

    :param mapping: list of positions in map format
    :return: averaged position over [x-coordinate,y-coordinate]
    """

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
    """
    :return: plots binary histogram of positions
    """
    y_predicted, y_target = predict_map(S, sess, X, y)
    distances = get_distances(y_predicted, y_target, [nd.x_step, nd.y_step])
    fig, ax = plt.subplots()
    x_axis_labels = np.linspace(0, 150, 15)
    ax.hist(distances, x_axis_labels, density=True, cumulative=True, rwidth=0.9, color='b')
    ax.grid(c='k', ls='-', alpha=0.3)
    ax.set_xlabel("Prediction error [cm]")
    ax.set_ylabel('Fraction of instances with prediction error')
    fig.tight_layout()
    x_axis_labels = np.linspace(0, 150, 150)
    fig2, ax2 = plt.subplots()
    ax2.hist(distances, x_axis_labels, density=False, rwidth=0.9, color='r')
    ax2.grid(c='k', ls='-', alpha=0.3)
    ax2.set_xlabel('Prediction error [cm]')
    ax2.set_ylabel("Instances with given prediction error")
    ax2.set_ylim([0, 31])
    fig.tight_layout()
    plt.show()
    # plt.savefig(PATH + "images/avg_dist" + "_epoch=" + str(training_step_list[-i]) + ".pdf")
    pass


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
        bin_1 = np.unravel_index(prediction_a.argmax(), prediction_a.shape) # returns the position in bin format
        bin_2 = average_position(y)
        y_predicted[j][0] = bin_1[0]
        y_predicted[j][1] = bin_1[1]
        y_target[j][0] = bin_2[0]
        y_target[j][1] = bin_2[1]
    return y_predicted, y_target


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


def predict_lickwell(S, sess, X, Y, nd):
    y_predicted = np.zeros((len(Y), nd.lw_classifications))
    y_target = np.zeros((len(Y), nd.lw_classifications))
    for j in range(0, len(X), 1):
        x = np.array([data_slice for data_slice in X[j:j + 1]])
        y = np.array(Y[j:j + 1])
        x = np.reshape(x, [1] + [nd.x_shape[1]] + [nd.x_shape[2]] + [nd.x_shape[3]])
        prediction = S.valid(sess, x)[0]
        y_predicted[j] = np.where(prediction == np.max(prediction), 1, 0)  # int(np.argmax(S.valid(sess, x)[0]))
        y_target[j] = y
    return y_predicted, y_target


def return_guesses(y_predicted, y_target, metadata,nd):
    guesses = []
    for i, e in enumerate(metadata):
        prediction_index = np.argmax(y_predicted[i])
        guess_is_correct = True if prediction_index == np.argmax(y_target[i]) else False
        lick_id = e.lick_id
        lickwell = e.lickwell
        prediction = prediction_index + 1 + nd.num_wells-nd.lw_classifications
        guess = Lick_Metric(guess_is_correct=guess_is_correct, lick_id=lick_id, lickwell=lickwell,
                            prediction=prediction)
        guesses.append(guess)
    return guesses


class Lick_Metric_By_Epoch:
    def __init__(self, epoch, guesses=[]):
        self.guesses = guesses
        self.epoch = epoch

    @classmethod
    def test_accuracy(cls, sess, S, nd, X, y, metadata, epoch):
        y_predicted, y_target = predict_lickwell(S=S, sess=sess, X=X, Y=y, nd=nd)
        guesses = return_guesses(y_predicted, y_target, metadata,nd=nd)
        return cls(epoch=epoch, guesses=guesses)


class Lick_Metric:
    def __init__(self, guess_is_correct, lick_id, lickwell, prediction):
        self.guess_is_correct = guess_is_correct
        self.lick_id = lick_id
        self.lickwell = lickwell
        self.prediction = prediction


def get_main_prediction(guesses, nd):
    """

    :param guesses:
    :param nd:
    :return: sorted list of lick_ids, most frequent guess for each lick_id, fraction of guesses being the most frequent guess
    """
    lick_ids = [a.lick_id for a in guesses]
    sorted_lick_ids = np.unique(lick_ids, return_counts=False)
    counter_by_id = np.zeros((len(sorted_lick_ids), nd.lw_classifications))
    # find most frequent prediction by lick_id
    for guess in guesses:
        counter_by_id[np.where(sorted_lick_ids == guess.lick_id)[0][0]][guess.prediction - nd.num_wells+nd.lw_classifications-1] += 1 # converts the guesses back to
    most_frequent_guess = [np.argmax(a) + 2 for a in
                           counter_by_id]  # TODO the + 2 refers to + 1: well id starts at 1 instead of zero, + 1: one well is excluded. No excluded wells parameter in nd yet

    # determine count of most predicted well and total count
    predicted_count = [np.max(a) for a in counter_by_id]
    total_count = [np.sum(a) for a in counter_by_id]

    fraction_predicted_by_id = np.divide(predicted_count, total_count)

    return sorted_lick_ids, most_frequent_guess, fraction_predicted_by_id


def cross_validate_lickwell_data(metrics, epoch, licks, nd):
    lick_ids = [lick.lick_id for lick in licks]
    guesses_by_id = np.zeros(max(lick_ids))
    correct_guesses_by_id = np.zeros(max(lick_ids))
    all_guesses = []
    # Count correct and false guesses sorted by lick id (cross validation normalization is not necessary)
    for k, metrics_k in enumerate(metrics):  # for each cross validation step (to average later)
        guesses = metrics_k[epoch].guesses
        all_guesses.append(guesses)
        for j, guess in enumerate(guesses):  # for each tested sample in epoch
            lick_id = guess.lick_id
            guesses_by_id[lick_id - 1] += 1
            if guess.guess_is_correct is True:
                correct_guesses_by_id[lick_id - 1] += 1
    all_guesses_c = all_guesses
    all_guesses = [a for li in all_guesses for a in li]  # flatten all_guesses

    # find the most frequently predicted well sorted by id
    fraction_decoded_by_id = np.divide(correct_guesses_by_id, guesses_by_id)
    fraction_decoded_by_id = [f for f in fraction_decoded_by_id if not np.isnan(f)]

    sorted_lick_ids, most_frequent_guess_by_id, fraction_predicted_by_id = get_main_prediction(all_guesses, nd)


    # Create list of evaluated licks containing all relevant information
    return_list = []
    for i, lick in enumerate(licks):

        # Determine all parameters for object
        if lick.lick_id in sorted_lick_ids:  # only accept licks confirmed to be in data set
            if i != 0:
                last_lick_id = licks[i - 1].lick_id
            else:
                last_lick_id = None
            if i != len(licks) - 1:
                next_lick_id = licks[i + 1].lick_id
            else:
                next_lick_id = None
            prediction = most_frequent_guess_by_id[np.where(lick.lick_id == sorted_lick_ids)[0][0]]
            fraction_predicted = fraction_predicted_by_id[np.where(lick.lick_id == sorted_lick_ids)[0][0]]
            fraction_decoded = fraction_decoded_by_id[np.where(lick.lick_id == sorted_lick_ids)[0][0]]

            evaluated_lick = Evaluated_Lick(lick_id=lick.lick_id, rewarded=lick.rewarded, time=lick.time,
                                            lickwell=lick.lickwell, prediction=prediction, target=lick.target,
                                            next_lick_id=next_lick_id, last_lick_id=last_lick_id,
                                            fraction_predicted=fraction_predicted,
                                            fraction_decoded=fraction_decoded,
                                            total_decoded=guesses_by_id[lick_id-1], phase=lick.phase,
                                            next_phase=lick.next_phase, last_phase=lick.last_phase)
            return_list.append(evaluated_lick)

    # for lick in return_list:
    #     print(lick.fraction_decoded)
    return return_list, all_guesses_c


def get_lickwell_accuracy(metrics_average):
    accuracy = []
    for metrics in metrics_average:
        accuracy.append(metrics.accuracy)
    return accuracy


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


class Network_output:  # object containing list of metrics by cross validation partition
    def __init__(self, net_data, metric_by_cvs=[], r2_avg=None, ape_avg=None, acc20_avg=None):
        self.metric_by_cvs = metric_by_cvs  # metric by cross validation step
        self.r2_avg = r2_avg
        self.net_data = net_data
        if r2_avg is not None:
            self.r2_avg = r2_avg
        else:
            self.set_average_r2()
        if ape_avg is not None:
            self.acc20_avg = acc20_avg
        else:
            self.set_average_ape()
        if ape_avg is not None:
            self.ape_avg = ape_avg
        else:
            self.set_average_acc20()

    def set_average_r2(self):
        self.r2_avg = np.average([a.r2_best for a in self.metric_by_cvs])

    def set_average_ape(self):
        self.ape_avg = np.average([a.ape_by_epoch[-1] for a in self.metric_by_cvs])

    def set_average_acc20(self):

        self.acc20_avg = np.average([a.acc20_best for a in self.metric_by_cvs])


class Metric:  # object containing evaluation metrics for all epochs
    def __init__(self, r2_by_epoch=[], ape_by_epoch=[], acc20_by_epoch=[], r2_best=None,
                 ape_best=None, acc20_best=None, epoch=None):
        self.r2_by_epoch = r2_by_epoch
        self.ape_by_epoch = ape_by_epoch  # absolute position error
        self.acc20_by_epoch = acc20_by_epoch
        self.r2_best = r2_best
        self.ape_best = ape_best
        self.acc20_best = acc20_best
        self.epoch = epoch

    def set_bests(self, stops_early):
        if stops_early is True:
            self.r2_best = self.r2_by_epoch[-1]
            self.ape_best = self.ape_by_epoch[-1]
            self.acc20_best = self.acc20_by_epoch[-1]
        else:
            self.r2_best = np.max(self.r2_by_epoch)
            self.ape_best = np.min(self.ape_by_epoch)
            self.acc20_best = np.max(self.acc20_by_epoch)

    def set_metrics(self, sess, S, nd, X, y):
        y_predicted, y_target = predict_map(S, sess, X, y)
        a = []
        b = []
        for i, y in enumerate(y_target):
            a.append(y[0])
            b.append(y_predicted[i][0])
        print("average position:", np.average(a), np.average(b))
        r2 = get_r2(y_predicted, y_target)  # r2 score
        ape = get_avg_distance(y_predicted, y_target, [nd.x_step, nd.y_step])  # absolute position error
        acc20 = get_radius_accuracy(y_predicted, y_target, [nd.x_step, nd.y_step], 20)  # accuracy 20
        self.r2_by_epoch.append(r2)
        self.ape_by_epoch.append(ape)
        self.acc20_by_epoch.append(acc20)
        self.epoch = self.epoch


def plot_all_planes(X, y, is_training_data, y_predicted, y_target, nd):
    map_list = y
    distance_list = get_radius_distance_list(y_predicted, y_target, [nd.x_step, nd.y_step],
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
    print("slice_size", nd.slice_size)
    print("y_slice_size", nd.y_slice_size)
    print("stride", nd.stride)
    print("win_size", nd.win_size)
    print("epochs", nd.epochs)
    print("time_shift_steps", nd.time_shift_steps)
    print("shuffle_data", nd.shuffle_data)
    print("shuffle_factor", nd.shuffle_factor)
    print("time_shift_iter", nd.time_shift_iter)
    print("metric_iter", nd.metric_iter)
    print("batch_size", nd.batch_size)
    print("= SEARCH_RADIUS", nd.search_radius)


def plot_plane(y, nd, path):
    pos_x_list = []
    pos_y_list = []
    for i, posxy in enumerate(y):
        pos_x_list.append(posxy[0])
        pos_y_list.append(posxy[1])
    a = position_as_map([pos_x_list, pos_y_list], nd.x_step, nd.y_step, nd.x_max,
                        nd.x_min, nd.y_max, nd.y_min)
    Y = a
    # print("plot:", "sample size is", len(y))
    fig, ax = plt.subplots()
    # fig.suptitle("Samples: "+str(len(y)))
    ax.axis('off')
    fig.subplots_adjust(left=0.01, right=0.99, bottom=0.01, top=0.99, hspace=0.01, wspace=0.01)
    Y -= np.min(y)
    # Y /= np.max(1,np.max(y))
    Y *= 255
    Y[30 // nd.x_step, 50 // nd.y_step] = 30
    Y[70 // nd.x_step, 50 // nd.y_step] = 30
    Y[115 // nd.x_step, 50 // nd.y_step] = 30
    Y[160 // nd.x_step, 50 // nd.y_step] = 30
    Y[205 // nd.x_step, 50 // nd.y_step] = 30

    ax.imshow(Y, cmap="gray")
    plt.savefig(path)
    # plt.close()


def convert_index_list_to_binary_array(list_length, index_list):
    return_list = np.zeros(list_length)
    for li in index_list:
        return_list[li] = 1
    return return_list



class Lick_id_details:
    """
    help class to make lickwell data print and plot easier to read. Contains list of evaluated lick objects and binary
    arrays which indicate if the named parameter applies to each lick in the list
    """

    def __init__(self, licks=None, licks_decoded=None, licks_not_decoded=None, licks_prior_to_switch=None,
                 licks_after_switch=None, filter=None, target_lick_correct=None, target_lick_false=None,
                 next_well_licked_2=None, next_well_licked_3=None, next_well_licked_4=None, next_well_licked_5=None,
                 next_phase_decoded=None, last_phase_decoded=None, next_lick_decoded=None, last_lick_decoded=None,
                 current_phase_decoded=None, fraction_decoded=None, fraction_predicted=None):
        self.licks = licks  # list (of evaluated) licks
        self.target_lick_correct = target_lick_correct  # binary array, 1 if the rat licked correctly at target well of lick in list
        self.target_lick_false = target_lick_false  # 1 if said lick was false
        self.next_well_licked_2 = next_well_licked_2  # 1 if next lick was at well 2
        self.next_well_licked_3 = next_well_licked_3  # 1 if next lick was at well 3
        self.next_well_licked_4 = next_well_licked_4  # 1 if next lick was at well 4
        self.next_well_licked_5 = next_well_licked_5  # 1 if next lick was at well 5
        self.licks_prior_to_switch = licks_prior_to_switch  # 1 if next lick in list is in a different phase
        self.licks_after_switch = licks_after_switch  # 1 if last lick is in a different phase
        self.licks_decoded = licks_decoded  # 1 if lick was decoded correctly
        self.licks_not_decoded = licks_not_decoded  # 1 if lick was not decoded correctly
        self.next_phase_decoded = next_phase_decoded  # 1 if prediction is identical to next phase
        self.last_phase_decoded = last_phase_decoded  # 1 if prediction is identical to last phase
        self.next_lick_decoded = next_lick_decoded  # 1 if prediction is identical to well after (or before) next (or last) lick in future (past) decoding
        self.last_lick_decoded = last_lick_decoded  # s.o.
        self.current_phase_decoded = current_phase_decoded  # 1 if prediction is identical to current phase
        self.filter = filter  # active filter function, is multiplied with the other arrays to filter their output (ie.
        # active filter is target_lick_correct, so self.next_well_licked_2*self.filter gives all correct target licks at well 2 as a new binary array
        self.fraction_decoded = fraction_decoded  # fractional array, indicates which confidence the network has in its guess (what fraction of samples of lick were decoded with active prediction)
        self.fraction_predicted = fraction_predicted  # fractional array, indicates what fraction of samples were decoded as the correct target

    def print_details(self):
        # prints object details filtered with active filter in the format used in the lickwell_prediction excel file
        correct_licks = self.licks_decoded * self.filter
        false_licks = self.licks_not_decoded * self.filter
        licks = []
        for i, li in enumerate(self.filter):
            if li == 1:
                licks.append(self.licks[i])
        licks = np.array(licks)
        last_phase_correct = self.last_phase_decoded * self.filter
        next_phase_correct = self.next_phase_decoded * self.filter
        current_phase_correct = self.current_phase_decoded * self.filter
        last_lick_correct = self.last_lick_decoded * self.filter
        next_lick_correct = self.next_lick_decoded * self.filter
        fraction_decoded = self.fraction_decoded * self.filter
        fraction_predicted = self.fraction_predicted * self.filter
        print(licks.shape[0])
        print(np.sum(correct_licks))
        print(np.sum(false_licks))
        print(np.sum(correct_licks) / (np.sum(false_licks) + np.sum(correct_licks)))
        print(np.sum(next_phase_correct))
        print(np.sum(last_phase_correct))
        print(np.sum(current_phase_correct))
        print(np.sum(current_phase_correct / licks.shape[0]))
        print(np.sum(next_lick_correct))
        print(np.sum(last_lick_correct))
        print(np.sum(fraction_decoded) / np.sum(self.filter))
        print(np.sum(fraction_predicted) / np.sum(self.filter))

    def from_metrics(self, nd, metrics, timeshift, licks):
        """
        :param nd: Net_data
        :param metrics:
        :param timeshift:
        :param licks:
        :return:
        """
        licks_decoded = np.zeros(len(metrics))
        licks_not_decoded = np.zeros(len(metrics))
        licks_prior_to_switch = np.zeros(len(metrics))
        licks_after_switch = np.zeros(len(metrics))
        next_phase_decoded = np.zeros(len(metrics))
        last_phase_decoded = np.zeros(len(metrics))
        current_phase_decoded = np.zeros(len(metrics))
        next_lick_decoded = np.zeros(len(metrics))  # misnomer, actually lick after next or before last lick
        last_lick_decoded = np.zeros(len(metrics))
        fraction_decoded = np.zeros(len(metrics))
        fraction_predicted = np.zeros(len(metrics))
        next_well_licked_2 = np.zeros(len(metrics))
        next_well_licked_3 = np.zeros(len(metrics))
        next_well_licked_4 = np.zeros(len(metrics))
        next_well_licked_5 = np.zeros(len(metrics))
        target_lick_correct = np.zeros(len(metrics))
        target_lick_false = np.zeros(len(metrics))
        for i, lick in enumerate(metrics):
            fraction_predicted[i] = lick.fraction_predicted
            fraction_decoded[i] = lick.fraction_decoded

            # filter by whether next lick is at well n

            if lick.target == 2:
                next_well_licked_2[i] = 1
            if lick.target == 3:
                next_well_licked_3[i] = 1
            if lick.target == 4:
                next_well_licked_4[i] = 1
            if lick.target == 5:
                next_well_licked_5[i] = 1

            # filter by whether prediction was correct
            if lick.prediction == lick.target:
                licks_decoded[i] = 1
            else:
                licks_not_decoded[i] = 1

            # filter by whether lick was prior or after a switch

            if i != len(metrics) - 1:
                next_lick = get_lick_from_id(lick.next_lick_id, licks)
                if next_lick.phase is not None and lick.phase != next_lick.phase:
                    licks_prior_to_switch[i] = 1
                    licks_after_switch[i + 1] = 1

            # filter by whether lick was correctly decoded

            # second next or last lick

            if timeshift == 1:
                target_lick = get_lick_from_id(lick.next_lick_id, licks)
                target_lick = get_lick_from_id(target_lick.lick_id + 1, licks)  # lick after next
            else:
                target_lick = get_lick_from_id(lick.last_lick_id, licks)
                target_lick = get_lick_from_id(target_lick.lick_id - 1, licks)  # lick before last

            if target_lick is not None and target_lick.lickwell == lick.prediction:
                next_lick_decoded[i] = 1

            if timeshift == 1:
                target_lick = get_lick_from_id(lick.last_lick_id, licks)
            else:
                target_lick = get_lick_from_id(lick.next_lick_id, licks)

            if target_lick is not None and target_lick.lickwell == lick.prediction:
                last_lick_decoded[i] = 1

            if lick.prediction == lick.phase:
                current_phase_decoded[i] = 1
            if lick.prediction == lick.next_phase:
                next_phase_decoded[i] = 1
            if lick.prediction == lick.last_phase:
                last_phase_decoded[i] = 1

            # filter by if target lick was correct or not
            if nd.initial_timeshift == 1:
                target_lick = get_lick_from_id(lick.next_lick_id, licks)
            else:
                target_lick = get_lick_from_id(lick.last_lick_id, licks)

            if target_lick.rewarded == 1:
                target_lick_correct[i] = 1
            else:
                target_lick_false[i] = 1

        self.licks = metrics
        self.filter = np.zeros(len(metrics)) + 1
        self.licks_decoded = licks_decoded
        self.licks_not_decoded = licks_not_decoded
        self.licks_prior_to_switch = licks_prior_to_switch
        self.licks_after_switch = licks_after_switch
        self.current_phase_decoded = current_phase_decoded
        self.last_phase_decoded = last_phase_decoded
        self.next_phase_decoded = next_phase_decoded
        self.next_lick_decoded = next_lick_decoded
        self.last_lick_decoded = last_lick_decoded
        self.fraction_decoded = fraction_decoded
        self.fraction_predicted = fraction_predicted
        self.target_lick_correct = target_lick_correct
        self.target_lick_false = target_lick_false
        self.next_well_licked_2 = next_well_licked_2
        self.next_well_licked_3 = next_well_licked_3
        self.next_well_licked_4 = next_well_licked_4
        self.next_well_licked_5 = next_well_licked_5

    def get_next_well_licked_n(self, next_well):
        """"
        return: currently not used. Gives correct filter back
        """
        if next_well == 2:
            return self.next_well_licked_2
        if next_well == 3:
            return self.next_well_licked_3
        if next_well == 4:
            return self.next_well_licked_4
        if next_well == 5:
            return self.next_well_licked_5


def get_metric_details(path, timeshift,pathname_metadata=""):
    metrics = load_pickle(path + "metrics_timeshift=" + str(timeshift) + pathname_metadata + ".pkl")
    metrics_k = load_pickle(path + "metrics_k_timeshift=" + str(timeshift) + pathname_metadata + ".pkl")
    nd = load_pickle(path + "nd_timeshift=" + str(timeshift) + pathname_metadata + ".pkl")
    licks = load_pickle(path + "licks_timeshift=" + str(timeshift) + pathname_metadata +".pkl")
    lick_id_details = Lick_id_details()
    lick_id_details.from_metrics(nd=nd, metrics=metrics, timeshift=timeshift, licks=licks)
    lick_id_details_k = []
    for metric in metrics_k:
        obj = Lick_id_details()
        obj.from_metrics(nd=nd, metrics=metric, timeshift=timeshift, licks=licks)
        lick_id_details_k.append(obj)


    # generate all_guesses_k for standard deviation
    return lick_id_details, lick_id_details_k


def print_metric_details(path, timeshift, pathname_metadata=""):
    """

    :param path: directory of network files, should be filled
    :param timeshift: +1 or - 1 for future or past decoding files
    :param pathname_metadata: if there were multiple experiments in one directory, this can be used to distinguish this by appending to the end of the searched file names
    :return: prints metric details and details by lick
    """
    path = path + "output/"
    # Create binary arrays for licks corresponding to each inspected filter
    metrics = load_pickle(path + "metrics_timeshift=" + str(timeshift) + pathname_metadata +".pkl")
    nd = load_pickle(path + "nd_timeshift=" + str(timeshift)+ pathname_metadata  + ".pkl")
    licks = load_pickle(path + "licks_timeshift=" + str(timeshift) + pathname_metadata + ".pkl")
    lick_id_details = Lick_id_details()
    lick_id_details.from_metrics(nd=nd, metrics=metrics, timeshift=timeshift, licks=licks)
    print("Filter: all licks")
    lick_id_details.print_details()
    print("Filter: correct licks")
    lick_id_details.filter = lick_id_details.target_lick_correct
    lick_id_details.print_details()
    print("Filter: false licks")
    lick_id_details.filter = lick_id_details.target_lick_false
    lick_id_details.print_details()
    print("Filter: licks prior to switch")
    lick_id_details.filter = lick_id_details.licks_prior_to_switch
    lick_id_details.print_details()
    print("Filter: licks after switch")
    lick_id_details.filter = lick_id_details.licks_after_switch
    lick_id_details.print_details()
    print("Filter: next well licked is 2")
    lick_id_details.filter = lick_id_details.next_well_licked_2
    lick_id_details.print_details()
    print("Filter: next well licked is 3")
    lick_id_details.filter = lick_id_details.next_well_licked_3
    lick_id_details.print_details()
    print("Filter: next well licked is 4")
    lick_id_details.filter = lick_id_details.next_well_licked_4
    lick_id_details.print_details()
    print("Filter: next well licked is 5")
    lick_id_details.filter = lick_id_details.next_well_licked_5
    lick_id_details.print_details()
    print_lickwell_metrics(metrics, nd,licks)
    print("Accuracy:", np.sum(lick_id_details.licks_decoded) / (np.sum(lick_id_details.licks_not_decoded) + np.sum(lick_id_details.licks_decoded))
)

def print_lickwell_metrics(metrics_i, nd, licks):
    """
    :param metrics_i: # list of evaluated licks
    :param nd: Net_data object
    :return: prints all data regarding the lick events in the format given in the lickwell prediction excel file
    """
    print(
        "lick_id: well_1->well_ 2, fraction decoded | most frequently predicted , fraction predicted, lick_was_correct")
    metrics = []
    for i, m in enumerate(metrics_i):
        if np.isnan(
                m.fraction_decoded) is not True:  # remove nans if any exist (should be obsolete now due to earlier filtering)
            metrics.append(m)
    for i, m in enumerate(metrics):
        print(m.lick_id, ",", end=" ")
        if nd.initial_timeshift == 1:
            target_lick = get_lick_from_id(m.next_lick_id, licks)
            print(m.target, ",", end=" ")
            after_predicted_lick = get_lick_from_id(m.next_lick_id, licks, shift=0)
            if after_predicted_lick is not None:
                print(after_predicted_lick.target, ",", end=" ")
        else:
            target_lick = get_lick_from_id(m.last_lick_id, licks)
            print(m.target, ",", end=" ")
            before_predicted_lick = get_lick_from_id(m.last_lick_id, licks, shift=0)
            if before_predicted_lick is not None:
                if before_predicted_lick.lick_id - 1 != 0:
                    print(get_lick_from_id(before_predicted_lick.lick_id - 1, licks).lickwell, ",", end=" ")
        print(np.round(m.fraction_decoded, 2), ",", m.prediction, ",", np.round(m.fraction_predicted, 2), ",", end=" ")
        if i != len(metrics) - 1:
            print(target_lick.rewarded)
            # if metrics[i + 1].lick_id in nd.phase_change_ids:
            #     print("->switch")
    pass
