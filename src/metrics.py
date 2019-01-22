import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from src.preprocessing import position_as_map
from src.database_api_beta import Evaluated_Lick
from src.settings import save_as_pickle, load_pickle


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
    distances = get_distances(y_predicted, y_target, [nd.x_step, nd.y_step])
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
        if np.max(prediction) == 0:  # TODO
            y_predicted[j] = np.array([1, 0, 0, 0, 0])
        y_target[j] = y
    return y_predicted, y_target


def return_guesses(y_predicted, y_target, metadata):
    guesses = []
    for i, e in enumerate(metadata):
        prediction = np.argmax(y_predicted[i])
        guess_is_correct = True if prediction == np.argmax(y_target[i]) else False
        lick_id = e.lick_id
        lickwell = e.lickwell
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
        guesses = return_guesses(y_predicted, y_target, metadata)
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
    counter_by_id = np.zeros((len(sorted_lick_ids), nd.num_wells))
    # find most frequent prediction by lick_id
    for guess in guesses:
        counter_by_id[np.where(sorted_lick_ids == guess.lick_id)[0][0]][guess.prediction - 1] += 1
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

    sorted_lick_ids, most_frequent_guess_by_id, fraction_predicted_by_id = get_main_prediction(all_guesses, nd)

    fraction_decoded_by_id = np.divide(correct_guesses_by_id, guesses_by_id)

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

            evaluated_lick = Evaluated_Lick(lick_id=lick.lick_id, rewarded=lick.rewarded, time=lick.time,
                                            lickwell=lick.lickwell, prediction=prediction, target=lick.target,
                                            next_lick_id=next_lick_id, last_lick_id=last_lick_id,
                                            fraction_predicted=fraction_predicted,
                                            fraction_decoded=fraction_decoded_by_id[lick.lick_id - 1],
                                            total_decoded=guesses_by_id[lick.lick_id - 1], phase=lick.phase,
                                            next_phase=lick.next_phase, last_phase=lick.last_phase)
            return_list.append(evaluated_lick)
    return return_list,all_guesses_c


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
        r2 = get_r2(y_predicted, y_target, [nd.x_step, nd.y_step])  # r2 score
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


def get_lick_from_id(id, licks, shift=0):
    """

    :param id: id of lick being searched
    :param licks: list of lick objects
    :param shift: if set, gets lick before or after id'd lick
    :return: corresponding lick
    """
    lick_id_list = [lick.lick_id for lick in licks]
    try:
        return [licks[i + shift] for i, lick in enumerate(licks) if lick.lick_id == id][0]
    except IndexError:
        return None


def convert_index_list_to_binary_array(list_length, index_list):
    return_list = np.zeros(list_length)
    for li in index_list:
        return_list[li] = 1
    return return_list


def logits_to_abs(logits, shift=0):
    """

    :param logits: binary list
    :param shift: added to final value
    :return: cast of binary list to index list. Shifts values by default since lick_id and well_no start at 1
    """
    return_list = []
    for i, li in enumerate(logits):
        if li == 1:
            return_list.append(i + shift)


class Lick_id_details:
    """
    help class to make print_metric_details more readable
    """

    def __init__(self, licks=None, licks_decoded=None, licks_not_decoded=None, licks_prior_to_switch=None,
                 licks_after_switch=None, filter=None, target_lick_correct=None, target_lick_false=None,
                 next_well_licked_2=None, next_well_licked_3=None, next_well_licked_4=None, next_well_licked_5=None,
                 next_phase_decoded=None, last_phase_decoded=None, next_lick_decoded=None, last_lick_decoded=None,
                 current_phase_decoded=None, fraction_decoded=None, fraction_predicted=None):
        self.licks = licks
        self.target_lick_correct = target_lick_correct
        self.target_lick_false = target_lick_false
        self.next_well_licked_2 = next_well_licked_2
        self.next_well_licked_3 = next_well_licked_3
        self.next_well_licked_4 = next_well_licked_4
        self.next_well_licked_5 = next_well_licked_5
        self.licks_prior_to_switch = licks_prior_to_switch
        self.licks_after_switch = licks_after_switch
        self.licks_decoded = licks_decoded
        self.licks_not_decoded = licks_not_decoded
        self.next_phase_decoded = next_phase_decoded
        self.last_phase_decoded = last_phase_decoded
        self.next_lick_decoded = next_lick_decoded
        self.last_lick_decoded = last_lick_decoded
        self.current_phase_decoded = current_phase_decoded
        self.filter = filter
        self.fraction_decoded = fraction_decoded
        self.fraction_predicted = fraction_predicted

    def print_details(self):
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

        # print("Licks in filter:",licks.shape[0])
        # print("Correctly decoded:", np.sum(correct_licks))
        # print("falsely decoded", np.sum(false_licks))
        # print("fraction decoded", np.sum(correct_licks) / (np.sum(false_licks) + np.sum(correct_licks)))
        # print("correctly predicted next phase",np.sum(next_phase_correct))
        # print("correctly predicted last phase",np.sum(last_phase_correct))
        # print("correctly predicted current phase",np.sum(current_phase_correct))
        # print("correctly predicted lick after next lick",np.sum(next_lick_correct))
        # print("correctly predicted last lick",np.sum(last_lick_correct))
        # print("average fraction of samples decoded",np.average(fraction_decoded))
        # print("average fraction of samples with correct well",np.average(fraction_predicted))
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
        if next_well == 2:
            return self.next_well_licked_2
        if next_well == 3:
            return self.next_well_licked_3
        if next_well == 4:
            return self.next_well_licked_4
        if next_well == 5:
            return self.next_well_licked_5


def fraction_decoded_in_array(filter_func, array):
    return np.sum(filter_func * array) / np.sum(filter_func)

def return_guesses_by_lick_id(lick_id_details,all_guesses):
    lick_ids = [lick.lick_id for lick in lick_id_details.licks]
    all_guesses_by_id = []
    all_guesses = [a for li in all_guesses for a in li] # flatten all_guesses
    for lick_id in lick_ids:
        all_guesses_id = []
        for guess in all_guesses:
            if guess.lick_id == lick_id:
                if guess.guess_is_correct is True:
                    all_guesses_id.append(1)
                else:
                    all_guesses_id.append(0)
        all_guesses_by_id.append(all_guesses_id)
    return all_guesses_by_id


def return_fraction_decoded_and_std(lick_id_details, lick_id_details_k,all_guesses_by_id):
    fraction_decoded = fraction_decoded_in_array(lick_id_details.filter,
                                                 lick_id_details.fraction_decoded) # TODO licks_decoded?

    std_lick = []
    for i,guesses in enumerate(all_guesses_by_id):
        if lick_id_details.filter[i]==1: #ignores invalid licks
            average = np.average(guesses)
            values = np.array(guesses)
            variance = np.average((values-average)**2)
            std_lick.append(np.sqrt(variance))

    # fraction_decoded_by_cvs = []
    # weights = []
    # for i, detail in enumerate(lick_id_details_k):
    #     filter_func = detail.filter
    #     weight = np.sum(filter_func)
    #     if weight != 0:  # catch cases where no samples are in cross validation step validation data
    #         fraction_decoded_by_cvs.append(fraction_decoded_in_array(filter_func, detail.licks_decoded))
    #         weights.append(weight)
    # values = np.array(fraction_decoded_by_cvs)
    # average = np.average(values, weights=weights)
    # variance = np.average((values - average) ** 2, weights=weights)
    # std_well = np.sqrt(variance)
    # # std_well = np.std(np.array(fraction_decoded_by_cvs)) # unweighted standard deviation for error bars, currently not used
    std_avg = np.average(std_lick)
    return fraction_decoded, std_avg


def return_fraction_decoded_and_std_by_well(lick_id_details, lick_id_details_k, next_well):
    next_well_licked = lick_id_details.get_next_well_licked_n(next_well)
    fraction_decoded_well_n = fraction_decoded_in_array(next_well_licked,
                                                        lick_id_details.licks_decoded)
    fraction_decoded_by_cvs = []
    weights = []
    for i, detail in enumerate(lick_id_details_k):
        next_well_licked = detail.get_next_well_licked_n(next_well)
        weight = np.sum(next_well_licked)
        if weight != 0:  # catch cases where no samples are in cross validation step validation data
            fraction_decoded_by_cvs.append(fraction_decoded_in_array(next_well_licked, detail.licks_decoded))
            weights.append(weight)
    values = np.array(fraction_decoded_by_cvs)
    average = np.average(values, weights=weights)
    variance = np.average((values - average) ** 2, weights=weights)
    std_well = np.sqrt(variance)
    # std_well = np.std(np.array(fraction_decoded_by_cvs)) # unweighted standard deviation for error bars, currently not used

    return fraction_decoded_well_n, std_well

def plot_accuracy_inside_phase(path,shift,title,save_path,color="darkviolet"):

    # load accuracy data

    metrics = load_pickle(path + "metrics_timeshift=" + str(shift) + ".pkl")

    # plot chart
    sample_counter = np.zeros(1000)
    bin_values = []
    accuracy_sum = np.zeros(1000)
    position = 0
    current_phase = metrics[0].phase
    for i,lick in enumerate(metrics):
        sample_counter[position]+=1
        bin_values.append(position)
        accuracy_sum[position] += lick.fraction_decoded
        position+=1
        if lick.phase!=current_phase: # new phase
            current_phase = lick.phase
            position = 0

    # remove trailing zeros and normalize phase
    sample_counter = np.trim_zeros(sample_counter, 'b')
    accuracy_sum = np.trim_zeros(accuracy_sum,'b')

    y = np.divide(accuracy_sum,sample_counter)
    fig, ax = plt.subplots()
    fontsize=12
    x = np.arange(0,len(y))
    ax.plot(x, y, label='average', color=color,marker='.',linestyle="None") #,linestyle="None"
    ax.legend()
    ax.grid(c='k', ls='-', alpha=0.3)
    ax.set_xlabel("Number of visits of well 1 inside phase")
    ax.set_ylabel("Average fraction of samples decoded correctly")
    ax.set_title(title)
    ax_b = ax.twinx()
    ax_b.set_ylabel("Phases with number of visits")
    z = np.arange(0,12)
    ax_b.hist(bin_values, bins=z, facecolor='g', alpha=0.2)
    # plt.show()
    plt.savefig(save_path)

    pass

def return_sample_count_by_lick_id(lick_id_details_k):
    lick_ids = [[lick.lick_id for lick in detail.licks] for detail in lick_id_details_k]
    total_decoded = [[lick.total_decoded for lick in detail.licks] for detail in lick_id_details_k]
    unique,counts = np.unique(lick_ids, return_counts=True)

    asd = np.sum(lick_id_details_k,axis=0)
    pass


def plot_performance_comparison(path_1,shift_1,path_2,shift_2,title_1,title_2,save_path,barcolor="darkviolet",add_trial_numbers=False):
    # load fraction and std data

    lick_id_details_1,lick_id_details_k_1,all_guesses_by_id_1 = get_metric_details(path_1, shift_1)
    # return_sample_count_by_lick_id(lick_id_details_k_1)
    x_1, std_1 = get_accuracy_for_comparison(lick_id_details_1,lick_id_details_k_1,all_guesses_by_id_1)
    lick_id_details_2,lick_id_details_k_2,all_guesses_by_id_2 = get_metric_details(path_2, shift_2)
    x_2, std_2 = get_accuracy_for_comparison(lick_id_details_2, lick_id_details_k_2,all_guesses_by_id_2)
    std_lower_1,std_upper_1 = get_corrected_std(x_1,std_1)
    std_lower_2,std_upper_2 = get_corrected_std(x_2,std_2)

    # plot bar charts


    width = 0.75
    fontsize=12
    font = {'family': 'normal',
            'size': 12}
    matplotlib.rc('font', **font)
    matplotlib.rc('xtick', labelsize=fontsize-3)

    ind = np.arange(5)  # the x locations for the groups
    fig, (ax1, ax2) = plt.subplots(2)
    ax_b1 = ax1.twinx()
    ax_b2 = ax2.twinx()
    error_kw = {'capsize': 5, 'capthick': 1, 'ecolor': 'black'}
    ax1.bar(ind, x_1, color=barcolor, yerr=[std_lower_1, std_upper_1], error_kw=error_kw, align='center')
    ax1.set_xticks(ind)
    ax1.set_xticklabels(['all licks', 'target correct', 'target false', 'prior switch','after switch'])
    ax_b1.set_ylim(0, 1)
    ax1.set_title(title_1)
    ax1.set_ylabel("fraction decoded correctly",fontsize=fontsize)
    if add_trial_numbers is True:
        for i, j in zip(ind, x_1):
            ax1.annotate(int(x_1[i]), xy=(i+0.05, j+0.05))

    ax2.bar(ind, x_2, color=barcolor, yerr=[std_lower_2, std_upper_2], error_kw=error_kw, align='center')
    ax2.set_xticks(ind)
    ax2.set_xticklabels(['all licks', 'target correct', 'target false', 'prior switch','after switch'])
    ax_b2.set_ylim(0, 1)
    ax2.set_title(title_2)
    ax2.set_ylabel("fraction decoded correctly",fontsize=fontsize)
    if add_trial_numbers is True:
        for i, j in zip(ind, x_2):
            ax2.annotate(int(x_2[i]), xy=(i+0.05, j+0.05))

    plt.tight_layout(pad=0.1, w_pad=0.5, h_pad=0)
    plt.savefig(save_path)
    plt.show()
    pass



def get_accuracy_for_comparison(lick_id_details, lick_id_details_k,all_guesses_by_id):
    # fraction decoded in all licks
    fractions_decoded_all, std_all = return_fraction_decoded_and_std(lick_id_details=lick_id_details,
                                                                     lick_id_details_k=lick_id_details_k,all_guesses_by_id=all_guesses_by_id)

    # fraction decoded if target lick is correct
    lick_id_details.filter = lick_id_details.target_lick_correct
    for i, li in enumerate(lick_id_details_k):
        lick_id_details_k[i].filter = li.target_lick_correct
    fractions_decoded_target_correct, std_target_correct = return_fraction_decoded_and_std(
        lick_id_details=lick_id_details, lick_id_details_k=lick_id_details_k,all_guesses_by_id=all_guesses_by_id)

    # fraction decoded if target lick is false
    lick_id_details.filter = lick_id_details.target_lick_false
    for i, li in enumerate(lick_id_details_k):
        lick_id_details_k[i].filter = li.target_lick_false
    fractions_decoded_target_false, std_target_false = return_fraction_decoded_and_std(lick_id_details=lick_id_details,
                                                                                       lick_id_details_k=lick_id_details_k,all_guesses_by_id=all_guesses_by_id)

    # fraction decoded in licks prior to a switch
    lick_id_details.filter = lick_id_details.licks_prior_to_switch
    for i, li in enumerate(lick_id_details_k):
        lick_id_details_k[i].filter = li.licks_prior_to_switch
    fractions_decoded_licks_prior_to_switch, std_licks_prior_to_switch = return_fraction_decoded_and_std(
        lick_id_details=lick_id_details, lick_id_details_k=lick_id_details_k,all_guesses_by_id=all_guesses_by_id)

    # fraction decoded in licks after a switch
    lick_id_details.filter = lick_id_details.licks_after_switch
    for i, li in enumerate(lick_id_details_k):
        lick_id_details_k[i].filter = li.licks_after_switch
    fractions_decoded_licks_after_switch, std_licks_after_switch = return_fraction_decoded_and_std(
        lick_id_details=lick_id_details, lick_id_details_k=lick_id_details_k,all_guesses_by_id=all_guesses_by_id)

    fra_list = [fractions_decoded_all, fractions_decoded_target_correct, fractions_decoded_target_false,
                fractions_decoded_licks_prior_to_switch, fractions_decoded_licks_after_switch]
    std_list = [std_all, std_target_correct, std_target_false,
                std_licks_prior_to_switch, std_licks_after_switch]
    return fra_list,std_list


def get_corrected_std(bar_values,std_well):
    """
    :param bar_values: list of fractional values for bar charts
    :param std_well: list of corresponding standard deviations
    :return: two standard deviation lists corrected for lower and upper limits 0 and 1 of bar values (so error bars don't go below 0 and above 1
    """
    std_lower = []
    std_upper = []
    for i, std in enumerate(std_well):
        if std + bar_values[i] <= 1:
            std_upper.append(std)
        else:
            std_upper.append(1 - bar_values[i])
        if bar_values[i] - std >= 0:
            std_lower.append(std)
        else:
            std_lower.append(bar_values[i])
    return std_lower,std_upper

def plot_lickwell_performance_comparison(lick_id_details, lick_id_details_k,all_guesses_k,savepath):

    fractions_decoded_2, std_2 = return_fraction_decoded_and_std_by_well(lick_id_details=lick_id_details,
                                                                             lick_id_details_k=lick_id_details_k,
                                                                             next_well=2)
    fractions_decoded_3, std_3 = return_fraction_decoded_and_std_by_well(lick_id_details=lick_id_details,
                                                                         lick_id_details_k=lick_id_details_k,
                                                                         next_well=3)
    fractions_decoded_4, std_4 = return_fraction_decoded_and_std_by_well(lick_id_details=lick_id_details,
                                                                         lick_id_details_k=lick_id_details_k,
                                                                         next_well=4)
    fractions_decoded_5, std_5 = return_fraction_decoded_and_std_by_well(lick_id_details=lick_id_details,
                                                                         lick_id_details_k=lick_id_details_k,
                                                                         next_well=5)

    bar_values = [fractions_decoded_2, fractions_decoded_3, fractions_decoded_4, fractions_decoded_5]
    std_well = [std_2, std_3, std_4, std_5]
    # so error bars show up correctly
    std_lower,std_upper = get_corrected_std(bar_values,std_well)

    # plot bars

    fontsize=12
    font = {'family': 'normal',
            'size': 12}
    matplotlib.rc('font', **font)
    matplotlib.rc('xtick', labelsize=fontsize-3)

    ind = np.arange(4)  # the x locations for the groups

    fig, ax = plt.subplots()
    ax_b = ax.twinx()
    error_kw = {'capsize': 5, 'capthick': 1, 'ecolor': 'black'}
    ax.bar(ind, bar_values, color='lightblue', yerr=[std_lower, std_upper], error_kw=error_kw, align='center')
    ax.set_xticks(ind)
    ax.set_xticklabels(['well 2', 'well 3', 'well 4', 'well 5'])
    ax_b.set_ylim(0, 1)
    ax.set_title("Decoding accuracy by well")
    ax.set_ylabel("fraction of labels decoded correctly",fontsize=fontsize)
    plt.savefig(savepath)
    pass


def get_metric_details(path,timeshift):
    metrics = load_pickle(path + "metrics_timeshift=" + str(timeshift) + ".pkl")
    metrics_k = load_pickle(path + "metrics_k_timeshift=" + str(timeshift) + ".pkl")
    nd = load_pickle(path + "nd_timeshift=" + str(timeshift) + ".pkl")
    licks = load_pickle(path + "licks_timeshift=" + str(timeshift) + ".pkl")
    all_guesses_k = load_pickle(path + "all_guesses_timeshift=" + str(timeshift) + ".pkl")
    lick_id_details = Lick_id_details()
    lick_id_details.from_metrics(nd=nd, metrics=metrics, timeshift=timeshift, licks=licks)
    lick_id_details_k = []
    for metric in metrics_k:
        obj = Lick_id_details()
        obj.from_metrics(nd=nd, metrics=metric, timeshift=timeshift, licks=licks)
        lick_id_details_k.append(obj)
    all_guesses_by_id = return_guesses_by_lick_id(lick_id_details,all_guesses_k)
    return lick_id_details,lick_id_details_k,all_guesses_by_id


def plot_metric_details_by_lickwell(path, timeshift,savepath):
    lick_id_details, lick_id_details_k,all_guesses_k = get_metric_details(path,timeshift)
    plot_lickwell_performance_comparison(lick_id_details, lick_id_details_k,all_guesses_k,savepath)
    pass


def print_metric_details(path, timeshift):
    path = path + "output/"
    # Create binary arrays for licks corresponding to each inspected filter
    metrics = load_pickle(path + "metrics_timeshift=" + str(timeshift) + ".pkl")
    nd = load_pickle(path + "nd_timeshift=" + str(timeshift) + ".pkl")
    licks = load_pickle(path + "licks_timeshift=" + str(timeshift) + ".pkl")
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
    print("fin")


def print_lickwell_metrics(metrics_i, nd, session):
    """

    :param metrics_i:
    :param nd:
    :return:
    """
    licks = session.licks
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
                print(get_lick_from_id(before_predicted_lick.lick_id - 1, licks).lickwell, ",", end=" ")
        print(np.round(m.fraction_decoded, 2), ",", m.prediction, ",", np.round(m.fraction_predicted, 2), ",", end=" ")
        if i != len(metrics) - 1:
            print(target_lick.rewarded)
            # if metrics[i + 1].lick_id in nd.phase_change_ids:
            #     print("->switch")
    pass
