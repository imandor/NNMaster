import numpy as np
import matplotlib.pyplot as plt
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

    all_guesses = [a for li in all_guesses for a in li]  # flatten all_guesses

    # find the most frequently predicted well sorted by id

    sorted_lick_ids, most_frequent_guess_by_id, fraction_predicted_by_id = get_main_prediction(all_guesses, nd)

    fraction_decoded_by_id = np.divide(correct_guesses_by_id, guesses_by_id)

    # Create list of objects containing all relevant information TODO: change to Lick object
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
                                            total_decoded=guesses_by_id[lick.lick_id - 1])
            return_list.append(evaluated_lick)
    return return_list


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
        return [licks[i+shift] for i,lick in enumerate(licks) if lick.lick_id == id][0]
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

    def __init__(self, licks, correct_licks=None, false_licks=None, prior_phase_change=None, post_phase_change=None,
                 filter=None,
                 next_phase_correct=None, last_phase_correct=None, next_lick_correct=None, last_lick_correct=None,
                 current_phase_correct=None):
        self.licks = licks
        self.correct_licks = correct_licks
        self.false_licks = false_licks
        self.next_phase_correct = next_phase_correct
        self.last_phase_correct = last_phase_correct
        self.next_lick_correct = next_lick_correct
        self.last_lick_correct = last_lick_correct
        self.current_phase_correct = current_phase_correct

        self.filter = filter

    def print_details(self):

        correct_licks = self.correct_licks * self.filter
        false_licks = self.false_licks * self.filter
        licks = []
        for i, li in enumerate(self.filter):
            if li == 1:
                licks.append(self.licks[i])
        last_phase_correct = self.last_phase_correct * self.filter
        next_phase_correct = self.next_phase_correct * self.filter
        current_phase_correct = self.current_phase_correct * self.filter
        last_lick_correct = self.last_lick_correct * self.filter
        next_lick_correct = self.next_lick_correct * self.filter

        print("Correctly decoded:", np.sum(correct_licks))
        print("falsely decoded", np.sum(false_licks))
        print("fraction decoded", np.sum(correct_licks) / (np.sum(false_licks) + np.sum(correct_licks)))
        print("correctly predicted next phase",np.sum(next_phase_correct))
        print("correctly predicted last phase",np.sum(last_phase_correct))
        print("correctly predicted current phase",np.sum(current_phase_correct))
        print("correctly predicted lick after current",np.sum(next_lick_correct))
        print("correctly predicted last lick",np.sum(last_lick_correct))
        print("average fraction decoded")
        print("average fraction of correct answers")


def print_metric_details(path, nd):
    # Create binary arrays for licks corresponding to each inspected filter
    metrics = load_pickle(path)

    # for metric in metrics:
    #     print(metric.get_phase(nd, shift=1))

    # correct wells, false wells, licks prior to and after switch
    correct_licks = np.zeros(len(metrics))
    false_licks = np.zeros(len(metrics))
    prior_switch_licks = np.zeros(len(metrics))
    after_switch_licks = np.zeros(len(metrics))
    next_phase_correct = np.zeros(len(metrics))
    last_phase_correct = np.zeros(len(metrics))
    current_phase_correct = np.zeros(len(metrics))
    next_lick_correct = np.zeros(len(metrics))
    last_lick_correct = np.zeros(len(metrics))
    for i, lick in enumerate(metrics):
        if lick.prediction == lick.target:
            correct_licks[i] = 1
        else:
            false_licks[i] = 1
        if i != len(metrics) - 1:
            if lick.next_lick_id in nd.phase_change_ids:
                print("->switch")
                prior_switch_licks[i] = 1
                after_switch_licks[i + 1] = 1


        next_lick = get_lick_from_id(lick.lick_id, nd.licks, shift=1)
        if next_lick is not None and lick.prediction == lick.target:  # if lick predicted was next lick
            next_lick_correct[i] = 1
        last_lick = get_lick_from_id(lick.lick_id, nd.licks, shift=-1)
        if last_lick is not None and lick.prediction == last_lick.lickwell:  # if lick predicted was last lick
            last_lick_correct[i] = 1
        if lick.prediction == lick.get_phase(nd, 0):
            current_phase_correct[i] = 1
        if lick.prediction == lick.get_phase(nd, 1):
            asd = lick.get_phase(nd,1)
            next_phase_correct[i] = 1
            asd = lick.get_phase(nd,-1)
        if lick.prediction == lick.get_phase(nd, -1):
            last_phase_correct[i] = 1
        print("---")
        print("lick id:",lick.lick_id)
        print("next lick:",lick.target)
        print("prediction:", lick.prediction)
        if next_lick is not None: print("after next lick:",next_lick.lickwell)
        if last_lick is not None: print("last lick:",last_lick.lickwell)
        print("current phase:",lick.get_phase(nd, 0))
        print("last phase:",lick.get_phase(nd, -1))
        print("next phase:",lick.get_phase(nd, 1))
    lick_id_details = Lick_id_details(licks=metrics, filter=np.zeros(len(metrics)) + 1, correct_licks=correct_licks,
                                      false_licks=false_licks, prior_phase_change=prior_switch_licks,
                                      post_phase_change=after_switch_licks,
                                      current_phase_correct=current_phase_correct,
                                      last_phase_correct=last_phase_correct,
                                      next_phase_correct=next_phase_correct, next_lick_correct=next_lick_correct,
                                      last_lick_correct=last_lick_correct)
    # convert_index_list_to_binary_array(list_length, list)

    lick_id_details.print_details()
    print("fin")


def print_lickwell_metrics(metrics_i, nd, licks):
    """

    :param metrics_i:
    :param nd:
    :return:
    """
    print(
        "lick_id: well_1->well_ 2, fraction decoded | most frequently predicted , fraction predicted, lick_was_correct")
    metrics = []
    for i, m in enumerate(metrics_i):
        if np.isnan(
                m.fraction_decoded) is not True:  # remove nans if any exist (should be obsolete now due to earlier filtering)
            metrics.append(m)
    for i, m in enumerate(metrics):
        print(m.lick_id, ",", m.lickwell, ",", end=" ")
        if nd.initial_timeshift == 1:
            predicted_lick = get_lick_from_id(m.next_lick_id, licks)
            print(predicted_lick.lickwell, ",", end=" ")
            after_predicted_lick = get_lick_from_id(m.next_lick_id, licks, shift=1)
            if after_predicted_lick is not None:
                print(after_predicted_lick.lickwell, ",", end=" ")
        else:
            predicted_lick = get_lick_from_id(m.last_lick_id, licks)
            print(predicted_lick.lickwell, ",", end=" ")
            before_predicted_lick = get_lick_from_id(m.last_lick_id, licks, shift=-1)
            if before_predicted_lick is not None:
                print(before_predicted_lick.lickwell, ",", end=" ")
        print(np.round(m.fraction_decoded, 2), ",", m.prediction, ",", np.round(m.fraction_predicted, 2), ",", end=" ")
        if i != len(metrics) - 1:
            print(predicted_lick.rewarded)
            if metrics[i + 1].lick_id in nd.phase_change_ids:
                print("->switch")
    pass
