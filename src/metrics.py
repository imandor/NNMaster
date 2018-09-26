import numpy as np
import matplotlib.pyplot as plt

def get_r2(y_actual, y_pred):
    R2_list=[] #Initialize a list that will contain the R2s for all the outputs
    for i in range(y_actual.shape[1]): #Loop through outputs
        #Compute R2 for each output
        y_mean=np.mean(y_actual[:, i])
        R2= 1 - np.sum((y_pred[:,i] - y_actual[:, i]) ** 2) / np.sum((y_pred[:, i] - y_mean) ** 2)
        R2_list.append(R2) #Append R2 of this output to the list
    R2_array=np.array(R2_list)
    return R2_array #Return an array of R2s

def get_avg_distance(y_actual, y_pred, margin=0):
    distance_list = np.sqrt(np.square(y_actual[:, 0] - y_pred[:, 0]) + np.square(y_actual[:, 1] - y_pred[:, 1]))
    return np.average(distance_list,axis=0)


def get_accuracy(y_actual, y_pred, margin=0):
    """ returns percentage of valid vs predicted with bin distance <= margin"""
    accuracy_list = []
    for i in range(y_actual.shape[1]):
        accuracy_list.append(np.abs(y_actual[:, i] - y_pred[:, i]) <= margin)
    return np.average(accuracy_list,axis=1)

def get_radius_accuracy(y_actual, y_pred,  absolute_margin=0):
    """ returns percentage of valid vs predicted with absolute distance <= margin in cm"""
    asd = np.sqrt(np.square( (y_actual[:, 0] - y_pred[:, 0])) + np.square((y_actual[:, 1] - y_pred[:, 1])))
    distance_list = np.sqrt(np.square( (y_actual[:, 0] - y_pred[:, 0])) + np.square((y_actual[:, 1] - y_pred[:, 1]))) <= absolute_margin
    return np.average(distance_list,axis=0)


def bin_distance(bin_1,bin_2):
    return [abs(bin_1[0]-bin_2[0]),abs(bin_1[1]-bin_2[1])]


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
        print("asd")
        return False


def test_accuracy(sess, S, network_dict, training_step, is_training_data=False, show_plot=False, plot_after_iter=1, print_distance=False):
    if is_training_data:
        X_test = network_dict["X_train"]
        y_test = network_dict["y_train"]
        plot_savefile = "plot_train_prediction_"
    else:
        X_test = network_dict["X_valid"]
        y_test = network_dict["y_valid"]
        plot_savefile = "plot_valid_prediction_"
    xshape = [1] + list(X_test[0].shape) + [1]
    yshape = [1]  + [2]
    prediction_list = np.zeros([len(y_test), 2])
    actual_list = np.zeros([len(y_test), 2])
    for j in range(0, len(X_test) - 1, 1):
        x = np.array([data_slice for data_slice in X_test[j:j + 1]])
        y = np.array(y_test[j:j + 1])
        x = np.reshape(x, xshape)
        y = np.reshape(y, yshape)
        a = S.valid(sess, x)
        # form softmax of output and remove nan values for columns with only zeros
        # exp_scores = np.exp(a)
        # a = np.nan_to_num(exp_scores / np.sum(exp_scores, axis=1, keepdims=True))
        prediction_list[j][0] = a[0][0]
        prediction_list[j][1] = a[0][1]
        actual_list[j][0] = y[0][0]
        actual_list[j][1] = y[0][1]


    r2 = get_r2(actual_list, prediction_list)
    distance = get_avg_distance(prediction_list, actual_list, [network_dict["X_STEP"], network_dict["Y_STEP"]])
    accuracy = []
    for i in range(0, 20):
        # print("accuracy",i,":",get_accuracy(prediction_list, actual_list,margin=i))
        acc = get_radius_accuracy(prediction_list, actual_list, i)
        accuracy.append(acc)
        if i == 19 and print_distance is True: print("accuracy", i, ":", acc)

    return r2, distance, accuracy


def print_net_dict(network_dict):
    print("session_filter",network_dict["session_filter"])
    print("EPOCHS",network_dict["EPOCHS"])
    print("TIME_SHIFT_STEPS",network_dict["TIME_SHIFT_STEPS"])
    print("SHUFFLE_DATA",network_dict["SHUFFLE_DATA"])
    print("SHUFFLE_FACTOR",network_dict["SHUFFLE_FACTOR"])
    print("TIME_SHIFT_ITER",network_dict["TIME_SHIFT_ITER"])
    print("METRIC_ITER",network_dict["METRIC_ITER"])
    print("BATCH_SIZE",network_dict["BATCH_SIZE"])
    print("BINS_BEFORE",network_dict["BINS_BEFORE"])
    print("BINS_AFTER",network_dict["BINS_AFTER"])
    print("SLICE_SIZE",network_dict["SLICE_SIZE"])
    print("= WIN_SIZE",network_dict["WIN_SIZE"])
    print("= SEARCH_RADIUS",network_dict["SEARCH_RADIUS"])