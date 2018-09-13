import numpy as np
import matplotlib.pyplot as plt

def get_r2(y_valid,y_pred):
    R2_list=[] #Initialize a list that will contain the R2s for all the outputs
    for i in range(y_valid.shape[1]): #Loop through outputs
        #Compute R2 for each output
        y_mean=np.mean(y_valid[:,i])
        R2=1-np.sum((y_pred[:,i]-y_valid[:,i])**2)/np.sum((y_pred[:,i]-y_mean)**2)
        R2_list.append(R2) #Append R2 of this output to the list
    R2_array=np.array(R2_list)
    return R2_array #Return an array of R2s

def get_avg_distance(y_valid,y_pred,step_size,margin=0):
    distance_list = np.sqrt(np.square(step_size[0]*(y_valid[:,0]-y_pred[:,0])) + np.square(step_size[1]*(y_valid[:,1]-y_pred[:,1])))
    return np.average(distance_list,axis=0)


def get_accuracy(y_valid,y_pred,margin=0):
    """ returns percentage of valid vs predicted with bin distance <= margin"""
    accuracy_list = []
    for i in range(y_valid.shape[1]):
        accuracy_list.append(np.abs(y_valid[:,i]-y_pred[:,i])<=margin)
    return np.average(accuracy_list,axis=1)

def get_radius_accuracy(y_valid,y_pred,step_size,absolute_margin=0):
    """ returns percentage of valid vs predicted with absolute distance <= margin in cm"""
    distance_list = np.sqrt(np.square(step_size[0]*(y_valid[:,0]-y_pred[:,0])) + np.square(step_size[1]*(y_valid[:,1]-y_pred[:,1])))<=absolute_margin
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
        X_test = network_dict["X_eval"]
        y_test = network_dict["y_eval"]
        plot_savefile = "plot_eval_prediction_"
    xshape = [1] + list(X_test[0].shape) + [1]
    yshape = [1] + list(y_test[0].shape) + [1]
    prediction_list = np.zeros([len(y_test), 2])
    actual_list = np.zeros([len(y_test), 2])
    for j in range(0, len(X_test) - 1, 1):
        x = np.array([data_slice for data_slice in X_test[j:j + 1]])
        y = np.array(y_test[j:j + 1])
        x = np.reshape(x, xshape)
        y = np.reshape(y, yshape)
        a = S.eval(sess, x)
        # form softmax of output and remove nan values for columns with only zeros
        # exp_scores = np.exp(a)
        # a = np.nan_to_num(exp_scores / np.sum(exp_scores, axis=1, keepdims=True))
        a = sigmoid(a[0, :, :, 0])
        y = y[0, :, :, 0]
        bin_1 = average_position(a)
        bin_2 = average_position(y)
        prediction_list[j][0] = bin_1[0]
        prediction_list[j][1] = bin_1[1]
        actual_list[j][0] = bin_2[0]
        actual_list[j][1] = bin_2[1]

        if print_distance is True:
            print("prediction:", bin_1)
            print("actual:    ", bin_2)
            print("distance:", bin_distance(bin_1, bin_2))
            print("_____________")
        if j % plot_after_iter == plot_after_iter - 1 and show_plot is True:
            print("plot")
            time = "{:.1f}".format(network_dict["metadata"][j]["lickwells"]["time"] / 1000)
            well = network_dict["metadata"][j]["lickwells"]
            title = "Next lickwell: " + str(well["lickwell"]) + " (" + str(
                well["rewarded"]) + ") in " + time + "s" + " at " + str(
                network_dict["metadata"][j]["position"]) + "cm" + " and " + str(
                network_dict["metadata"][j]["time"])
            y_prime = a
            # fig = plt.figure()
            fig = plt.gcf()
            fig.canvas.set_window_title(title)
            ax1 = fig.add_subplot(1, 2, 1)
            ax2 = fig.add_subplot(1, 2, 2)
            ax1.axis('off')
            ax2.axis('off')
            fig.subplots_adjust(left=0.01, right=0.99, bottom=0.01, top=0.99, hspace=0.01, wspace=0.01)
            Y = y
            # Y -= np.min(Y)
            # Y /= np.max(Y)
            Y *= 255

            Y_prime = a
            Y_prime -= np.min(Y_prime)
            Y_prime /= np.max(Y_prime)
            Y_prime *= 255
            # Y_prime[6, 9] = 30
            # Y_prime[14, 9] = 30
            # Y_prime[23, 9] = 30
            # Y_prime[32, 9] = 30
            # Y_prime[41, 9] = 30
            # Y[6, 9] = 30
            # Y[14, 9] = 30
            # Y[23, 9] = 30
            # Y[32, 9] = 30
            # Y[41, 9] = 30
            ax1.imshow(Y, cmap="gray")
            ax2.imshow(Y_prime, cmap="gray")
            plt.savefig(network_dict["MODEL_PATH"]+"images/"+ plot_savefile + "_shift=" + str(network_dict["TIME_SHIFT"])+ "_epoch=" + str(training_step) + "_img"+ str(j) + ".pdf")
            plt.close()

    r2 = get_r2(actual_list, prediction_list)
    distance = get_avg_distance(prediction_list, actual_list, [network_dict["X_STEP"], network_dict["Y_STEP"]])
    accuracy = []
    for i in range(0, 20):
        # print("accuracy",i,":",get_accuracy(prediction_list, actual_list,margin=i))
        acc = get_radius_accuracy(prediction_list, actual_list, [network_dict["X_STEP"], network_dict["Y_STEP"]], i)
        accuracy.append(acc)
        print("accuracy", i, ":", acc)

    return r2, distance, accuracy


def print_network_dict(network_dict):
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