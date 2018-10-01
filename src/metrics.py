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

def get_distances(y_actual,y_pred,step_size):
    return np.sqrt(np.square(step_size[0] * (y_actual[:, 0] - y_pred[:, 0])) + np.square(step_size[1] * (y_actual[:, 1] - y_pred[:, 1])))

def get_avg_distance(y_actual, y_pred, step_size, margin=0):
    distance_list = get_distances(y_actual,y_pred,step_size)
    return np.average(distance_list,axis=0)


def get_accuracy(y_actual, y_pred, margin=0):
    """ returns percentage of valid vs predicted with bin distance <= margin"""
    accuracy_list = []
    for i in range(y_actual.shape[1]):
        accuracy_list.append(np.abs(y_actual[:, i] - y_pred[:, i]) <= margin)
    return np.average(accuracy_list,axis=1)

def get_radius_accuracy(y_actual, y_pred, step_size, absolute_margin=0):
    """ returns percentage of valid vs predicted with absolute distance <= margin in cm"""
    distance_list = np.sqrt(np.square(step_size[0] * (y_actual[:, 0] - y_pred[:, 0])) + np.square(step_size[1] * (y_actual[:, 1] - y_pred[:, 1]))) <= absolute_margin
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
        print("Value error, check validation output")
        return "Value error, check validation output"


def plot_histogram(sess,S,net_dict,X,y):

    prediction_list,actual_list = predict(S,sess,X,y)
    distances = get_distances(prediction_list,actual_list,[net_dict["X_STEP"], net_dict["Y_STEP"]])
    x_axis_labels = np.linspace(0,150,150)
    fig, ax = plt.subplots()
    # ax.plot(time_shift_list,distance_scores_train,label='Training set',color='r')
    ax.hist(distances,x_axis_labels,density=True, cumulative= True,rwidth=0.9,color='r')
    ax.legend('total samples ='+ str(len(y)))
    ax.grid(c='k', ls='-', alpha=0.3)
    # ax.set_title(r'$\varnothing$distance of validation wrt time-shift')
    ax.set_xlabel("Percentage of samples")
    ax.set_ylabel('Prediction error [cm]')
    fig.tight_layout()

    fig2, ax2 = plt.subplots()
    # ax.plot(time_shift_list,distance_scores_train,label='Training set',color='r')
    ax2.hist(distances, x_axis_labels, density=False, rwidth=0.9, color='r')
    ax2.legend('total samples ='+ str(len(y)))
    ax2.grid(c='k', ls='-', alpha=0.3)
    # ax.set_title(r'$\varnothing$distance of validation wrt time-shift')
    ax2.set_xlabel("Sample count")
    ax2.set_ylabel('Prediction error [cm]')
    fig.tight_layout()
    plt.show()


    plt.show()


    # plt.savefig(PATH + "images/avg_dist" + "_epoch=" + str(training_step_list[-i]) + ".pdf")


def predict(S,sess,X,Y):
    """ returns list of x/y tuples for predicted and actual positions"""
    xshape = [1] + list(X[0].shape) + [1]
    yshape = [1] + list(Y[0].shape) + [1]
    prediction_list = np.zeros([len(Y), 2])
    actual_list = np.zeros([len(Y), 2])
    for j in range(0, len(X) - 1, 1):
        x = np.array([data_slice for data_slice in X[j:j + 1]])
        y = np.array(Y[j:j + 1])
        x = np.reshape(x, xshape)
        y = np.reshape(y, yshape)
        prediction = S.valid(sess, x)
        a = sigmoid(prediction[0, :, :, 0])
        y = y[0, :, :, 0]
        bin_1 = average_position(a)
        bin_2 = average_position(y)
        prediction_list[j][0] = bin_1[0]
        prediction_list[j][1] = bin_1[1]
        actual_list[j][0] = bin_2[0]
        actual_list[j][1] = bin_2[1]
    return prediction_list,actual_list


def test_accuracy(sess, S, net_dict, is_training_data=False,  print_distance=False):
    if is_training_data:
        X = net_dict["X_train"]
        y = net_dict["y_train"]
    else:
        X = net_dict["X_valid"]
        y = net_dict["y_valid"]

    prediction_list,actual_list = predict(S,sess,X,y)
    r2 = get_r2(actual_list, prediction_list)
    distance = get_avg_distance(prediction_list, actual_list, [net_dict["X_STEP"], net_dict["Y_STEP"]])
    accuracy = []
    for i in range(0, 20):
        # print("accuracy",i,":",get_accuracy(prediction_list, actual_list,margin=i))
        acc = get_radius_accuracy(prediction_list, actual_list, [net_dict["X_STEP"], net_dict["Y_STEP"]], i)
        accuracy.append(acc)
        if i == 19 and print_distance is True: print("accuracy", i, ":", acc)
    return r2, distance, accuracy


def print_net_dict(net_dict):
    print("session_filter",net_dict["session_filter"])
    print("EPOCHS",net_dict["EPOCHS"])
    print("TIME_SHIFT_STEPS",net_dict["TIME_SHIFT_STEPS"])
    print("SHUFFLE_DATA",net_dict["SHUFFLE_DATA"])
    print("SHUFFLE_FACTOR",net_dict["SHUFFLE_FACTOR"])
    print("TIME_SHIFT_ITER",net_dict["TIME_SHIFT_ITER"])
    print("METRIC_ITER",net_dict["METRIC_ITER"])
    print("BATCH_SIZE",net_dict["BATCH_SIZE"])
    print("BINS_BEFORE",net_dict["BINS_BEFORE"])
    print("BINS_AFTER",net_dict["BINS_AFTER"])
    print("SLICE_SIZE",net_dict["SLICE_SIZE"])
    print("= WIN_SIZE",net_dict["WIN_SIZE"])
    print("= SEARCH_RADIUS",net_dict["SEARCH_RADIUS"])