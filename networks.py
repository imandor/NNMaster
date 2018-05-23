from models import SimpleRNNDecoder
from session_loader import make_dense_np_matrix
from database_api_beta import Slice
import numpy as np
from settings import save_as_pickle, load_pickle
from filters import bin_filter


def get_R2(y_test, y_test_pred):
    R2_list = []  # Initialize a list that will contain the R2s for all the outputs
    for i in range(y_test.shape[1]):  # Loop through outputs
        # Compute R2 for each output
        y_mean = np.mean(y_test[:, i])
        R2 = 1 - np.sum((y_test_pred[:, i] - y_test[:, i]) ** 2) / np.sum((y_test[:, i] - y_mean) ** 2)
        R2_list.append(R2)  # Append R2 of this output to the list
    R2_array = np.array(R2_list)
    return R2_array  # Return an array of R2s

def process_input(X_train):
    return_list = []
    for i in range(0,len(X_train[0])):
        neuron_list = []
        for j in range(0,len(X_train)):
            neuron_list.append(X_train[j][i])
        return_list.append([neuron_list])
    return return_list


def test_CNN():
    # Declare model
    model_rnn = SimpleRNNDecoder(units=400, dropout=0, num_epochs=5)



    data_slice = Slice.from_path(load_from="slice.pkl")
    data_slice = data_slice[0:200]
    maximum_value = len(data_slice.position_x)
    size = len(data_slice.position_x)
    train_slice = data_slice[0:int(size / 2)]
    test_slice = data_slice[int(size / 2):]
    y_train = list(zip(train_slice.position_x,train_slice.position_y))
    y_valid = list(zip(test_slice.position_x,test_slice.position_y))

    X_train = make_dense_np_matrix(mat=train_slice.spikes, minimum_value=0, maximum_value=int(maximum_value / 2))
    X_valid = make_dense_np_matrix(mat=train_slice.spikes, minimum_value=int(maximum_value / 2),
                                   maximum_value=maximum_value)
    # Fit model


    X_train = process_input(X_train)
    X_valid = process_input(X_valid)
    X_train = np.asarray(X_train)
    X_valid = np.asarray(X_valid)
    y_train = np.asarray(y_train)
    y_valid = np.asarray(y_valid)
    model_data = dict(
        X_train=X_train,
        X_valid=X_valid,
        y_train=y_train,
        y_valid=y_valid
    )
    save_as_pickle("model_data.pkl", model_data)

    model_data = load_pickle("model_data.pkl")
    X_train = model_data["X_train"]
    X_valid = model_data["X_valid"]
    y_train = model_data["y_train"]
    y_valid = model_data["y_valid"]
    model_rnn.fit(X_train, y_train)
    save_as_pickle("model_2018-5-23.pkl", model_data)
    # Get predictions
    y_valid_predicted_rnn = model_rnn.predict(X_valid)

    # Get metric of fit
    R2s_rnn = get_R2(y_valid, y_valid_predicted_rnn)
    print('R2s:', R2s_rnn)