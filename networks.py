from models import SimpleRNNDecoder
from database_api_beta import Slice
import numpy as np
from filters import bin_filter

def get_R2(y_test,y_test_pred):
    R2_list=[] #Initialize a list that will contain the R2s for all the outputs
    for i in range(y_test.shape[1]): #Loop through outputs
        #Compute R2 for each output
        y_mean=np.mean(y_test[:,i])
        R2=1-np.sum((y_test_pred[:,i]-y_test[:,i])**2)/np.sum((y_test[:,i]-y_mean)**2)
        R2_list.append(R2) #Append R2 of this output to the list
    R2_array=np.array(R2_list)
    return R2_array #Return an array of R2s

def test_CNN():
    #Declare model
    model_rnn=SimpleRNNDecoder(units=400,dropout=0,num_epochs=5)
    data_slice = Slice.from_path(load_from="slice.pkl")
    size = len(data_slice.position_x)
    train_slice = data_slice[0:int(size/2)]
    test_slice = data_slice[int(size/2):]
    y_train = train_slice.position_x
    y_valid = test_slice.position_y
    X_train = train_slice.set_filter(filter=bin_filter, window=0)
    X_valid = test_slice.spikes.set_filter(filter=bin_filter, window=0)
    #Fit model
    model_rnn.fit(X_train,y_train)

    #Get predictions
    y_valid_predicted_rnn=model_rnn.predict(X_valid)

    #Get metric of fit
    R2s_rnn=get_R2(y_valid,y_valid_predicted_rnn)
    print('R2s:', R2s_rnn)