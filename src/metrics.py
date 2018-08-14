import numpy as np


def get_r2(y_valid,y_pred):
    R2_list=[] #Initialize a list that will contain the R2s for all the outputs
    for i in range(y_valid.shape[1]): #Loop through outputs
        #Compute R2 for each output
        y_mean=np.mean(y_valid[:,i])
        R2=1-np.sum((y_pred[:,i]-y_valid[:,i])**2)/np.sum((y_valid[:,i]-y_mean)**2)
        R2_list.append(R2) #Append R2 of this output to the list
    R2_array=np.array(R2_list)
    return R2_array #Return an array of R2s