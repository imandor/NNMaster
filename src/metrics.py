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

def get_avg_distance(y_valid,y_pred,step_size,margin=0):
    distance_list = []
    for i in range(y_valid.shape[1]):
        distance_list.append(np.abs(step_size[i]*(y_valid[:,i]-y_pred[:,i])))
    return np.average(np.linalg.norm(distance_list,axis=0))


def get_accuracy(y_valid,y_pred,margin=0):
    accuracy_list = []
    for i in range(y_valid.shape[1]):
        accuracy_list.append(np.abs(y_valid[:,i]-y_pred[:,i])<=margin)
    return np.average(accuracy_list,axis=1)


def bin_distance(bin_1,bin_2):
    return [abs(bin_1[1]-bin_2[1]),abs(bin_1[2]-bin_2[2])]