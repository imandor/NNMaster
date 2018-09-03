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
    return [abs(bin_1[1]-bin_2[1]),abs(bin_1[2]-bin_2[2])]