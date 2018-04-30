#Saves and loads parameters used in all networks
import pickle
#GENERAL-------------------------------------------------------------------------------------------------
#type of network being called
network_type = 2 
#key:
#1: n_testing.py (general testing)




def saveParameters():
#saves the parameters in a pickle file
    with open('config.pkl', 'wb') as f:
        pickle.dump([network_type], f)

def loadParameters():
#loads the parameters from file
    with open('config.pkl','rb') as f:
        network_type = pickle.load(f)


