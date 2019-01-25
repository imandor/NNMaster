import numpy as np
import seaborn as sn
import pandas as pd
import matplotlib.pyplot as plt
from src.settings import  load_pickle



if __name__ == '__main__':

    path = "C:/Users/NN/Desktop/Master/experiments/Lickwell_prediction/"
    model_path_list = [
                       path + "MLP_PFC_lickwell/",
                       path + "MLP_HC_lickwell/"]
    image_title_list = ["pfc","hc"]
    for timeshift in [-1,1]:
        for j,path in enumerate(model_path_list):
            metrics = load_pickle(path + "output/metrics_timeshift=" + str(timeshift) + ".pkl")
            metrics_k = load_pickle(path + "output/metrics_k_timeshift=" + str(timeshift) + ".pkl")
            nd = load_pickle(path + "output/nd_timeshift=" + str(timeshift) + ".pkl")
            licks = load_pickle(path + "output/licks_timeshift=" + str(timeshift) + ".pkl")
            array = np.zeros((4,4))
            for i,lick in enumerate(metrics):
                x = lick.prediction - 2
                if lick.target is not None:
                    y = lick.target - 2
                array[x][y] += 1
            df_cm = pd.DataFrame(array, index = [i for i in [2,3,4,5]],
                              columns = [i for i in [2,3,4,5]])
            plt.figure(figsize = (10,7))
            plt.title('Predicted vs. target label',fontsize=16)
            ax = sn.heatmap(df_cm, annot=True)
            ax.set_xlabel("Target well",fontsize=16)
            ax.set_ylabel("Predicted well",fontsize=16)
            if timeshift == -1:
                shift = "last"
            else:
                shift = "next"
            plt.savefig(path+"images/confusion_matrix_"+image_title_list[j]+"_"+shift)
