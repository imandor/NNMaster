import numpy as np
import seaborn as sn
import pandas as pd
import matplotlib.pyplot as plt
from src.settings import  load_pickle
from src.metrics import get_lick_from_id


if __name__ == '__main__':
    exclude_phase_change_trials = True
    path = "C:/Users/NN/Desktop/Master/experiments/Lickwell_prediction/"
    model_path_list = [
                       path + "MLP_PFC/",
                       path + "MLP_HC/"]
    image_title_list = ["pfc","hc"]
    for timeshift in [-1,1]:
        for j,path in enumerate(model_path_list):
            metrics = load_pickle(path + "output/metrics_timeshift=" + str(timeshift) + ".pkl")
            metrics_k = load_pickle(path + "output/metrics_k_timeshift=" + str(timeshift) + ".pkl")
            nd = load_pickle(path + "output/nd_timeshift=" + str(timeshift) + ".pkl")
            licks = load_pickle(path + "output/licks_timeshift=" + str(timeshift) + ".pkl")
            array = np.zeros((4,4))
            metrics_flattened = [item for sublist in metrics_k for item in sublist]



            # metrics_a = load_pickle(model_path_list[0] + "output/metrics_k_timeshift=" + str(timeshift) + ".pkl")
            # metrics_b = load_pickle(model_path_list[1] + "output/metrics_k_timeshift=" + str(timeshift) + ".pkl")
            # metrics_af = np.reshape(metrics_a,-1)
            # metrics_bf = np.reshape(metrics_b,-1)



            for i,lick in enumerate(metrics_flattened): # TODO change back
                if exclude_phase_change_trials is True:
                    next_lick = get_lick_from_id(lick.next_lick_id, licks, shift=0,get_next_best=True,dir=1)
                    last_lick = get_lick_from_id(lick.last_lick_id, licks, shift=0, get_next_best=True, dir=-1)
                    if (last_lick is not None and last_lick.phase!=lick.phase) or (next_lick is not None and next_lick.phase!=lick.phase):
                        continue
                x = lick.prediction - 2
                if lick.target is not None:
                    y = lick.target - 2
                array[x][y] += 1
            df_cm = pd.DataFrame(array, index = [i for i in [2,3,4,5]],
                              columns = [i for i in [2,3,4,5]])
            plt.figure(figsize = (10,7))
            plt.title('decoded vs. target label',fontsize=16)
            ax = sn.heatmap(df_cm, annot=True)
            ax.set_xlabel("target well",fontsize=16)
            ax.set_ylabel("decoded well",fontsize=16)
            if timeshift == -1:
                shift = "last"
            else:
                shift = "next"
            pathname_metadata = ""
            if exclude_phase_change_trials is True:
                pathname_metadata = "_exclude_phasechange"
            plt.savefig(path+"images/confusion_matrix_"+image_title_list[j]+"_"+shift+pathname_metadata)
