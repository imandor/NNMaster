import matplotlib
import numpy as np
from matplotlib import pyplot as plt

from src.plots import metric_details_by_lickwell






if __name__ == '__main__':
    model_path_list = ["C:/Users/NN/Desktop/Master/experiments/Lickwell_prediction/MLP_PFC/",
                       "C:/Users/NN/Desktop/Master/experiments/Lickwell_prediction/MLP_PFC_more_epochs/"
                       ]
    image_save_path = "C:/Users/NN/Desktop/Master/experiments/Lickwell_prediction/compare_wells.png"
    add_trial_numbers = True
    bar_values_1,std_lower_1,std_upper_1,n_well_1 = metric_details_by_lickwell(model_path_list[0],1)
    bar_values_2,std_lower_2,std_upper_2,n_well_2 = metric_details_by_lickwell(model_path_list[0],-1)
    bar_values_3,std_lower_3,std_upper_3,n_well_3 = metric_details_by_lickwell(model_path_list[1],1)
    bar_values_4,std_lower_4,std_upper_4,n_well_4 = metric_details_by_lickwell(model_path_list[1],-1)

    fontsize = 16
    font = {'family': 'normal',
            'size': 12}
    matplotlib.rc('font', **font)
    matplotlib.rc('xtick', labelsize=fontsize - 3)

    ind = np.arange(4)  # the x locations for the groups

    fig, ((ax_1,ax_2),(ax_3,ax_4)) = plt.subplots(2,2)
    error_kw = {'capsize': 5, 'capthick': 1, 'ecolor': 'black'}


    ax_1.bar(ind, bar_values_1, color='r', yerr=[std_lower_1, std_upper_1], error_kw=error_kw, align='center',label="PFC 20 epochs")
    ax_1.set_xticks(ind)
    ax_1.set_xticklabels(['well 2', 'well 3', 'well 4', 'well 5'],fontsize=fontsize)
    ax_1.set_title("next well accuracy", fontsize=fontsize)
    ax_1.legend()
    ax_1.set_ylabel("fraction decoded")
    if add_trial_numbers is True:
        for i, j in zip(ind, bar_values_1):
            if j < 0.2:
                offset = 0.1
            else:
                offset = -0.1
            ax_1.annotate(int(n_well_1[i]), xy=(i - 0.1, j + offset))

    ax_2.bar(ind, bar_values_2, color='r', yerr=[std_lower_2, std_upper_2], error_kw=error_kw, align='center',label="PFC 20 epochs")
    ax_2.set_xticks(ind)
    ax_2.set_xticklabels(['well 2', 'well 3', 'well 4', 'well 5'], fontsize=fontsize)
    ax_2.set_title("previous well accuracy", fontsize=fontsize)
    ax_2.legend()
    if add_trial_numbers is True:
        for i, j in zip(ind, bar_values_2):
            if j < 0.2:
                offset = 0.1
            else:
                offset = -0.1
            ax_2.annotate(int(n_well_2[i]), xy=(i - 0.1, j + offset))

    ax_3.bar(ind, bar_values_3, color='darkred', yerr=[std_lower_3, std_upper_3], error_kw=error_kw, align='center',label="PFC 40 epochs")
    ax_3.set_xticks(ind)
    ax_3.set_xticklabels(['well 2', 'well 3', 'well 4', 'well 5'], fontsize=fontsize)
    ax_3.set_ylabel("fraction decoded")
    ax_3.legend()
    if add_trial_numbers is True:
        for i, j in zip(ind, bar_values_3):
            if j < 0.2:
                offset = 0.1
            else:
                offset = -0.1
            ax_3.annotate(int(n_well_3[i]), xy=(i - 0.1, j + offset))
    ax_4.bar(ind, bar_values_4, color='darkred', yerr=[std_lower_4, std_upper_4], error_kw=error_kw, align='center',label="PFC 40 epochs")
    ax_4.set_xticks(ind)
    ax_4.set_xticklabels(['well 2', 'well 3', 'well 4', 'well 5'], fontsize=fontsize)
    ax_4.legend()
    if add_trial_numbers is True:
        for i, j in zip(ind, bar_values_4):
            if j < 0.2:
                offset = 0.1
            else:
                offset = -0.1
            ax_4.annotate(int(n_well_4[i]), xy=(i - 0.1, j + offset))


    ax_1.set_ylim(0,1)
    ax_2.set_ylim(0,1)
    ax_3.set_ylim(0,1)
    ax_4.set_ylim(0,1)
    plt.show()
    plt.savefig(image_save_path)