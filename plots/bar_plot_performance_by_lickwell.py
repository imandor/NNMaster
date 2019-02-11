from src.plots import plot_metric_details_by_lickwell

if __name__ == '__main__':
    model_path_list = ["C:/Users/NN/Desktop/Master/experiments/Lickwell_prediction/MLP_PFC_phasetarget/",
                       "C:/Users/NN/Desktop/Master/experiments/Lickwell_prediction/MLP_HC_phasetarget/",
                       "C:/Users/NN/Desktop/Master/experiments/Lickwell_prediction/MLP_HC_lickwell/",
                       "C:/Users/NN/Desktop/Master/experiments/Lickwell_prediction/MLP_PFC_lickwell/",
                       ]
    save_path = "C:/Users/NN/Desktop/Master/experiments/Lickwell_prediction/compare_wells.png"
    ax_list = []
    plot_metric_details_by_lickwell(model_path_list, 1,save_path,add_trial_numbers=True)
        # plot_metric_details_by_lickwell(path + "output/", -1,path+"images/compare_wells_" + image_title_list[i]+"_last.png",add_trial_numbers=True)

