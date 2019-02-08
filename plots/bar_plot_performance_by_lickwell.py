from src.plots import plot_metric_details_by_lickwell

if __name__ == '__main__':
    model_path_list = ["C:/Users/NN/Desktop/Master/experiments/Lickwell_prediction/MLP_PFC_phasetarget/",
                       "C:/Users/NN/Desktop/Master/experiments/Lickwell_prediction/MLP_HC_phasetarget/",
                       "C:/Users/NN/Desktop/Master/experiments/Lickwell_prediction/MLP_PFC_lickwell/",
                       "C:/Users/NN/Desktop/Master/experiments/Lickwell_prediction/MLP_HC_lickwell/"]
    image_title_list = ["pfc_phase","hc_phase","pfc","hc"]
    for i,path in enumerate(model_path_list):
        plot_metric_details_by_lickwell(path + "output/", 1,path+"images/compare_wells_" + image_title_list[i]+"_next.png",add_trial_numbers=True)
        plot_metric_details_by_lickwell(path + "output/", -1,path+"images/compare_wells_" + image_title_list[i]+"_last.png",add_trial_numbers=True)

