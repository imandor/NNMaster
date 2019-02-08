from src.plots import plot_accuracy_inside_phase, plot_performance_comparison

if __name__ == '__main__':


    model_path_list = ["C:/Users/NN/Desktop/Master/experiments/Lickwell_prediction/MLP_PFC_phasetarget/",
                       "C:/Users/NN/Desktop/Master/experiments/Lickwell_prediction/MLP_HC_phasetarget/",
                       "C:/Users/NN/Desktop/Master/experiments/Lickwell_prediction/MLP_PFC_lickwell/",
                       "C:/Users/NN/Desktop/Master/experiments/Lickwell_prediction/MLP_HC_lickwell/"

                       ]
    image_title_list = ["pfc_phase","hc_phase","pfc","hc"]
    for i,model_path in enumerate(model_path_list):
        path_1 = model_path+ "output/"
        path_2 = model_path + "output/"
        save_path = model_path + "images/compare_next_last_" + image_title_list[i]+".png"
        plot_performance_comparison(path_1, 1, path_2, -1, "Accuracy, when decoding next well","Accuracy, when decoding last well",save_path,"darkviolet",add_trial_numbers=True)