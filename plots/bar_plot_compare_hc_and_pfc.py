from src.plots import plot_performance_comparison

if __name__ == '__main__':
    barcolor = "darkred"
    path = "C:/Users/NN/Desktop/Master/experiments/Lickwell_prediction/"
    model_path_list = [path + "MLP_PFC_phase/",
                       path + "MLP_HC_phase/",
                       path + "MLP_PFC/",
                       path + "MLP_HC/"]
    # phase data set

    # path_1 = model_path_list[0] + "output/"
    # path_2 =model_path_list[1] + "output/"
    # save_path = path + "compare_hc_pfc_phase_next.png"
    # plot_performance_comparison(path_1, 1, path_2, 1, "Prefrontal Cortex accuracy", "Hippocampus accuracy", save_path, barcolor,add_trial_numbers=True)
    # save_path = path + "compare_hc_pfc_phase_last.png"
    # plot_performance_comparison(path_1, -1, path_2, -1, "Prefrontal Cortex accuracy", "Hippocampus accuracy", save_path, barcolor,add_trial_numbers=True)

    # regular data set

    path_1 = model_path_list[2] + "output/"
    path_2 =model_path_list[3] + "output/"
    save_path = path + "compare_hc_pfc_next.png"
    plot_performance_comparison(path_1, 1, path_2, 1, "Prefrontal Cortex accuracy", "Hippocampus accuracy", save_path, barcolor,add_trial_numbers=True)
    save_path = path + "compare_hc_pfc_last.png"
    plot_performance_comparison(path_1, -1, path_2, -1, "Prefrontal Cortex accuracy", "Hippocampus accuracy", save_path, barcolor,add_trial_numbers=True)
