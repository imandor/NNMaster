from src.plots import plot_metric_details_by_lickwell

if __name__ == '__main__':



    # Data set 1 Prefrontal Cortex

    MODEL_PATH = "G:/master_datafiles/trained_networks/MLP_PFC_2019-01-18_lickwell_phasetarget/"
    RAW_DATA_PATH = "G:/master_datafiles/raw_data/PFC/"
    FILTERED_DATA_PATH = "session_pfc_lw_phasetarget.pkl"

    # Data set 2 Hippocampus

    # MODEL_PATH = "G:/master_datafiles/trained_networks/MLP_HC_2019-01-17_lickwell_phasetarget/"
    # RAW_DATA_PATH = "G:/master_datafiles/raw_data/HC/"
    # FILTERED_DATA_PATH = "session_hc_lw_phasetarget.pkl"

    model_path_list = ["C:/Users/NN/Desktop/Master/experiments/Lickwell_prediction/MLP_PFC_lickwell_phasetarget/",
                       "C:/Users/NN/Desktop/Master/experiments/Lickwell_prediction/MLP_HC_lickwell_phasetarget/",
                       "C:/Users/NN/Desktop/Master/experiments/Lickwell_prediction/MLP_PFC_lickwell/",
                       "C:/Users/NN/Desktop/Master/experiments/Lickwell_prediction/MLP_HC_lickwell/"]
    image_title_list = ["pfc_phase","hc_phase","pfc","hc"]
    for i,path in enumerate(model_path_list):
        plot_metric_details_by_lickwell(path + "output/", 1,path+"images/compare_wells_" + image_title_list[i]+"_next.png",add_trial_numbers=True)
        plot_metric_details_by_lickwell(path + "output/", -1,path+"images/compare_wells_" + image_title_list[i]+"_last.png",add_trial_numbers=True)

