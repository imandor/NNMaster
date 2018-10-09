
from src.plots import load_trained_network


PATH = "G:/master_datafiles/trained_networks/MLP_hippocampus_2018-09-18/"


r2_scores_valid_list,r2_scores_train_list,acc_scores_valid_list,acc_scores_train_list,avg_scores_valid_list,avg_scores_train_list,net_dict,time_shift_list = load_trained_network(PATH)

