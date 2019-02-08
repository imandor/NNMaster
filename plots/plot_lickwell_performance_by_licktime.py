from src.database_api_beta import  Filter, hann, Net_data

from src.preprocessing import lickwells_io
from src.plots import plot_performance_by_licktime,plot_position_by_licktime
from src.network_functions import initiate_lickwell_network, run_lickwell_network
from src.metrics import print_metric_details
from random import seed
import numpy as np
if __name__ == '__main__':

# Removes a portion of the lick events and checks if performance changes

    model_path_list = [
        # "C:/Users/NN/Desktop/Master/experiments/Lickwell_prediction/MLP_PFC_lickwell_phasetarget/",
        #                "C:/Users/NN/Desktop/Master/experiments/Lickwell_prediction/MLP_HC_lickwell_phasetarget/",
                       "C:/Users/NN/Desktop/Master/experiments/Lickwell_prediction/MLP_PFC_timetest/",
                       # "C:/Users/NN/Desktop/Master/experiments/Lickwell_prediction/MLP_HC_timetest/"
    ]
    plotrange = range(0, 28)
    shift = 1
    if shift == 1:
        shiftpath = "next"
    else:
        shiftpath = "last"
    for model_path in model_path_list:
        plot_performance_by_licktime(path=model_path + "output/", shift=shift,save_path=model_path+"images/by_licktime_"+shiftpath+".png",
                                     add_trial_numbers=True,title="Fraction decoded correctly by time into lick-event",
                                     plotrange=plotrange)
