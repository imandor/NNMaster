import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from matplotlib.colors import BoundaryNorm
from src.settings import save_as_pickle, load_pickle
import glob
import numpy as np
from scipy.interpolate import interp1d
from matplotlib.patches import Patch
from statsmodels.nonparametric.smoothers_lowess import lowess
from src.network_functions import run_network_process, initiate_network, run_network
from src.database_api_beta import  Net_data
import matplotlib.lines as mlines


def newline(p1, p2):
    ax = plt.gca()
    xmin, xmax = ax.get_xbound()

    if(p2[0] == p1[0]):
        xmin = xmax = p1[0]
        ymin, ymax = ax.get_ybound()
    else:
        ymax = p1[1]+(p2[1]-p1[1])/(p2[0]-p1[0])*(xmax-p1[0])
        ymin = p1[1]+(p2[1]-p1[1])/(p2[0]-p1[0])*(xmin-p1[0])

    l = mlines.Line2D([xmin,xmax], [ymin,ymax])
    ax.add_line(l)
    return l


if __name__ == '__main__':
    combination_data_set  = False
    filter_tetrodes = None
    # prefrontal cortex
    # MODEL_PATH = "G:/master_datafiles/trained_networks/pfc_bc_correct_trials/"
    # RAW_DATA_PATH = "G:/master_datafiles/raw_data/PFC/"
    # FILTERED_DATA_PATH = "session_pfc"

    # hippocampus
    #
    MODEL_PATH = "G:/master_datafiles/trained_networks/hc_bc_correct_trials/"
    RAW_DATA_PATH = "G:/master_datafiles/raw_data/HC/"
    FILTERED_DATA_PATH = "session_hc"

    # Combination data set

    # RAW_DATA_PATH = "G:/master_datafiles/raw_data/C"
    # combination_data_set = True # for some reason, the combination data set has switched x and y axis, which needs to be manually switched back
    # only Hippocampus neurons

    # MODEL_PATH = "G:/master_datafiles/trained_networks/chc_bc_test/"
    # FILTERED_DATA_PATH = "session_CHC.pkl"
    # filter_tetrodes=range(13,1000)

    # only Prefrontal Cortex neurons

    # MODEL_PATH = "G:/master_datafiles/trained_networks/cpfc_bc_correct_trials/"
    # FILTERED_DATA_PATH = "session_CPFC.pkl"
    # filter_tetrodes=range(0,13)

    # all neurons

    # MODEL_PATH = "G:/master_datafiles/trained_networks/c_bc_false_trials/"
    # FILTERED_DATA_PATH = "slice_C.pkl"


    nd = Net_data(
        initial_timeshift=0,
        time_shift_iter=500,
        time_shift_steps=1,
        early_stopping=True,
        model_path=MODEL_PATH,
        raw_data_path=RAW_DATA_PATH,
        filtered_data_path=FILTERED_DATA_PATH,
        k_cross_validation = 10,
        valid_ratio=0.1,
        naive_test=False,
        from_raw_data=False,
        epochs = 20,
        dropout=0.65,
        behavior_component_filter=None,
        filter_tetrodes=filter_tetrodes,
        shuffle_factor=50
    )
    session = initiate_network(nd)
    framerate = 250
    p1 = [int(session.position_x[0]),int(session.position_y[0])]
    i = 0
    for posx,posy in zip(session.position_x[1:-1],session.position_y[1:-1]):
        i = i + 1
        if i % framerate == 0:
            p2 = [posx, posy]
            if p1 != p2:
                plt.plot([p1[0],p2[0]], [p1[1],p2[1]],linewidth=2,color="b")
                p1 = p2
    plt.xlim(0,240)
    plt.ylim(100,190)
    plt.xlabel("X axis",fontsize=18)
    plt.ylabel("Y axis",fontsize=18)
    plt.show()
    pass