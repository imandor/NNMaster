import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from matplotlib.colors import BoundaryNorm
from src.settings import save_as_pickle, load_pickle
import glob
import numpy as np
from scipy.interpolate import interp1d
from matplotlib.patches import Patch
from statsmodels.nonparametric.smoothers_lowess import lowess
from matplotlib import rc
from src.network_functions import run_network_process, initiate_network, run_network
from src.database_api_beta import  Net_data
from src.model_data import c_dmf,chc_dmf,cpfc_dmf,hc_dmf,pfc_dmf

if __name__ == '__main__':

    model_data = hc_dmf
    nd = Net_data(
        initial_timeshift=0,
        time_shift_iter=500,
        time_shift_steps=1,
        early_stopping=True,
        model_path=model_data.model_path,
        raw_data_path=model_data.raw_data_path,
        filtered_data_path=model_data.filtered_data_path,
        k_cross_validation = 10,
        valid_ratio=0.1,
        naive_test=False,
        from_raw_data=False,
        epochs = 20,
        dropout=0.65,
        behavior_component_filter=None,
        filter_tetrodes=model_data.filter_tetrodes,
        shuffle_factor=50,
        switch_x_y=model_data.switch_x_y
    )
    session = initiate_network(nd)

    # Settings
    fontsize = 24
    plot_error_bars = False
    rc('font',**{'family':'serif','serif':['Palatino']})
    rc('text', usetex=True)
    rc('xtick', labelsize=fontsize)
    rc('ytick', labelsize=fontsize)
    rc('axes', labelsize=fontsize)
    avg_list = []
    std_list = []
    for i in range (0,len(session.position_x)-10000,10000):
        pos = []
        pos_0 = session.position_x[i]
        for j in range(0,10000,1000):
            pos.append(abs(session.position_x[i+j]-pos_0))
        std_list.append(pos)
        avg_list.append(pos)
    std_list = np.std(std_list,axis=0)
    avg_list = np.average(avg_list,axis=0)
    fig,ax = plt.subplots()


    ax.errorbar(x=range(0,10000,1000), y=avg_list, yerr=std_list, capsize=2,label="hippocampus",
                color="b", marker="None", linestyle=":")


    # ax.set_xlim(0,240)
    # ax.set_ylim(120,190)
    ax.tick_params(labelsize=24)
    ax.set_xlabel("time [ms]",fontsize=24)
    ax.set_ylabel("average movement [cm]",fontsize=24)


    model_data = c_dmf
    nd = Net_data(
        initial_timeshift=0,
        time_shift_iter=500,
        time_shift_steps=1,
        early_stopping=True,
        model_path=model_data.model_path,
        raw_data_path=model_data.raw_data_path,
        filtered_data_path=model_data.filtered_data_path,
        k_cross_validation = 10,
        valid_ratio=0.1,
        naive_test=False,
        from_raw_data=False,
        epochs = 20,
        dropout=0.65,
        behavior_component_filter=None,
        filter_tetrodes=model_data.filter_tetrodes,
        shuffle_factor=50,
        switch_x_y=model_data.switch_x_y
    )
    session = initiate_network(nd)
    avg_list = []
    std_list = []
    for i in range (0,len(session.position_x)-10000,10000):
        pos = []
        pos_0 = session.position_x[i]
        for j in range(0,10000,1000):
            pos.append(abs(session.position_x[i+j]-pos_0))
        std_list.append(pos)
        avg_list.append(pos)
    std_list = np.std(std_list,axis=0)
    avg_list = np.average(avg_list,axis=0)


    ax.errorbar(x=range(0,10000,1000), y=avg_list, yerr=std_list,color="purple" , capsize=2,label="combination",
                marker="None", linestyle=":")
    ax.legend(fontsize=fontsize)
    plt.show()
    plt.close()