from src.database_api_beta import  Filter, hann, Net_data
from src.preprocessing import lickwells_io
from src.network_functions import initiate_lickwell_network, run_lickwell_network
from src.metrics import print_metric_details
from random import seed
from src.model_data import hc_lw,pfc_lw
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rc
from src.plots import get_corrected_std
from src.settings import load_pickle,save_as_pickle

def edit_axis_2(paths,ax_label,color,fontsize,ax=None,xlabel="",ylabel=""):
    ax = ax or plt.gca()
    x_list = np.arange(0.0,1.0,0.1)
    y_list = []
    for x in x_list:
        y_list.append(print_average_overlap(paths,x))
    ax.plot(x_list, y_list, label=ax_label, color=color, marker="None",linestyle=":") #,linestyle="None"
    ax.legend(fontsize=fontsize-3)
    ax.set_ylabel(ylabel)
    ax.set_xlabel(xlabel)
    ax.grid(c='k', ls='-', alpha=0.3)

def print_average_overlap(paths,filter_factor=0.0):
    """

    :param paths: list of paths of the two neural activity output files produced by this module
    :param filter_factor: keeps fraction of neurons with highest firing rate and filters out rest
    :return:
    """
    data_collection_i = []
    for path in paths:
        data_collection_i.append(load_pickle(path)[0:56])
    data_collection = (list(zip(data_collection_i[0],data_collection_i[1])))
    spikerate_collection = (data_collection_i[2] + data_collection_i[3])
    sorted_collection = spikerate_collection.copy()
    sorted_collection.sort()#
    max_spikerate = sorted_collection[int(filter_factor*len(spikerate_collection))]
    counter = []
    for i, data_row in enumerate(data_collection):
        if spikerate_collection[i]>max_spikerate:
            counter.append(data_row[0] == data_row[1])
    print(np.average(counter))
    return np.average(counter)


def plot_average_overlap(pathlist):
    fontsize=24
    rc('font',**{'family':'serif','serif':['Palatino']})
    rc('text', usetex=True)
    rc('xtick', labelsize=fontsize)
    rc('ytick', labelsize=fontsize)
    rc('axes', labelsize=fontsize)
    fig, ((ax1,ax2),(ax3,ax4)) = plt.subplots(nrows=2,ncols=2,sharey='all')


    edit_axis_2(paths=pathlist[0], ax_label="HC next", color="b", ax=ax1,fontsize=fontsize,ylabel="overlap")
    edit_axis_2(paths=pathlist[1], ax_label="HC last", color="darkblue", ax=ax2,fontsize=fontsize)
    edit_axis_2(paths=pathlist[2], ax_label="PFC next", color="r", ax=ax3,fontsize=fontsize,xlabel="fraction of neurons removed",ylabel="overlap")
    edit_axis_2(paths=pathlist[3], ax_label="PFC last", color="darkred", ax=ax4,fontsize=fontsize,xlabel="fraction of neurons removed")
    plt.show()
    pass

def print_table(paths):
    data_collection = []
    for path in paths:
        data_collection.append(load_pickle(path)[0:56])
    data_collection = np.transpose(data_collection)
    for i,datarow in enumerate(data_collection):
        print(i+1,end=" & ")
        for j,data in enumerate(datarow):
            if j in [0,1,4,5]:
                print(str(int(data)) + " & ",end="")
            else:
                print(format(float(data), ".1f") + " & ",end="")
        print(" \ ")

def filter_firing_rates_by_spikecount(filter_list,firing_rate_list,no_spikes):
    # return firing_rate_list # TODO works, but currently not needed
    return_list = []
    for i,well in enumerate(firing_rate_list):
        if well >=no_spikes:
            return_list.append(filter_list[i])
    return return_list

def filter_firing_rates_by_well(firing_rate_list, target_well_list, target_well):
    """

    :param firing_rate_list: list of average firing rates for each valid lick event
    :param target_well_list: list of target wells for each valid lick event
    :param target_well: firing_rate_list is filtered by this well
    :return: filtered firing_rate_list
    """
    return_list = []
    for i,well in enumerate(target_well_list):
        if well == target_well:
            return_list.append(firing_rate_list[i])
    return return_list



def edit_axis(y_values,xticklabels,time_range,color,label,ax=None,errorbar=None,plot_error_bars=True,y_label="",x_label=""):
    if plot_error_bars is False:
        errorbar=None
    ax = ax or plt.gca()
    ind = np.arange(5)  # the x locations for the groups
    error_kw = {'capsize': 5, 'capthick': 1, 'ecolor': 'black'}
    if time_range!=1:
        for i in range(len(y_values)):
            y_values[i] = y_values[i]/time_range
            errorbar[0][i] = errorbar[0][i]/time_range
            errorbar[1][i] = errorbar[1][i]/time_range

    ax.bar(xticklabels, y_values, color=color, yerr=errorbar, error_kw=error_kw, align='center',label=label)
    ax.set_xticks(ind)
    if label!="":
        ax.legend(fontsize=fontsize)
    ax.set_ylabel(y_label)
    ax.set_xlabel(x_label)
    for i, j in zip(ind, y_values):
        offset = 0.5
        if isinstance(y_values[0], int):
            ax.annotate(y_values[i], xy=(i - 0.1, j + offset), fontsize=fontsize - 3)
        else:
            ax.annotate(format(float(y_values[i]),".1f"), xy=(i - 0.1, j + offset), fontsize=fontsize - 3)
    return ax


if __name__ == '__main__':
    # paths = [
    #     "C:/Users/NN/Desktop/deleteme/current_well_hc.pkl",
    #     "C:/Users/NN/Desktop/deleteme/target_well_hc.pkl",
    #     "C:/Users/NN/Desktop/deleteme/current_firing_rate_hc.pkl",
    #     "C:/Users/NN/Desktop/deleteme/target_firing_rate_hc.pkl",
    #     "C:/Users/NN/Desktop/deleteme/current_well_pfc.pkl",
    #     "C:/Users/NN/Desktop/deleteme/target_well_pfc.pkl",
    #     "C:/Users/NN/Desktop/deleteme/current_firing_rate_pfc.pkl",
    #     "C:/Users/NN/Desktop/deleteme/target_firing_rate_pfc.pkl"
    # ]
    paths_hc_1 = [
        "C:/Users/NN/Desktop/spikerates/current_well_hc_1.pkl",
        "C:/Users/NN/Desktop/spikerates/target_well_hc_1.pkl",
        "C:/Users/NN/Desktop/spikerates/current_firing_rate_hc_1.pkl",
        "C:/Users/NN/Desktop/spikerates/target_firing_rate_hc_1.pkl"
    ]

    paths_hc_2 = [
        "C:/Users/NN/Desktop/spikerates/current_well_hc_-1.pkl",
        "C:/Users/NN/Desktop/spikerates/target_well_hc_-1.pkl",
        "C:/Users/NN/Desktop/spikerates/current_firing_rate_hc_-1.pkl",
        "C:/Users/NN/Desktop/spikerates/target_firing_rate_hc_-1.pkl"
    ]
    paths_pfc_1 = [
        "C:/Users/NN/Desktop/spikerates/current_well_pfc_1.pkl",
        "C:/Users/NN/Desktop/spikerates/target_well_pfc_1.pkl",
        "C:/Users/NN/Desktop/spikerates/current_firing_rate_pfc_1.pkl",
        "C:/Users/NN/Desktop/spikerates/target_firing_rate_pfc_1.pkl"
    ]

    paths_pfc_2 = [
        "C:/Users/NN/Desktop/spikerates/current_well_pfc_-1.pkl",
        "C:/Users/NN/Desktop/spikerates/target_well_pfc_-1.pkl",
        "C:/Users/NN/Desktop/spikerates/current_firing_rate_pfc_-1.pkl",
        "C:/Users/NN/Desktop/spikerates/target_firing_rate_pfc_-1.pkl"
    ]


    pathlist = [
        paths_hc_1,
        paths_hc_2,
        paths_pfc_1,
        paths_pfc_2
    ]

    print_table(paths_hc_1 + paths_pfc_1)
    # plot_average_overlap(pathlist)




    model_data =hc_lw # Warning: model_path has to be changed separately so the evaluated data can be plotted
    dataset = "hc"
    # preferred_well = "current" # which well is currently trained for
    # preferred_well = "target"
    preferred_well_list = ["target","current"]
    # finds preferred spike rate for each neuron
    for preferred_well in preferred_well_list:
        nd = Net_data(
            # Program execution settings
            epochs=20,
            evaluate_training=False,
            slice_size=100,
            stride=100,
            y_step=100,
            win_size=100,
            search_radius=100,
            k_cross_validation=10,
            session_filter=Filter(func=hann, search_radius=100, step_size=100),
            valid_ratio=0.1,
            testing_ratio=0,
            time_shift_steps=1,
            early_stopping=False,
            model_path=model_data.model_path,
            raw_data_path=model_data.raw_data_path,
            filtered_data_path=model_data.filtered_data_path,
            metric="discrete",
            shuffle_data=True,
            shuffle_factor=1,
            lw_classifications=4,
            lw_normalize=True,
            lw_differentiate_false_licks=False,
            num_wells=5,
            initial_timeshift=1,
            from_raw_data=False,
            dropout=0.65,
            number_of_bins=10,
        )
        # model_path = "C:/Users/NN/Desktop/Master/experiments/Experiments for thesis 2/well decoding/hc/"
        # nd = load_pickle(model_path + "output/nd_timeshift=" + str(nd_target.initial_timeshift)  + ".pkl")
        # licks = load_pickle(model_path + "output/metrics_timeshift=" + str(nd_target.initial_timeshift) +".pkl")

        lickstart=0
        lickstop = 5000
        lower_limit = 0 # upper and lower limit of y range shown on plot
        upper_limit = 15
        valid_spikecount = 0 # number of spikes the neuron must have to be plotted
        time_range = (lickstop-lickstart)/1000 # divide amount of spikes in range by this value to get spikes/s
        session = initiate_lickwell_network(nd)  # Initialize session
        X = []
        licks = []
        for i,lick in enumerate(session.licks):
            offset = 0
            lick_start = int(lick.time + lickstart+offset)
            shift = i + nd.initial_timeshift
            if shift>=0 and shift<len(session.licks):
                lick.target = session.licks[i+nd.initial_timeshift].lickwell
            lick_end = int(lick.time + lickstop+offset)
            X.append(session[lick_start+offset:lick_end+offset])
            licks.append(lick)
        # licks = []
        # for lick in loaded_licks:
        #     if lick.lick_id in nd.valid_licks:
        #         licks.append(lick)
        target_well_list = []
        firing_rate_list = []

        for i, lick in enumerate(licks):
            if preferred_well == "current" :
                target_well_list.append(lick.lickwell)
            if preferred_well == "target":
                target_well_list.append(lick.target)
            firing_rates = np.zeros(len(X[0].spikes))
            for j,neuron_spikes in enumerate(X[i].spikes):
                firing_rates[j] = len(neuron_spikes)
            firing_rate_list.append(firing_rates)
        spike_rate_1 = np.average(filter_firing_rates_by_well(firing_rate_list, target_well_list, 1), axis=0)
        spike_rate_2 = np.average(filter_firing_rates_by_well(firing_rate_list, target_well_list, 2), axis=0)
        spike_rate_3 = np.average(filter_firing_rates_by_well(firing_rate_list, target_well_list, 3), axis=0)
        spike_rate_4 = np.average(filter_firing_rates_by_well(firing_rate_list, target_well_list, 4), axis=0)
        spike_rate_5 = np.average(filter_firing_rates_by_well(firing_rate_list, target_well_list, 5), axis=0)
        spike_rates_by_well = np.transpose([spike_rate_1,spike_rate_2,spike_rate_3,spike_rate_4,spike_rate_5])
        preferred_well_by_neuron_list = []
        average_well_firing_rate_list = []
        preferred_well_firing_rate_list = []
        minimum_well_firing_rate_list = []

        for spikes in spike_rates_by_well:
            preferred_well_by_neuron_list.append(np.argmax(spikes)+1) # + 1 converts index to well_no
            preferred_well_firing_rate_list.append(np.max(spikes))
            average_well_firing_rate_list.append(np.average(spikes))
            minimum_well_firing_rate_list.append(np.min(spikes))

        if preferred_well == "current":
            save_as_pickle("C:/Users/NN/Desktop/spikerates/current_well_" + dataset+"_"+str(nd.initial_timeshift)+ ".pkl",preferred_well_by_neuron_list)
            save_as_pickle("C:/Users/NN/Desktop/spikerates/current_firing_rate_" + dataset+"_"+str(nd.initial_timeshift)+ ".pkl",preferred_well_firing_rate_list)
        if preferred_well == "target":
            save_as_pickle("C:/Users/NN/Desktop/spikerates/target_well_" + dataset+"_"+str(nd.initial_timeshift)+ ".pkl",preferred_well_by_neuron_list)
            save_as_pickle("C:/Users/NN/Desktop/spikerates/target_firing_rate_" + dataset+"_"+str(nd.initial_timeshift)+ ".pkl",preferred_well_firing_rate_list)

        # At this point we have for each neuron: preferred well, avg firing rate at preferred well, general average firing rate, minimum firing rate at any well
        # Now we average by well rather than by neuron
        preferred_firing_rate_1 = filter_firing_rates_by_well(preferred_well_firing_rate_list,preferred_well_by_neuron_list,1)
        preferred_firing_rate_2 = filter_firing_rates_by_well(preferred_well_firing_rate_list,preferred_well_by_neuron_list,2)
        preferred_firing_rate_3 = filter_firing_rates_by_well(preferred_well_firing_rate_list,preferred_well_by_neuron_list,3)
        preferred_firing_rate_4 = filter_firing_rates_by_well(preferred_well_firing_rate_list,preferred_well_by_neuron_list,4)
        preferred_firing_rate_5 = filter_firing_rates_by_well(preferred_well_firing_rate_list,preferred_well_by_neuron_list,5)


        preferred_firing_rate_well_1 = filter_firing_rates_by_spikecount(filter_firing_rates_by_well(preferred_well_firing_rate_list, preferred_well_by_neuron_list, 1),preferred_firing_rate_1,valid_spikecount)
        preferred_firing_rate_well_2 = filter_firing_rates_by_spikecount(filter_firing_rates_by_well(preferred_well_firing_rate_list, preferred_well_by_neuron_list, 2),preferred_firing_rate_2,valid_spikecount)
        preferred_firing_rate_well_3 = filter_firing_rates_by_spikecount(filter_firing_rates_by_well(preferred_well_firing_rate_list, preferred_well_by_neuron_list, 3),preferred_firing_rate_3,valid_spikecount)
        preferred_firing_rate_well_4 = filter_firing_rates_by_spikecount(filter_firing_rates_by_well(preferred_well_firing_rate_list, preferred_well_by_neuron_list, 4),preferred_firing_rate_4,valid_spikecount)
        preferred_firing_rate_well_5 = filter_firing_rates_by_spikecount(filter_firing_rates_by_well(preferred_well_firing_rate_list, preferred_well_by_neuron_list, 5),preferred_firing_rate_5,valid_spikecount)
        std_preferred_well = [np.std(preferred_firing_rate_well_1),np.std(preferred_firing_rate_well_2),np.std(preferred_firing_rate_well_3),np.std(preferred_firing_rate_well_4),np.std(preferred_firing_rate_well_5)]
        preferred_firing_rate = [np.average(preferred_firing_rate_well_1),np.average(preferred_firing_rate_well_2),np.average(preferred_firing_rate_well_3),
                np.average(preferred_firing_rate_well_4),np.average(preferred_firing_rate_well_5)]
        std_preferred_well_lower,std_preferred_well_upper = get_corrected_std(preferred_firing_rate,std_preferred_well,lower_limit=lower_limit*time_range,upper_limit=upper_limit*time_range)

        average_firing_rate_well_1 = filter_firing_rates_by_spikecount(filter_firing_rates_by_well(average_well_firing_rate_list, preferred_well_by_neuron_list, 1),preferred_firing_rate_1,valid_spikecount)
        average_firing_rate_well_2 = filter_firing_rates_by_spikecount(filter_firing_rates_by_well(average_well_firing_rate_list, preferred_well_by_neuron_list, 2),preferred_firing_rate_2,valid_spikecount)
        average_firing_rate_well_3 = filter_firing_rates_by_spikecount(filter_firing_rates_by_well(average_well_firing_rate_list, preferred_well_by_neuron_list, 3),preferred_firing_rate_3,valid_spikecount)
        average_firing_rate_well_4 = filter_firing_rates_by_spikecount(filter_firing_rates_by_well(average_well_firing_rate_list, preferred_well_by_neuron_list, 4),preferred_firing_rate_4,valid_spikecount)
        average_firing_rate_well_5 = filter_firing_rates_by_spikecount(filter_firing_rates_by_well(average_well_firing_rate_list, preferred_well_by_neuron_list, 5),preferred_firing_rate_5,valid_spikecount)
        std_average_well = [np.std(average_firing_rate_well_1),np.std(average_firing_rate_well_2), np.std(average_firing_rate_well_3),
                            np.std(average_firing_rate_well_4), np.std(average_firing_rate_well_5)]
        average_firing_rate = [np.average(average_firing_rate_well_1),np.average(average_firing_rate_well_2), np.average(average_firing_rate_well_3),
                               np.average(average_firing_rate_well_4), np.average(average_firing_rate_well_5)]
        std_average_well_lower, std_average_well_upper = get_corrected_std(average_firing_rate, std_average_well,lower_limit=lower_limit*time_range,upper_limit=upper_limit*time_range)

        minimum_firing_rate_well_1 = filter_firing_rates_by_spikecount(filter_firing_rates_by_well(minimum_well_firing_rate_list, preferred_well_by_neuron_list, 1),preferred_firing_rate_1,valid_spikecount)
        minimum_firing_rate_well_2 = filter_firing_rates_by_spikecount(filter_firing_rates_by_well(minimum_well_firing_rate_list, preferred_well_by_neuron_list, 2),preferred_firing_rate_2,valid_spikecount)
        minimum_firing_rate_well_3 = filter_firing_rates_by_spikecount(filter_firing_rates_by_well(minimum_well_firing_rate_list, preferred_well_by_neuron_list, 3),preferred_firing_rate_3,valid_spikecount)
        minimum_firing_rate_well_4 = filter_firing_rates_by_spikecount(filter_firing_rates_by_well(minimum_well_firing_rate_list, preferred_well_by_neuron_list, 4),preferred_firing_rate_4,valid_spikecount)
        minimum_firing_rate_well_5 = filter_firing_rates_by_spikecount(filter_firing_rates_by_well(minimum_well_firing_rate_list, preferred_well_by_neuron_list, 5),preferred_firing_rate_5,valid_spikecount)
        std_minimum_well = [np.std(minimum_firing_rate_well_1),np.std(minimum_firing_rate_well_2), np.std(minimum_firing_rate_well_3),
                            np.std(minimum_firing_rate_well_4), np.std(minimum_firing_rate_well_5)]
        minimum_firing_rate = [np.average(minimum_firing_rate_well_1),np.average(minimum_firing_rate_well_2), np.average(minimum_firing_rate_well_3),
                               np.average(minimum_firing_rate_well_4), np.average(minimum_firing_rate_well_5)]
        std_minimum_well_lower, std_minimum_well_upper = get_corrected_std(minimum_firing_rate, std_minimum_well,lower_limit=lower_limit*time_range,upper_limit=upper_limit*time_range)
        neurons_by_well = [len(preferred_firing_rate_well_1),len(preferred_firing_rate_well_2),len(preferred_firing_rate_well_3),len(preferred_firing_rate_well_4),len(preferred_firing_rate_well_5)]


        # Plot results by well
        fontsize=24
        rc('font',**{'family':'serif','serif':['Palatino']})
        rc('text', usetex=True)
        rc('xtick', labelsize=fontsize)
        rc('ytick', labelsize=fontsize)
        rc('axes', labelsize=fontsize)
        xticklabels = ["1","2","3","4","5"]
        fig, ((ax1,ax2),(ax3,ax4)) = plt.subplots(2,2)
        color_list = ["b","navy","dodgerblue","royalblue"]
        # color_list = ["r","darkred","firebrick","maroon"]

        edit_axis(neurons_by_well, xticklabels, time_range=1,color=color_list[0], label="HC", ax=ax1, errorbar=None, plot_error_bars=False, y_label="neurons", x_label="")
        edit_axis(preferred_firing_rate, xticklabels, time_range,color=color_list[1], label="", ax=ax2, errorbar=[std_preferred_well_lower,std_preferred_well_upper], plot_error_bars=True, y_label="spikes/s", x_label="")
        edit_axis(minimum_firing_rate, xticklabels,time_range, color=color_list[3], label="", ax=ax3, errorbar=[std_minimum_well_lower,std_minimum_well_upper], plot_error_bars=True, y_label="min spikes/s", x_label="preferred well")
        edit_axis(average_firing_rate, xticklabels,time_range, color=color_list[2], label="", ax=ax4, errorbar=[std_average_well_lower,std_average_well_upper], plot_error_bars=True, y_label="avg spikes/s", x_label="preferred well")

        ax1.set_ylim(0,55)
        ax2.set_ylim(lower_limit,upper_limit)
        ax3.set_ylim(lower_limit,upper_limit)
        ax4.set_ylim(lower_limit,upper_limit)

        plt.show()
        pass
