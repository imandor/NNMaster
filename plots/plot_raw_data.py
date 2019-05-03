from src.database_api_beta import  Net_data
import numpy as np
from src.metrics import print_metric_details
from src.network_functions import run_network_process, initiate_network, run_network
from src.model_data import c_dmf,chc_dmf,cpfc_dmf,hc_dmf,pfc_dmf
from matplotlib import rc
from matplotlib import pyplot as plt
import seaborn as sn
import pandas as pd
import matplotlib.colors as colors
from matplotlib.gridspec import GridSpec

well_to_color = {0: "#ff0000", 1: "#669900", 2: "#0066cc", 3: "#cc33ff", 4: "#003300", 5: "#996633"}


if __name__ == '__main__':
    from matplotlib import rc
    model_data = hc_dmf
    # model_data = pfc_dmf
    # model_data = chc_dmf

    model_data.model_path="G:/master_datafiles/trained_networks/speedtest/"
    nd = Net_data(
        initial_timeshift=10000,
        time_shift_iter=500,
        time_shift_steps=1,
        early_stopping=False,
        model_path=model_data.model_path,
        raw_data_path=model_data.raw_data_path,
        filtered_data_path=model_data.filtered_data_path,
        k_cross_validation = 10,
        valid_ratio=0.1,
        naive_test=False,
        from_raw_data=False,
        epochs = 30,
        dropout=0.65,
        # behavior_component_filter="rest",
        # behavior_component_filter="not at lickwell",
        # behavior_component_filter="correct trials",
        # behavior_component_filter="incorrect trials",
        # behavior_component_filter="move",

        filter_tetrodes=model_data.filter_tetrodes,
        shuffle_data=True,
        shuffle_factor=10,
        batch_size=50,
        switch_x_y=model_data.switch_x_y
    )
    starttime = 0
    endtime = 60000
    session = initiate_network(nd)
    slice = session[starttime:endtime]
    licks_timestamp = [lick.time for lick in session.licks]


    width = 0.75
    fontsize = 24
    rc('font', **{'family': 'serif', 'serif': ['Palatino']})
    rc('text', usetex=True)
    rc('xtick', labelsize=fontsize)
    rc('ytick', labelsize=fontsize)
    rc('axes', labelsize=fontsize)

    ind = np.arange(3)  # the x locations for the groups
    # fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2)
    fig = plt.figure(constrained_layout=True)
    gs = GridSpec(12, 12, figure=fig)
    ax1 = fig.add_subplot(gs[0:4, 0:6])
    ax2 = fig.add_subplot(gs[4:8, 0:6])
    ax3 = fig.add_subplot(gs[8:11, 0:6])
    ax4 = fig.add_subplot(gs[3:9, 6:11])

    # ax1 = plt.subplot(321)
    # ax2 = plt.subplot(323)
    # ax4 = plt.subplot(325)
    # ax3 = plt.subplot(122)

    # ax1.set_ylim(0, 1.0) # raw spikes
    # ax2.set_ylim(0, 1.0) # filtered spikes
    # ax3.set_ylim(0, 1.0) # wells
    # ax4.set_ylim(0, 1.0) # positions
    # ax5 # speed
    # ax6 licks

    error_kw = {'capsize': 5, 'capthick': 1, 'ecolor': 'black'}
    # ax1
    spikes = slice.spikes
    for i, spike in enumerate(reversed(spikes)):
        print("[{:4d}/{:4d}] Ploting raw spike data...".format(i + 1, slice.n_neurons), end="\r")
        for s in spike:
            ax1.vlines(s, i, i + 0.8)
    ax1.set_ylabel("neuron")
    # ax1.set_xlabel("time (ms)")
    ax1.set_xticklabels("")
    ax1.set_yticklabels("")

    # ax2
    n_bins = slice.filtered_spikes.shape[1]
    neuron_px_height = 20
    margin_px = 2
    normalize_each_neuron = False
    image = np.zeros((session.n_neurons * (neuron_px_height + margin_px) - margin_px, n_bins))
    for i, f in enumerate(slice.filtered_spikes):
        n = (f - np.min(f)) / (np.max(f) - np.min(f)) if normalize_each_neuron else f
        image[i * (neuron_px_height + margin_px): (i + 1) * (neuron_px_height + margin_px) - margin_px] = n
    ax2.imshow(image, cmap='hot', interpolation='none', aspect='auto')
    # ax2.set_xlabel("time (ms, binned)")
    ax2.set_xticklabels("")
    ax2.set_yticklabels("")
    ax2.set_ylabel("neuron")


    # ax 3

    min_time = 1e10
    max_time = -1
    position_resolution = 1000
    for d in slice.licks:
        well = d.lickwell
        time = d.time
        min_time = min_time if time > min_time else time
        max_time = max_time if time < max_time else time
        bbox = dict(boxstyle="circle", fc=well_to_color[well], ec="k")
        ax3.text(time, 0, str(well), bbox=bbox)
    y_list = []
    x_list = []
    for i, pos in enumerate(slice.position_x[0::position_resolution]):
        x_list.append(i*position_resolution+starttime)
        y_list.append(pos)
    ax3.plot(x_list,y_list)
    ax3.set_xlabel("time [ms]")
    ax3.set_ylabel("position [cm]")
    # ax4.hlines(0, min_time, max_time)

    # ax 4
    array = np.zeros((5, 5))
    licks = session.licks

    # metrics_a = load_pickle(model_path_list[0] + "output/metrics_k_timeshift=" + str(timeshift) + ".pkl")
    # metrics_b = load_pickle(model_path_list[1] + "output/metrics_k_timeshift=" + str(timeshift) + ".pkl")
    # metrics_af = np.reshape(metrics_a,-1)
    # metrics_bf = np.reshape(metrics_b,-1)

    for i, lick in enumerate(licks):
        if i != len(licks) - 1:
            next_lick = licks[i + 1]
        else:
            next_lick = None

        x = lick.lickwell - 1
        if next_lick is not None:
            y = next_lick.lickwell - 1
            array[x][y] += 1
    array = array.astype(int)
    df_cm = pd.DataFrame(array, index=[i for i in [1, 2, 3, 4, 5]],
                         columns=[i for i in [1, 2, 3, 4, 5]])
    # ax3.figure(figsize=(10, 7))
    ax4 = sn.heatmap(df_cm, annot=True, fmt="d", cmap="BuPu", annot_kws={"size": fontsize - 3},xticklabels=True,yticklabels=True)
    ax4.set_ylabel("starting well")
    ax4.set_xlabel("target well")
    plt.show()
    plt.close()









    #
    #
    #
    # filtered_spikes_kwargs = {}):
    # share_axis_set = set([ax_raw_spikes, ax_licks, ax_trial_timestamps])
    # share_axis_set.discard(None)
    # if len(share_axis_set) > 2:
    #     reference_ax = share_axis_set.pop()
    # for other_ax in share_axis_set:
    #     reference_ax.get_shared_x_axes().join(reference_ax, other_ax)
    # if ax_filtered_spikes is not None:
    #     self.plot_filtered_spikes(ax_filtered_spikes, **filtered_spikes_kwargs)
    # if ax_raw_spikes is not None:
    #     self.plot_raw_spikes(ax_raw_spikes)
    # if ax_licks is not None:
    #     self.plot_licks(ax_licks)
    # if ax_trial_timestamps is not None:
    #     self.plot_trial_timestamps(ax_trial_timestamps)