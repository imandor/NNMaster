import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from matplotlib.colors import BoundaryNorm
from src.settings import save_as_pickle, load_pickle
import glob
import numpy as np


PATH = "G:/master_datafiles/trained_networks/Special_CNN_hippocampus/"
dict_files = glob.glob(PATH + "*.pkl")
r2_scores_eval_list = []
r2_scores_train_list = []
acc_scores_eval_list = []
acc_scores_train_list = []
avg_scores_eval_list = []
avg_scores_train_list = []
time_shift_list = []
for file_path in sorted(dict_files):
    print("processing",file_path)
    net_dict = load_pickle(file_path)
    r2_scores_train_list.append(net_dict["r2_scores_train"])
    r2_scores_eval_list.append(net_dict["r2_scores_eval"])
    acc_scores_train_list.append( net_dict["acc_scores_train"]) # TODO remove [0]
    acc_scores_eval_list.append(net_dict["acc_scores_eval"])# TODO remove [0]
    avg_scores_train_list.append(net_dict["avg_scores_train"])
    avg_scores_eval_list.append(net_dict["avg_scores_eval"])
    time_shift_list.append(net_dict["TIME_SHIFT"])
plot_dict = dict(
r2_scores_eval_list = r2_scores_eval_list,
r2_scores_train_list = r2_scores_train_list,
acc_scores_eval_list = acc_scores_eval_list,
acc_scores_train_list = acc_scores_train_list,
avg_scores_eval_list = avg_scores_eval_list,
avg_scores_train_list = avg_scores_train_list,
time_shift_list = time_shift_list,
)
save_as_pickle("deleteme.pkl",plot_dict)
plot_dict = load_pickle("deleteme.pkl")
for i in range(1,len(plot_dict["r2_scores_eval_list"][0]) + 1):
    r2_scores_eval_list = [x[-i] for x in plot_dict["r2_scores_eval_list"]]
    r2_scores_train_list = [x[-i] for x in plot_dict["r2_scores_train_list"]]
    acc_scores_eval_list = plot_dict["acc_scores_eval_list"][-i]
    acc_scores_train_list = plot_dict["acc_scores_train_list"][-i]
    distance_scores_eval_list = [x[-1] for x in plot_dict["avg_scores_eval_list"]] # takes the latest trained value for each time shift
    distance_scores_train_list = [x[-1] for x in plot_dict["avg_scores_train_list"]]
    time_shift_list = [str(x) for x in plot_dict["time_shift_list"]]



    acc_scores_eval_list =np.array(acc_scores_eval_list).T.tolist()
    acc_scores_train_list = np.array(acc_scores_train_list).T.tolist()
    distance_scores_eval_list = np.array(distance_scores_eval_list).T.tolist()
    distance_scores_eval_list = np.array(distance_scores_eval_list).T.tolist()

    levels = MaxNLocator(nbins=15).tick_values(np.min(acc_scores_train_list), np.max(acc_scores_train_list))
    cmap = plt.get_cmap('PiYG')
    norm = BoundaryNorm(levels, ncolors=cmap.N, clip=True)
    fig, (ax0, ax1) = plt.subplots(nrows=2)
    cf = ax0.contourf(acc_scores_train_list, levels=levels, cmap=cmap)
    fig.colorbar(cf, ax=ax0)
    ax0.set_title('Portion of training predictions in radius wrt time-shift')
    ax0.set_xlabel("Time shift (s)")
    ax0.set_ylabel("distance to actual position(cm)")
    # ax0.set_xticks(time_shift_list)
    levels = MaxNLocator(nbins=15).tick_values(np.min(acc_scores_eval_list), np.max(acc_scores_eval_list))

    cf = ax1.contourf(acc_scores_eval_list, levels=levels, cmap=cmap)
    fig.colorbar(cf, ax=ax1)
    ax1.set_title('Portion of eval predictions in radius wrt time-shift')
    ax1.set_xlabel("Time shift (s)")
    ax1.set_ylabel("distance to actual position(cm)")
    # ax1.set_xticks(time_shift_list)
    fig.tight_layout()
    plt.show(block=True)
    plt.close()



    fig, (ax0, ax1) = plt.subplots(nrows=2,sharey=True)
    ax0.plot(time_shift_list,distance_scores_train_list)
    ax0.set_title('Average distance of training wrt time-shift')
    ax0.set_xlabel("Time shift (s)")
    ax0.set_ylabel("Average distance to actual position(cm)")
    ax1.plot(time_shift_list,distance_scores_eval_list)
    ax1.set_title('Average distance of evaluation wrt time-shift')
    ax1.set_xlabel("Time shift (s)")
    ax1.set_ylabel("Average distance to actual position(cm)")
    fig.tight_layout()
    plt.show(block=True)
    plt.close()


    fig, (ax0, ax1) = plt.subplots(nrows=2,sharey=True)
    ax0.plot(time_shift_list,r2_scores_train_list)
    ax0.set_title('r2 of training wrt time-shift')
    ax0.set_xlabel("Time shift (s)")
    ax0.set_ylabel("r2 score")
    ax1.plot(time_shift_list,r2_scores_eval_list)
    ax1.set_title('r2 of evaluation wrt time-shift')
    ax1.set_xlabel("Time shift (s)")
    ax1.set_ylabel("r2 score")
    fig.tight_layout()
    plt.show(block=True)
    plt.close()
print("fin")

