import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from matplotlib.colors import BoundaryNorm
from src.settings import save_as_pickle, load_pickle
import glob
import numpy as np


PATH = "G:/master_datafiles/trained_networks/MLP_OFC_2/"
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
    acc_scores_train_list.append( net_dict["acc_scores_train"])
    acc_scores_eval_list.append(net_dict["acc_scores_eval"])
    avg_scores_train_list.append(net_dict["avg_scores_train"])
    avg_scores_eval_list.append(net_dict["avg_scores_eval"])
    time_shift_list.append(net_dict["TIME_SHIFT"])

time_shift_list = [str(x) for x in time_shift_list]

# Plot for all metric epochs

trainied_steps = net_dict["trained_steps"]
for i in range(1,len(r2_scores_eval_list[0]) + 1):

    # Get data for current amount of training steps
    r2_scores_eval = [x[-i] for x in r2_scores_eval_list]
    r2_scores_train = [x[-i] for x in r2_scores_train_list]
    acc_scores_eval = list(map(list, zip(*[e[-i] for e in acc_scores_eval_list])))
    acc_scores_train = list(map(list, zip(*[e[-i] for e in acc_scores_train_list])))
    distance_scores_eval = [x[-1] for x in avg_scores_eval_list] # takes the latest trained value for each time shift
    distance_scores_train = [x[-1] for x in avg_scores_train_list]

    # acc_scores_eval_list =np.array(acc_scores_eval_list).T.tolist()
    # acc_scores_train_list = np.array(acc_scores_train_list).T.tolist()
    # distance_scores_eval_list = np.array(distance_scores_eval_list).T.tolist()
    # distance_scores_eval_list = np.array(distance_scores_eval_list).T.tolist()

    levels = MaxNLocator(nbins=15).tick_values(np.min(acc_scores_train), np.max(acc_scores_train))
    cmap = plt.get_cmap('PiYG')
    norm = BoundaryNorm(levels, ncolors=cmap.N, clip=True)
    fig, (ax0, ax1) = plt.subplots(nrows=2)
    cf = ax0.contourf(acc_scores_train, levels=levels, cmap=cmap)
    fig.colorbar(cf, ax=ax0)
    ax0.set_title('Portion of training predictions in radius wrt time-shift')
    ax0.set_xlabel("Time shift (s)")
    ax0.set_ylabel("distance to actual position(cm)")
    # ax0.set_xticks(time_shift_list)
    levels = MaxNLocator(nbins=15).tick_values(np.min(acc_scores_eval), np.max(acc_scores_eval))

    cf = ax1.contourf(acc_scores_eval, levels=levels, cmap=cmap)
    fig.colorbar(cf, ax=ax1)
    ax1.set_title('Portion of eval predictions in radius wrt time-shift')
    ax1.set_xlabel("Time shift (s)")
    ax1.set_ylabel("distance to actual position(cm)")
    # ax1.set_xticks(time_shift_list)
    fig.tight_layout()
    plt.ion()
    plt.show()



    fig, (ax0, ax1) = plt.subplots(nrows=2,sharey=True)
    ax0.plot(time_shift_list,distance_scores_train)
    ax0.set_title('Average distance of training wrt time-shift')
    ax0.set_xlabel("Time shift (s)")
    ax0.set_ylabel("Average distance to actual position(cm)")
    ax1.plot(time_shift_list,distance_scores_eval)
    ax1.set_title('Average distance of evaluation wrt time-shift')
    ax1.set_xlabel("Time shift (s)")
    ax1.set_ylabel("Average distance to actual position(cm)")
    fig.tight_layout()
    plt.show()


    fig, (ax0, ax1) = plt.subplots(nrows=2,sharey=True)
    ax0.plot(time_shift_list,r2_scores_train)
    ax0.set_title('r2 of training wrt time-shift')
    ax0.set_xlabel("Time shift (s)")
    ax0.set_ylabel("r2 score")
    ax1.plot(time_shift_list,r2_scores_eval)
    ax1.set_title('r2 of evaluation wrt time-shift')
    ax1.set_xlabel("Time shift (s)")
    ax1.set_ylabel("r2 score")
    fig.tight_layout()
    plt.show()
    trainied_steps = trainied_steps - net_dict[""]
print("fin")

