import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from matplotlib.colors import BoundaryNorm
from src.settings import save_as_pickle, load_pickle
import glob
import numpy as np

# PATH = "G:/master_datafiles/trained_networks/MLP_hippocampus/"
# PATH = "G:/master_datafiles/trained_networks/MLP_OFC/"
# PATH = "G:/master_datafiles/trained_networks/MLP_OFC_2018-09-13/"
PATH = "G:/master_datafiles/trained_networks/MLP_hippocampus_2018-09-12/"

dict_files = glob.glob(PATH + "output/" + "*.pkl")
r2_scores_eval_list = []
r2_scores_train_list = []
acc_scores_eval_list = []
acc_scores_train_list = []
avg_scores_eval_list = []
avg_scores_train_list = []
time_shift_list = []
for file_path in sorted(dict_files ):
    print("processing",file_path)
    net_dict = load_pickle(file_path)
    r2_scores_train_list.append(net_dict["r2_scores_train"])
    r2_scores_eval_list.append(net_dict["r2_scores_eval"])
    acc_scores_train_list.append( net_dict["acc_scores_train"])
    acc_scores_eval_list.append(net_dict["acc_scores_eval"])
    avg_scores_train_list.append(net_dict["avg_scores_train"])
    avg_scores_eval_list.append(net_dict["avg_scores_eval"])
    time_shift_list.append(net_dict["TIME_SHIFT"])

training_step_list = [net_dict["METRIC_ITER"]]
for i in range(0,len(r2_scores_eval_list[0])-1):
    training_step_list.append(training_step_list[-1] + net_dict["METRIC_ITER"])

# time_shift_list = [str(x) for x in time_shift_list]

# Plot for all metric epochs


# for i in range(0,len(r2_scores_train_list)): # for each time_shift
#     r2_scores_eval = r2_scores_eval_list[i]
#     r2_scores_train = r2_scores_train_list[i]
#     acc_scores_eval = acc_scores_eval_list[i]
#     acc_scores_train = acc_scores_train_list[i]
#     avg_scores_eval = avg_scores_eval_list[i]
#     avg_scores_train = avg_scores_train_list[i]
#
#     acc_scores_eval = [[a[i] for a in acc_scores_eval] for i in range(len(acc_scores_eval[0]))]
#     acc_scores_train = [[a[i] for a in acc_scores_train] for i in range(len(acc_scores_train[0]))]
#
#     #     # Get data for current amount of training steps
#     #
#     levels = MaxNLocator(nbins=20).tick_values(np.min(acc_scores_train), np.max(acc_scores_train))
#     cmap = plt.get_cmap('gist_heat')
#     norm = BoundaryNorm(levels, ncolors=cmap.N, clip=True)
#     fig, (ax0, ax1) = plt.subplots(nrows=2)
#     cf = ax0.contourf(acc_scores_train, levels=levels, cmap=cmap)
#     fig.colorbar(cf, ax=ax0)
#     ax0.set_title('Portion of training predictions in radius wrt training_step')
#     ax0.set_xlabel("Training step")
#     ax0.set_ylabel("distance to actual position(cm)")
#     ax0.set_xticklabels(training_step_list)
#     # ax0.set_xticks(range(len(training_step_list)),training_step_list)
#     # levels = MaxNLocator(nbins=20).tick_values(np.min(acc_scores_eval), np.max(acc_scores_eval))
#
#     cf = ax1.contourf(acc_scores_eval, levels=levels, cmap=cmap)
#     fig.colorbar(cf, ax=ax1)
#     ax1.set_title('Portion of eval predictions in radius wrt training_step')
#     ax1.set_xlabel("Training step")
#     ax1.set_ylabel("distance to actual position(cm)")
#     # ax1.set_xticks(range(len(training_step_list)),training_step_list)
#     fig.tight_layout()
#     plt.ion()
#     # plt.show()
#     plt.savefig(PATH +"images/epoch_acc_score" + "_shift=" + time_shift_list[i] + ".pdf")
#
#     fig, (ax0, ax1) = plt.subplots(nrows=2,sharey=True)
#     ax0.plot(training_step_list,avg_scores_train)
#     ax0.set_title('Average distance of training wrt time-shift')
#     ax0.set_xlabel("Training step")
#     ax0.set_ylabel("Average distance to actual position(cm)")
#     ax1.plot(training_step_list,avg_scores_eval)
#     ax1.set_title('Average distance of evaluation wrt time-shift')
#     ax1.set_xlabel("Training step")
#     ax1.set_ylabel("Average distance to actual position(cm)")
#     fig.tight_layout()
#     # plt.show()
#     plt.savefig(PATH  +"images/epoch_avg_dist" + "_shift=" + time_shift_list[i] + ".pdf")
#
#
#     fig, (ax0, ax1) = plt.subplots(nrows=2,sharey=True)
#     ax0.plot(training_step_list,r2_scores_train)
#     ax0.set_title('r2 of training wrt training_steps')
#     ax0.set_xlabel("Training step")
#     ax0.set_ylabel("r2 score")
#     ax1.plot(training_step_list,r2_scores_eval)
#     ax1.set_title('r2 of evaluation wrt time-shift')
#     ax1.set_xlabel("Training step")
#     ax1.set_ylabel("r2 score")
#     fig.tight_layout()
#     # plt.show()
#     plt.savefig(PATH + "images/epoch_r2_score" + "_shift=" + time_shift_list[i] +  ".pdf")

for i in range(1,len(r2_scores_eval_list[0]) + 1): # for each evaluation in epoch range

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

    distance_list = np.linspace(0, 20, 20)
    levels = MaxNLocator(nbins=20).tick_values(np.min(acc_scores_train), np.max(acc_scores_train))
    cmap = plt.get_cmap('inferno')
    norm = BoundaryNorm(levels, ncolors=cmap.N, clip=True)
    fig, (ax0, ax1) = plt.subplots(nrows=2)
    cf = ax0.contourf(time_shift_list,distance_list,acc_scores_train, levels=levels, cmap=cmap)
    fig.colorbar(cf, ax=ax0)
    # ax0.grid(c='k', ls='-', alpha=0.3)
    ax0.set_title('Portion of training predictions in radius wrt time-shift')
    ax0.set_xlabel("Time shift (s)")
    ax0.set_ylabel("distance to actual position(cm)")
    # ax0.set_xticklabels(time_shift_list)

    # ax0.set_xticks(time_shift_list)
    # levels = MaxNLocator(nbins=20).tick_values(np.min(acc_scores_eval), np.max(acc_scores_eval))
    cf = ax1.contourf(time_shift_list,distance_list,acc_scores_eval, levels=levels, cmap=cmap)
    fig.colorbar(cf, ax=ax1)
    ax1.set_title('Portion of eval predictions in radius wrt time-shift')
    ax1.set_xlabel("Time shift (s)")
    ax1.set_ylabel("distance to actual position(cm)")
    # ax1.set_xticks(time_shift_list)
    fig.tight_layout()
    plt.ion()
    plt.savefig(PATH +"images/acc_score" + "_epoch=" + str(training_step_list[-i])  + ".pdf")






    fig, ax = plt.subplots()
    ax.plot(time_shift_list,distance_scores_train,label='Training set',color='r')
    ax.plot(time_shift_list,distance_scores_eval,label='Evaluation set',color='b')
    ax.legend()
    ax.grid(c='k', ls='-', alpha=0.3)
    ax.set_title('Average distance of evaluation wrt time-shift')
    ax.set_xlabel("Time shift (s)")
    ax.set_ylabel("Average distance to actual position(cm)")
    fig.tight_layout()
    plt.savefig(PATH  +"images/avg_dist" + "_epoch="+ str(training_step_list[-i]) + ".pdf")






    fig, (ax0, ax1) = plt.subplots(nrows=2,sharey=True)
    ax0.grid(c='k', ls='-', alpha=0.3)
    ax0.plot(time_shift_list,r2_scores_train)
    ax0.set_title('r2 of training wrt time-shift')
    ax0.set_xlabel("Time shift (s)")
    ax0.set_ylabel("r2 score")
    ax1.plot(time_shift_list,r2_scores_eval)
    ax1.grid(c='k', ls='-', alpha=0.3)
    ax1.set_title('r2 of evaluation wrt time-shift')
    ax1.set_xlabel("Time shift (s)")
    ax1.set_ylabel("r2 score")
    ax1.set_ylim([-10,1])
    fig.tight_layout()
    plt.savefig(PATH + "images/r2_score" + "_epoch=" + str(training_step_list[-i]) + ".pdf")
print("fin")

