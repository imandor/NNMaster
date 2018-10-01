import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from matplotlib.colors import BoundaryNorm
from src.settings import save_as_pickle, load_pickle
import glob
import numpy as np
from scipy.interpolate import interp1d
from matplotlib.patches import Patch



def load_imagefile(path):
    dict_files = glob.glob(path + "output/" + "*.pkl")
    r2_scores_valid_list = []
    r2_scores_train_list = []
    acc_scores_valid_list = []
    acc_scores_train_list = []
    avg_scores_valid_list = []
    avg_scores_train_list = []
    time_shift_list = []
    sorted_list = []
    for i,file_path in enumerate(dict_files):
        net_dict = load_pickle(file_path)
        sorted_list.append([file_path,net_dict["TIME_SHIFT"]])
    sorted_list = sorted(sorted_list, key=lambda x: x[1])
    dict_files = [i[0] for i in sorted_list]
    for file_path in dict_files:
        print("processing", file_path)
        net_dict = load_pickle(file_path)
        r2_scores_train_list.append(net_dict["r2_scores_train"])
        r2_scores_valid_list.append(net_dict["r2_scores_valid"])
        acc_scores_train_list.append(net_dict["acc_scores_train"])
        acc_scores_valid_list.append(net_dict["acc_scores_valid"])
        avg_scores_train_list.append(net_dict["avg_scores_train"])
        avg_scores_valid_list.append(net_dict["avg_scores_valid"])
        time_shift_list.append(net_dict["TIME_SHIFT"])
    return r2_scores_valid_list,r2_scores_train_list,acc_scores_valid_list,acc_scores_train_list,avg_scores_valid_list,avg_scores_train_list,net_dict,time_shift_list

# PATH = "G:/master_datafiles/trained_networks/MLP_hippocampus/"
# PATH = "G:/master_datafiles/trained_networks/MLP_hippocampus_2018-09-20_ff/"
# PATH_2 = "G:/master_datafiles/trained_networks/MLP_hippocampus_2018-09-18/"
# PATH = "G:/master_datafiles/trained_networks/MLP_hippocampus_2018-09-12/"
PATH_2 = "G:/master_datafiles/trained_networks/MLP_OFC_2018-09-28_stride/"
PATH = "G:/master_datafiles/trained_networks/MLP_hippocampus_2018-09-26_stride/"


r2_scores_valid_list,r2_scores_train_list,acc_scores_valid_list,acc_scores_train_list,avg_scores_valid_list,avg_scores_train_list,net_dict,time_shift_list = load_imagefile(PATH)

training_step_list = [net_dict["METRIC_ITER"]]
for i in range(0,len(r2_scores_valid_list[0])-1):
    training_step_list.append(training_step_list[-1] + net_dict["METRIC_ITER"])



# ----------------------------------------------------


r2_scores_valid = [x[-1] for x in r2_scores_valid_list]
r2_scores_train = [x[-1] for x in r2_scores_train_list]
acc_scores_valid = list(map(list, zip(*[e[-1] for e in acc_scores_valid_list])))
acc_scores_train = list(map(list, zip(*[e[-1] for e in acc_scores_train_list])))
distance_scores_valid = [x[-1] for x in avg_scores_valid_list] # takes the latest trained value for each time shift
distance_scores_train = [x[-1] for x in avg_scores_train_list]


#
#     # Get data for current amount of training steps
#
#     r2_scores_valid = [x[-i] for x in r2_scores_valid_list]
#     r2_scores_train = [x[-i] for x in r2_scores_train_list]
#     acc_scores_valid = list(map(list, zip(*[e[-i] for e in acc_scores_valid_list])))
#     acc_scores_train = list(map(list, zip(*[e[-i] for e in acc_scores_train_list])))
#     distance_scores_valid = [x[-i] for x in avg_scores_valid_list] # takes the latest trained value for each time shift
#     distance_scores_train = [x[-i] for x in avg_scores_train_list]
#
#     # acc_scores_valid_list =np.array(acc_scores_valid_list).T.tolist()
#     # acc_scores_train_list = np.array(acc_scores_train_list).T.tolist()
#     # distance_scores_valid_list = np.array(distance_scores_valid_list).T.tolist()
#     # distance_scores_valid_list = np.array(distance_scores_valid_list).T.tolist()
#
distance_list = np.linspace(0, 20, 20)
levels = MaxNLocator(nbins=20).tick_values(np.min(acc_scores_train), np.max(acc_scores_train))
cmap = plt.get_cmap('inferno')
norm = BoundaryNorm(levels, ncolors=cmap.N, clip=True)
# fig, (ax0, ax1) = plt.subplots(nrows=2)
fig, (ax1) = plt.subplots()

# cf = ax0.contourf(time_shift_list,distance_list,acc_scores_train, levels=levels, cmap=cmap)
# fig.colorbar(cf, ax=ax0)
# # ax0.grid(c='k', ls='-', alpha=0.3)
# ax0.set_title('Portion of training predictions in radius wrt time-shift')
# ax0.set_xlabel("time shift [ms]")
# ax0.set_ylabel("distance to actual position(cm)")
# # ax0.set_xticklabels(time_shift_list)

# ax0.set_xticks(time_shift_list)
levels = MaxNLocator(nbins=20).tick_values(np.min(acc_scores_valid), np.max(acc_scores_valid))
cf = ax1.contourf(time_shift_list,distance_list,acc_scores_valid, levels=levels, cmap=cmap)
fig.colorbar(cf, ax=ax1)
# ax1.set_title('Portion of predictions inside given radius wrt time-shift')
ax1.set_xlabel("time shift [ms]")
ax1.set_ylabel("maximum position error [cm]")
# ax1.set_xticks(time_shift_list)
fig.tight_layout()
plt.ion()
plt.show()
plt.savefig(PATH +"images/acc_score" + "_epoch=" + str(training_step_list[-i])  + ".pdf")




fig, ax = plt.subplots()
# ax.plot(time_shift_list,distance_scores_train,label='Training set',color='r')
ax.plot(time_shift_list,distance_scores_valid,label='validation set',color='r')
ax.legend()
ax.grid(c='k', ls='-', alpha=0.3)
# ax.set_title(r'$\varnothing$distance of validation wrt time-shift')
ax.set_xlabel("Time shift [ms]")
ax.set_ylabel(r'$\varnothing$ distance to actual position in cm')
fig.tight_layout()
plt.savefig(PATH  +"images/avg_dist" + "_epoch="+ str(training_step_list[-i]) + ".pdf")




# fig, (ax0, ax1) = plt.subplots(nrows=2,sharey=True)
fig, ax1 = plt.subplots()
# ax0.grid(c='k', ls='-', alpha=0.3)
# ax0.plot(time_shift_list,r2_scores_train)
# ax0.set_title('r2 of training wrt time-shift')
# ax0.set_xlabel("Time shift (s)")
# ax0.set_ylabel("r2 score")
ax1.plot(time_shift_list,r2_scores_valid)
ax1.grid(c='k', ls='-', alpha=0.3)
# ax1.set_title('R2 of validation wrt time-shift')
ax1.set_xlabel("Time shift in ms")
ax1.set_ylabel("R2 score")
ax1.set_ylim([-1,0.6])
fig.tight_layout()
plt.savefig(PATH + "images/r2_score" + "_epoch=" + str(training_step_list[-i]) + ".pdf")




# ---------------------------------------------------------------

# asd asd asd asd



r2_scores_valid_list_2,r2_scores_train_list_2,acc_scores_valid_list_2,acc_scores_train_list_2,avg_scores_valid_list_2,avg_scores_train_list_2,net_dict_2,time_shift_list_2 = load_imagefile(PATH_2)
acc_scores_valid = list(map(list, zip(*[e[-1] for e in acc_scores_valid_list])))
acc_scores_valid_2 = list(map(list, zip(*[e[-1]for e in acc_scores_valid_list_2])))
r2_scores_valid = [x[-1][0] for x in r2_scores_valid_list]
r2_scores_valid_2 = [x[-1][0] for x in r2_scores_valid_list_2]
distance_scores_valid = [x[-1] for x in avg_scores_valid_list] # takes the latest trained value for each time shift
distance_scores_valid_2 = [x[-1] for x in avg_scores_valid_list_2] # takes the latest trained value for each time shift


acc_scores_middle = np.ndarray.tolist(np.array(acc_scores_valid) - np.array(acc_scores_valid_2))
r2_scores_middle = np.ndarray.tolist(np.array(r2_scores_valid) - np.array(r2_scores_valid_2))
distance_scores_middle = np.ndarray.tolist(np.array(distance_scores_valid) - np.array(distance_scores_valid_2))


# Compare accuracy plot

cmap = plt.get_cmap('PiYG')
fig, ax1 = plt.subplots()
distance_list = np.linspace(0, 20, 20)
levels = MaxNLocator(nbins=20).tick_values(np.min(acc_scores_middle), np.max(acc_scores_middle))
cf = ax1.contourf(time_shift_list, distance_list, acc_scores_middle, levels=levels, cmap=cmap)
fig.colorbar(cf, ax=ax1)
custom_lines = [Patch(facecolor='mediumvioletred', edgecolor='b',
                         label='acc_1'),
                Patch(facecolor='green', edgecolor='b',
                            label='acc_2')
                ]
ax1.legend(custom_lines, ['integral filter underperforms', 'integral filter overperforms'],bbox_to_anchor=(1, 1),
           bbox_transform=plt.gcf().transFigure)
# ax1.set_title('Portion of predictions inside given radius wrt time-shift')
ax1.set_xlabel("time shift [ms]")
ax1.set_ylabel(r'$\Delta$maximum position error [cm]')
# ax1.set_xticks(time_shift_list)
fig.tight_layout()
plt.ion()
plt.savefig(PATH + "images/acc_score_middle" + "_epoch=" + str(training_step_list[-i]) + ".pdf")



# Compare distances plot

fig, ax = plt.subplots()
# ax.plot(time_shift_list,distance_scores_train,label='Training set',color='r')
ax.plot(time_shift_list,distance_scores_middle,color='k')
ax.plot(time_shift_list,distance_scores_valid,color='darkgreen')
ax.plot(time_shift_list,distance_scores_valid_2,color='maroon')
# ax.legend()
custom_lines = [Patch(facecolor='green', edgecolor='b',
                         label='d_1'),
                Patch(facecolor='red', edgecolor='b',
                            label='d_2')
                ]
ax.legend(custom_lines, [r'$\varnothing$ distance integral filter', r'$\varnothing$distance hann filter'])
ax.grid(c='k', ls='-', alpha=0.3)
# ax.set_title(r'$\varnothing$distance of validation wrt time-shift')
ax.set_xlabel("time shift [ms]")
ax.set_ylabel(r'$\Delta$distance in cm')
f = interp1d(time_shift_list, distance_scores_middle)
x = np.linspace(time_shift_list[0],time_shift_list[-1],1000)
ax.fill_between(x, 0, f(x), where=(np.array(f(x))) < 0 , color='green')
ax.fill_between(x, 0, f(x), where=(np.array(f(x))) > 0 , color='red')
fig.tight_layout()
plt.savefig(PATH  +"images/avg_dist_middle" + "_epoch="+ str(training_step_list[-i]) + ".pdf")




# fig, (ax0, ax1) = plt.subplots(nrows=2,sharey=True)
fig, ax1 = plt.subplots()
# ax0.grid(c='k', ls='-', alpha=0.3)
# ax0.plot(time_shift_list,r2_scores_train)
# ax0.set_title('r2 of training wrt time-shift')
# ax0.set_xlabel("time shift [ms]")
# ax0.set_ylabel("r2 score")
custom_lines = [Patch(facecolor='green', edgecolor='b',
                         label='R2_1'),
                Patch(facecolor='red', edgecolor='b',
                            label='R2_2')
                ]
ax1.legend(custom_lines, ['R2 integral filter', 'R2 hann filter'])
f = interp1d(time_shift_list, r2_scores_middle)
x = np.linspace(time_shift_list[0],time_shift_list[-1],1000)
ax1.fill_between(x, 0, f(x), where=(np.array(f(x))) < 0 , color='maroon')
ax1.fill_between(x, 0, f(x), where=(np.array(f(x))) > 0 , color='darkgreen')
ax1.plot(time_shift_list,r2_scores_middle)
ax1.plot(time_shift_list,r2_scores_valid,color="g")
ax1.plot(time_shift_list,r2_scores_valid_2,color="r")
ax1.plot(time_shift_list,r2_scores_middle)
ax1.grid(c='k', ls='-', alpha=0.3)
# ax1.set_title('R2 of validation wrt time-shift')
ax1.set_xlabel("time shift [ms]")
ax1.set_ylabel(r'$\Delta$R2')
ax1.set_ylim([-1,0.6])
fig.tight_layout()
plt.savefig(PATH + "images/r2_score_middle" + "_epoch=" + str(training_step_list[-i]) + ".pdf")
print("fin")

