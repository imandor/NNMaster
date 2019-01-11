import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from matplotlib.colors import BoundaryNorm
from src.settings import save_as_pickle, load_pickle
import glob
import numpy as np
from scipy.interpolate import interp1d
from matplotlib.patches import Patch
from statsmodels.nonparametric.smoothers_lowess import lowess


def metric_plot_a(save_path, x, y,y_all, axis_label_x, axis_label_y):
    fig, ax = plt.subplots()
    # plt.rc('font', family='serif', serif='Times')
    plt.rc('text', usetex=True)
    plt.rc('xtick', labelsize=8)
    plt.rc('ytick', labelsize=8)
    plt.rc('axes', labelsize=8)
    width = 3.5
    height = width / 1.5
    fig.set_size_inches(width, height)
    # ys = lowess(y, x)[:, 1]
    for i in range(len(y_all[0])):
        y_i = [a[i] for a in y_all]
        if i == 0:
            c = "b"
        if i == 1:
            c = "g"
        if i == 2:
            c = "r"
        if i == 3:
            c = "c"
        if i == 4:
            c = "m"
        if i == 5:
            c = "y"
        if i == 6:
            c = "k"
        if i == 7:
            c = "g"
        if i == 8:
            c = "sandybrown"
        if i == 9:
            c = "goldenrod"
        ax.plot(x,y_i,color=c)#label="cv "+str(i+1)+"/10",
    ax.plot(x, y, label='average', color='r', marker="X") #,linestyle="None"

    ax.legend()
    ax.grid(c='k', ls='-', alpha=0.3)
    ax.set_xlabel(axis_label_x)
    ax.set_ylabel(axis_label_y)
    # fig.tight_layout()
    plt.show()
    # plt.savefig(save_path)
    plt.close()

class Metrics_Wrt_Time:  # object containing list of metrics by cross validation partition
    def __init__(self, path):
        self.path = path
        self.network_output_list = self.load_network() # metric by cross validation step
        self.timeshift_list = None

    def load_network(self):
        dict_files = glob.glob(self.path + "output/" + "*.pkl")
        net_out = []
        sorted_list = []
        if len(dict_files) == 0:
            raise OSError("Warning: network Directory is empty")
        for i, file_path in enumerate(dict_files):
            net_dict_i = load_pickle(file_path)
            sorted_list.append([file_path, net_dict_i.net_data.time_shift])
        sorted_list = sorted(sorted_list, key=lambda x: x[1])
        dict_files = [i[0] for i in sorted_list]
        for file_path in dict_files:
            # print("processing", file_path)
            net_dict_i = load_pickle(file_path)
            net_out.append(net_dict_i)
        return net_out


    def plot_r2_wrt_ts(self,epoch=-1):
        ape_avg_list = [a.r2_avg for a in self.network_output_list]
        all_samples = [a.metric_by_cvs for a in self.network_output_list]
        y_all = [[a.r2_by_epoch[epoch] for a in metric] for metric in all_samples]
        save_path = self.path + "images/ape.pdf"
        axis_label_x = 'Time shift [ms]'
        axis_label_y = r'$\varnothing$ absolute position error [cm]'
        metric_plot_a(save_path, self.timeshift_list, y=ape_avg_list, y_all=y_all,axis_label_x=axis_label_x, axis_label_y=axis_label_y)
        pass

    def plot_acc20_wrt_ts(self):
        pass


    def plot_ape_wrt_ts(self,epoch=-1):
        """ creates plots for absolute position error with respect to timeshift
        :param epoch: epoch to be plotted
        :return:
        """
        ape_avg_list = [np.average([b.ape_by_epoch[-1] for b in a.metric_by_cvs]) for a in self.network_output_list]
        all_samples = [a.metric_by_cvs for a in self.network_output_list]
        y_all = [[a.ape_by_epoch[epoch] for a in metric] for metric in all_samples]
        save_path = self.path + "images/ape.pdf"
        axis_label_x = 'Time shift [ms]'
        axis_label_y = r'$\varnothing$ absolute position error [cm]'
        metric_plot_a(save_path, self.timeshift_list, y=ape_avg_list, y_all=y_all,axis_label_x=axis_label_x, axis_label_y=axis_label_y)

    def plot_paired_t_test(self):

        # Preprocess absolute position error

        ape_avg_list = [a.ape_avg for a in self.network_output_list]
        test_samples = (len(ape_avg_list) - 1) // 2
        if self.network_output_list[test_samples].net_data.time_shift!= 0:
            raise ValueError("Warning: paired t-test datasets are not centered around zero")

        positive_range = range(test_samples , len(ape_avg_list))
        negative_range = range(test_samples , -1, -1)
        timeshift_list_t_test = self.timeshift_list[test_samples:]
        positive_ape_avg_list = [ape_avg_list[i] for i in positive_range]
        negative_ape_avg_list = [ape_avg_list[i] for i in negative_range]
        t_score_list = np.ndarray.tolist(np.array(positive_ape_avg_list) - np.array(negative_ape_avg_list))


        # Plot

        fig, ax = plt.subplots()
        # plt.rc('font', family='serif', serif='Times')
        # plt.rc('text', usetex=True)
        plt.rc('xtick', labelsize=8)
        plt.rc('ytick', labelsize=8)
        plt.rc('axes', labelsize=8)
        width = 3.5
        height = width / 1.5
        fig.set_size_inches(width, height)
        ax.plot(timeshift_list_t_test, t_score_list, color='k')
        ax.plot(timeshift_list_t_test, positive_ape_avg_list, color='darkblue')
        ax.plot(timeshift_list_t_test, negative_ape_avg_list, color='maroon')
        ax.axhline(y=0)
        # ax.legend()
        custom_lines = [Patch(facecolor='blue', edgecolor='b',
                              label='d_1'),
                        Patch(facecolor='red', edgecolor='b',
                              label='d_2')
                        ]
        ax.legend(custom_lines, ['positive time shifts', 'negative time shifts'])
        ax.grid(c='k', ls='-', alpha=0.3)
        # ax.set_title(r'$\varnothing$ distance of validation wrt time-shift')
        ax.set_xlabel("absolute time shift [ms]")
        ax.set_ylabel('absolute position error [cm]')
        f = interp1d(timeshift_list_t_test, t_score_list)
        x = np.linspace(timeshift_list_t_test[0], timeshift_list_t_test[-1], 1000)
        ax.fill_between(x, 0, f(x), where=(np.array(f(x))) < 0, color='blue')
        ax.fill_between(x, 0, f(x), where=(np.array(f(x))) > 0, color='red')
        plt.show()
        plt.savefig(self.path + "images/t_test.pdf")


    def set_timeshift_list(self):
        self.timeshift_list = [a.net_data.time_shift for a in self.network_output_list]

path = "C:/Users/NN/Desktop/Master/experiments/decode memory future/MLP_HC_2018-11-13_dmf/"
# path = "C:/Users/NN/Desktop/Master/experiments/decode memory future/DMF_CHC_2019-01-04_dmf/"
# path = "C:/Users/NN/Desktop/Master/experiments/decode memory future/DMF_CPFC_2019-01-04_dmf/"
# path = "C:/Users/NN/Desktop/Master/experiments/decode memory future/DMF_C_2019-01-03_dmf/"
# path = "C:/Users/NN/Desktop/Master/experiments/decode memory future/DMF_PFC_2018-12-13_dmf/"
net_out = Metrics_Wrt_Time(path)
net_out.set_timeshift_list()

net_out.plot_paired_t_test()
# net_out.plot_r2_wrt_ts()
# net_out.plot_ape_wrt_ts()
print("fin")
