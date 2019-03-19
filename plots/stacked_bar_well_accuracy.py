import matplotlib
from matplotlib import pyplot as plt

from src.database_api_beta import  Filter, hann, Net_data

from src.preprocessing import lickwells_io
from src.plots import get_accuracy_for_comparison_2, get_corrected_std
from src.metrics import print_metric_details, get_metric_details
from random import seed
import numpy as np
from matplotlib import rc

if __name__ == '__main__':

# Plots a stacked bar representation of well accuracy. Plot is not automated and data has to be entered manually

    barcolor = "darkviolet"

    width = 0.75
    fontsize = 24

    rc('font', **{'family': 'serif', 'serif': ['Palatino']})
    rc('text', usetex=True)
    rc('xtick', labelsize=fontsize)
    rc('ytick', labelsize=fontsize)
    rc('axes', labelsize=fontsize)

    ind = np.arange(4)  # the x locations for the groups
    fig, ax = plt.subplots()
    ax.set_ylim(0, 1.0)
    # Data
    first_guess = [0.68,0.64,0.58,0.59]
    second_guess = [0.22,0.25,0.26,0.25]
    ax.bar(ind, first_guess, color="purple",  align='center',label="most frequent guess",edgecolor="black",linewidth=1)
    ax.bar(ind, second_guess, color="indigo",  align='center',label="second most frequent guess",edgecolor="black",bottom=first_guess,linewidth=1)
    ax.set_ylabel("fraction of guesses", fontsize=fontsize)
    ax.set_xticks(ind)
    ax.set_xticklabels(['hc next', 'hc last', 'pfc next','pfc last'], fontsize=fontsize)
    # ax.set_title("decoding next well",fontsize=fontsize)

    ax.grid(True,axis="y")
    ax.legend(fontsize=fontsize)
    plt.show()
    pass