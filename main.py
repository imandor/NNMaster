from database_api_beta import  Slice, plot_time_x_trials, get_nth_trial_in_each_phase
from filters import bin_filter
from session_loader import read_file
from tdd_test import save_sample_session, test_trial_sample
# data_slice = Slice.from_path(save_as="slice.pkl")
from numpy.random import seed

from networks import test_CNN
import matplotlib.pyplot as plt
from database_api_beta import filter_trials_by_well
seed(1)
# test_CNN()
data_slice = Slice.from_path(load_from="slice.pkl")
phases_2 = data_slice.get_all_phases()
medium_data_slice = data_slice
nth_trial_list = filter_trials_by_well(phases_2[2].get_all_trials(),start_well=1)

for i in range(0,166):
    plot_time_x_trials(nth_trial_list,neuron_no=i)

# ------------- test functions -----------------
# medium_data_slice = data_slice[0:200000]
# phases_1 = medium_data_slice.get_all_phases()
# phases_2 = data_slice.get_all_phases()
# smaller_data_slice = data_slice[0:500]
# smaller_data_slice.set_filter(filter=bin_filter, window=0)
# trial = phases_2[0].get_nth_trial(0)
# nth_trial_list = get_nth_trial_in_each_phase(phases_2,1)
# for i in range(0,len(smaller_data_slice.filtered_spikes[5])):
#     print(i,": ",smaller_data_slice.filtered_spikes[5][i])
# trial = data_slice.get_nth_trial(5)
#
# full_list_of_trials = smaller_data_slice.get_all_trials()
#
# list_of_trials = data_slice.get_trials(slice(0,300))
# sub_list_of_trials = full_list_of_trials[0:10]
# trial.bin_spikes(bin_size=100)
# ------------------- end test ------------------
print("fin")


