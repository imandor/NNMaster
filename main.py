from database_api_beta import Slice, plot_positionx_x_trials, get_nth_trial_in_each_phase
from src.filters import bin_filter
# data_slice = Slice.from_path(save_as="slice.pkl")
from numpy.random import seed

from database_api_beta import filter_trials_by_well

seed(1)
# test_CNN()
# data_slice = Slice.from_path(save_as="slice.pkl")

data_slice = Slice.from_path(load_from="data/pickle/slice.pkl")  # load a data slice containing entire session

smaller_data_slice = data_slice[0:200000] # slices first 200 seconds of session

smaller_data_slice.set_filter(filter=bin_filter,window=1) # convolves data into bins
phases = data_slice.get_all_phases()  # gets all training phases as list of data slices
trial = phases[0].get_nth_trial(0) # gets first trial in first training phase
list_of_trials = data_slice.get_trials(slice(0,200000)) # returns a list of trials in the first 200000 ms of session
sub_list_of_trials = list_of_trials[0:10] # slices first 10 entries in list of trials

nth_trial_list = phases.get_nth_trial_in_each_phase(1) # gets list of all second trials in list of phases
all_trials_in_second_phase = phases[2].get_all_trials()  # returns a list of all trials in second training phase

all_trials_starting_with_well_1 = \
    filter_trials_by_well(all_trials_in_second_phase, start_well=1)  # returns all trials in second phase that start with a lick at well 1

for i in range(0, 30): # for neuron no 0 to 9:
    # plot_time_x_trials(all_trials_star>ting_with_well_1,neuron_no=i)
    all_trials_starting_with_well_1.plot_positionx_x_trials( neuron_no=i)
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
