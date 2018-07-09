from database_api_beta import Slice
from src.networks import test_full_session,test_trials, test_phase
from numpy.random import seed
from tdd_test import test_objects
from src.multi_processing import bin_slices_spikes
from src.networks_tf import run_network_test
from src.filters import bin_filter

seed(1)
bin_size = 700
step_size=700
units = 500
epochs = 10
data_slice = Slice.from_path(load_from="data/pickle/slice.pkl")  # load a data slice containing entire session
data_slice = data_slice[0:100000]
data_slice.test_convolve()
# dropout=0.1
# data_slice = Slice.from_path(load_from="data/pickle/slice.pkl")  # load a data slice containing entire session
# smaller_data_slice = data_slice[0:100000]
# smaller_data_slice.set_filter(filter=bin_filter, window=1,step_size=100)
data_slice = Slice.from_path(load_from="data/pickle/slice.pkl")  # load a data slice containing entire session
phases = data_slice.get_all_phases()
phases = phases[0:2]
run_network_test(data_slice,bin_size=bin_size,step_size=step_size)
# smaller_data_slice = data_slice[0:200]
# run_network_test(data_slice=data_slice)
# data_slice = Slice.from_path(load_from="data/pickle/slice.pkl")  # load a data slice containing entire session


# smaller_data_slice = data_slice[0:200000]  # slices first 200 seconds of session
#
# # smaller_data_slice.set_filter(filter=bin_filter, window=1,step_size=100)  # convolves data into bins
#
phases = data_slice.get_all_phases()  # gets all training phases as list of data slices
# trial = phases[0].get_nth_trial(0)  # gets first trial in first training phase
#
# list_of_trials = data_slice.get_trials(
#     slice(0, 200000))  # returns a list of trials in the first 200000 ms of session
# sub_list_of_trials = list_of_trials[0:10]  # slices first 10 entries in list of trials
# # for trial_i in sub_list_of_trials:
# #     trial_i.plot_filtered_spikes(filter=bin_filter, window=1, step_size=100)
#
# nth_trial_list = phases.get_nth_trial_in_each_phase(1)  # gets list of all second trials in list of phases
# all_trials_in_second_phase = phases[2].get_all_trials()  #
# all_trials_starting_with_well_1 = all_trials_in_second_phase.filter_trials_by_well(
#     start_well=1)
#
# for i in range(0, 166):  # for neuron no 0 to 9:
#     all_trials_starting_with_well_1.plot_time_x_trials(neuron_no=i)
#     all_trials_starting_with_well_1.plot_positionx_x_trials(neuron_no=i)

print("fin")
