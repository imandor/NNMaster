from database_api import Slice,Session
from database_api_beta import  Slice
from filters import bin_filter
from session_loader import read_file
from tdd_test import save_sample_session, test_trial_sample
# data_slice = Slice.from_path(save_as="slice.pkl")
data_slice = Slice.from_path(load_from="slice.pkl")
medium_data_slice = data_slice[0:200000]
phases = medium_data_slice.get_all_phases()
smaller_data_slice = data_slice[0:500]
smaller_data_slice.set_filter(filter=bin_filter,window=0)
for i in range(0,len(smaller_data_slice.filtered_spikes[5])):
    print(i,": ",smaller_data_slice.filtered_spikes[5][i])
trial = data_slice.get_trial_by_id(5)

full_list_of_trials = smaller_data_slice.get_all_trials()

list_of_trials = data_slice.get_trials(slice(0,300))
sub_list_of_trials = full_list_of_trials[0:10]
trial.bin_spikes(bin_size=100)

# slice = Slice.from_path(save_as="slice.pkl")
a = data_slice[0:2]
s = slice(1,2)

# trials = d_slice.get_all_trials()


# trial_sample.plot_spikes(filtered=False)

# trials_sample = slice.get_all_trials_by_time(start=start, stop=stop)
print("fin")


