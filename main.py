from database_api import Slice,Session
from database_api_beta import  Slice
from session_loader import read_file
from tdd_test import save_sample_session, test_trial_sample
d_slice = Slice.from_path(load_from="slice.pkl")
# slice = Slice.from_path(save_as="slice.pkl")
a = d_slice[0:2]
s = slice(1,2)
trials = d_slice.get_all_trials()
c = trials[0:2]

# trial_sample.plot_spikes(filtered=False)

# trials_sample = slice.get_all_trials_by_time(start=start, stop=stop)
print("fin")


