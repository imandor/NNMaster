from database_api import Slice,Session
from database_api_beta import  Slice
from session_loader import read_file
from tdd_test import save_sample_session, test_trial_sample
# slice = Slice.from_path(load_from="slice.pkl")
slice = Slice.from_path()
a = slice[2:3]

# trial_sample.plot_spikes(filtered=False)

# trials_sample = slice.get_all_trials_by_time(start=start, stop=stop)
print("fin")


