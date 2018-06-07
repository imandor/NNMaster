from database_api_beta import Slice
from src.filters import bin_filter
# data_slice = Slice.from_path(save_as="slice.pkl")
from numpy.random import seed
import tdd_test

seed(1)
tdd_test.save_test_objects()
tdd_test.compare_test_objects()
# for i in range(0, 1):  # for neuron no 0 to 9:
    # plot_time_x_trials(all_trials_star>ting_with_well_1,neuron_no=i)
    # all_trials_starting_with_well_1.plot_positionx_x_trials(neuron_no=i)

print("fin")
