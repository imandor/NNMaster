import database_api_beta as api
import numpy as np
import matplotlib.pyplot as plt
import time
from src.filters import bin_filter

def f(x):
    return 1.0 if np.abs(x) < 350 else 0.0

data_slice = api.Slice.from_path(load_from="data/pickle/slice.pkl")
data_slice = data_slice[0:100000]
data_slice.set_filter(filter=bin_filter, search_window_size=700, step_size=700, num_threads=20)

fig = plt.figure()
ax_filtered_spikes = fig.add_subplot(1, 2, 2)
ax_raw_spikes = fig.add_subplot(2, 2, 1)
ax_licks = fig.add_subplot(4, 2, 5)
ax_trial_timestamps = fig.add_subplot(4, 2, 7)

data_slice.plot(ax_filtered_spikes=ax_filtered_spikes, ax_raw_spikes=ax_raw_spikes, ax_licks=ax_licks, ax_trial_timestamps=ax_trial_timestamps)

plt.show()
print("fin")