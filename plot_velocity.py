import database_api_beta as api
import numpy as np
import matplotlib.pyplot as plt
import time
from src.multi_processing import bin_slices_spikes

def f(x):
    return 1.0 if np.abs(x) < 350 else 0.0


fig = plt.figure()
ax_position_x = fig.add_subplot(1, 2, 2)
ax_speed = fig.add_subplot(2, 2, 1)
ax_licks = fig.add_subplot(4, 2, 5)
ax_trial_timestamps = fig.add_subplot(4, 2, 7)

data_slice = api.Slice.from_path(load_from="data/pickle/slice.pkl")
data_slice = data_slice[0:100000]
data_slice = bin_slices_spikes(data_slice, search_window_size=700, step_size=700, num_threads=20)
data_slice.plot_velocity(ax_position_x=ax_position_x, ax_speed=ax_speed, ax_licks=ax_licks, ax_trial_timestamps=None,
             ax_speed_kwargs={})


plt.show()
