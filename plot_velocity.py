import database_api_beta as api
import numpy as np
import matplotlib.pyplot as plt
from src.filters import bin_filter
import time

def f(x):
    return 1.0 if np.abs(x) < 350 else 0.0


fig = plt.figure()
ax_position_x = fig.add_subplot(4, 1, 1)
ax_speed = fig.add_subplot(4, 1, 2,sharex=ax_position_x)
ax_filtered_spikes = fig.add_subplot(4,1,3,sharex=ax_position_x)
ax_filtered_spikes_sum = fig.add_subplot(4,1,4,sharex=ax_position_x) # TODO Scala für binned data dazuaddieren
# ax_licks = fig.add_subplot(4, 2, 5)
# ax_trial_timestamps = fig.add_subplot(4, 2, 6)

data_slice = api.Slice.from_path(load_from="data/pickle/slice.pkl")
for i in range(len(data_slice.licks)):
    t = int(data_slice.licks[i]["time"])
    l = data_slice.licks[i]["lickwell"]
    p = int(data_slice.position_x[int(t)])
    print("lickwell ",l,",   position ", p, "cm,   time ",t,"ms")
data_slice = data_slice[0:300000]
data_slice.neuron_filter(300)
data_slice.set_filter(filter=bin_filter, search_window_size=500, step_size=500, num_threads=1)
filtered_spikes_kwargs = {'sharex': True}
data_slice.plot_2(ax_speed,ax_position_x,ax_filtered_spikes, ax_filtered_spikes_sum,sharex=True)
plt.show()
# data_slice.set_filter(filter=bin_filter, search_window_size=700, step_size=700, num_threads=20)
plt.show()
