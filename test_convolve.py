import database_api_beta as api
import numpy as np
import matplotlib.pyplot as plt
import time

def f(x):
    return 1.0 if np.abs(x) < 350 else 0.0


fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)

data_slice = api.Slice.from_path(load_from="data/pickle/slice.pkl")
data_slice.set_filter(f, search_window_size=700, step_size=700)
data_slice.plot_filtered_spikes(ax, neuron_px_height=20, normalize_each_neuron=True)

fig.show()
time.sleep(10)
