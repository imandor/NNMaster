import database_api_beta as api
import numpy as np
import matplotlib.pyplot as plt
from src.thread_convolution import _set_convolution_queue
def f(x):
    return 1.0 if np.abs(x) < 350 else 0.0

data_slice = api.Slice.from_path(load_from="data/pickle/slice.pkl")
data_slice = data_slice[0:100000]
data_slice = _set_convolution_queue(data_slice, search_window_size=700, step_size=700, num_threads=10)

data_slice.set_filter(f, search_window_size=700, step_size=700)
print("finn")
