from multiprocessing.pool import ThreadPool
import itertools
from src.filters import bin_filter


def multi_process_convolution(work, search_window_size, step_size):
    work.set_filter(filter=bin_filter, search_window_size=search_window_size, step_size=step_size)
    return work


def test_process(n,m):
    print("Test " , n,m)
    return n


def bin_slices_spikes(data_slice, search_window_size, step_size, num_threads):
    # split work into even chunks
    n = len(data_slice.position_x) // num_threads # size of chunks
    work = [data_slice[i:i + n] for i in range(0, len(data_slice.position_x), n)]
    pool = ThreadPool(processes=num_threads)
    args = list(zip(work, itertools.repeat(search_window_size,num_threads),itertools.repeat(step_size,num_threads))) # zips all arguments for each process instance
    pool.starmap(multi_process_convolution,args)

    return_slice = work[0]
    for i in range(1,num_threads):# TODO ask Charles for better method
        return_slice = return_slice + work[i]
    return return_slice
