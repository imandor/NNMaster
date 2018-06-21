from queue import Queue
from threading import Thread, Lock


def threading_convolution(work, search_window_size, step_size, ):
    work._convolve(search_window_size=search_window_size, step_size=step_size)
        # return work




def _set_convolution_queue(data_slice_c, search_window_size, step_size, num_threads):
    data_slice = data_slice_c
    # slices work into even chunks
    n = len(data_slice.position_x) // num_threads# size of chunks
    work = [data_slice[i:i + n] for i in range(0, len(data_slice.position_x), n)]
    for i in range(num_threads):
        worker = Thread(target=threading_convolution, args=(work[i], search_window_size, step_size))
        worker.setDaemon(True)
        worker.start()
    return_slice = work[0]
    for i in range(1,num_threads):
        return_slice = return_slice + work[i]
    return return_slice


