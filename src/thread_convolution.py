from queue import Queue
from threading import Thread, Lock


def threading_convolution(q, search_window_size, step_size, ):
    while True:
        q.get()._convolve(search_window_size=search_window_size, step_size=step_size)
        q.task_done()


def _set_convolution_queue(data_slice_c, search_window_size, step_size, num_threads, queue_size):
    q = Queue(maxsize=queue_size)
    queue_lock = Lock()
    data_slice = data_slice_c
    # slices work into even chunks
    n = len(data_slice.position_x) // queue_size  # size of chunks
    work = [data_slice[i:i + queue_size] for i in range(0, len(data_slice.position_x), n)]
    for i in range(num_threads):
        worker = Thread(target=threading_convolution, args=(q, search_window_size, step_size))
        worker.setDaemon(True)
        worker.start()

    # add work to queue
    queue_lock.acquire()
    for i in range(0, len(work)):
        q.put(work[i])
    queue_lock.release()
    q.join()
