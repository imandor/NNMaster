from database_api import Slice,Session
from session_loader import make_session
from database_api import TimePoint, Slice
from tdd_test import test_all

start = TimePoint(ms=0)
stop = TimePoint(ms=10000000)

# session = Session(session_file="session.pkl")
# session = Session(save_session_as="session.pkl")
# session = Session(session_file="session.pkl")

test_all()
print("fin")


# time_slice = session.time_slice(start, stop)
# trials = session.get_trials()
# trials = session.trials  # list of Slices corresponding to the trials
#
# filtered_time_slice = time_slice.convolve()  # window
# frame_size = TimePoint(ms=0)
# frame_stride = TimePoint(ms=150)
# filtered_frames = filtered_time_slice.to_frames(frame_size, frame_stride)  # a list of FilteredSlice
# frame1 = filtered_frames[0]
# filtered_spikes = frame1.data  # a numpy array
# metadata = frame1.data  # a dictionary
#
# start = TimePoint(ms=0)
# stop = TimePoint(ms=1000)
# # or equivalently
# start = TimePoint(s=0)
# stop = TimePoint(s=1)
#
# time_slice = session.time_slice(start, stop)
# trials = session.trials  # list of Slices corresponding to the trials
#
# filtered_time_slice = time_slice.convolve()  # window)
# frame_size = TimePoint(ms=0)
# frame_stride = TimePoint(ms=150)
# filtered_frames = filtered_time_slice.to_frames(frame_size, frame_stride)  # a list of FilteredSlice
# frame1 = filtered_frames[0]
# filtered_spikes = frame1.data  # a numpy array
# metadata = frame1.data  # a dictionary
#
# my_data_slice = Slice("../data/path_to_folder/")
# start = TimePoint(ms=0)
# end = TimePoint(ms=150)
# my_new_data_slice = my_data_slice.time_slice(start, end)
# my_data_slice.is_convolved  # ---> returns False
# my_data_slice.set_filter()  # np_filter)
# my_data_slice.is_convolved  # ---> returns True
#
# # returns a list of numy arrays with the convolved data
# my_frames = my_data_slice.to_frames(TimePoint(ms=100), TimePoint(ms=10))
#
# my_trial = my_data_slice.get_trials()

print("fin")
