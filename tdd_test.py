
from database_api import Slice,Session
from database_api import TimePoint


def session_test():
    try:
        start = TimePoint(ms=0)
        stop = TimePoint(ms=10000000)
        session = Session(session_file="session.pkl")
        slice = session.make_slice(start=start, stop=stop)
        spike_sample = slice.spikes[6]
        lick_sample = slice.licks[0][0]
        slice_sample = slice.make_slice(start=TimePoint(ms=70000), stop=TimePoint(ms=80000))
        trial_sample = slice.get_trial_by_time(TimePoint(ms=70000))
        trials_sample = slice.get_all_trials_by_time(start=start, stop=stop)
    except ValueError:
        print("Error: session test failed")


def test_all():
    session_test()
