
from database_api import Slice, Session, Trial
from database_api import TimePoint
from settings import save_as_pickle, load_pickle


def save_sample_session():
    """ creates session, slice and trial file for testing"""
    start = TimePoint(ms=0)
    stop = TimePoint(ms=10000000)
    session = Session(session_file="session.pkl")
    slice = session.make_slice(start=start, stop=stop)
    slice_sample = slice.make_slice(start=TimePoint(ms=70000), stop=TimePoint(ms=80000))
    trial_sample = slice.get_trial_by_time(TimePoint(ms=70000))
    save_as_pickle("session.pkl",session)
    save_as_pickle("slice_sample.pkl", slice_sample)
    save_as_pickle("trial_sample.pkl", trial_sample)
    pass


def test_trial_sample():
    session = Session(session_file="session.pkl")
    trial = load_pickle("trial_sample.pkl")
    trial.plot_spikes()
    pass