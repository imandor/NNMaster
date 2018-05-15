
from database_api_beta import Slice, Trial
from settings import save_as_pickle, load_pickle


def save_sample_session():
    """ creates session, slice and trial file for testing"""
    start = 0
    stop = 10000000

    # slice = session.make_slice(start=start, stop=stop)
    slice_sample = slice.make_slice(start=70000, stop=80000)
    trial_sample = slice.get_trial_by_time(trial_time=70000)
    save_as_pickle("session.pkl",session)
    save_as_pickle("slice_sample.pkl", slice_sample)
    save_as_pickle("trial_sample.pkl", trial_sample)
    pass


def test_trial_sample():
    trial = load_pickle("trial_sample.pkl")
    trial.plot_spikes()
    pass