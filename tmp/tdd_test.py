from src.database_api_beta import Slice
from src.settings import save_as_pickle, load_pickle
from numpy.random import seed
from src.settings import config


def test_objects():
    seed(1)
    # test_CNN()
    # data_slice = Slice.from_path(save_as="slice.pkl")
    data_slice = Slice.from_path()
    data_slice = Slice.from_path(load_from="data/pickle/slice.pkl")  # load a data slice containing entire session

    smaller_data_slice = data_slice[0:200000]  # slices first 200 seconds of session

    # smaller_data_slice.set_filter(filter=bin_filter,search_window_size=1) # convolves data into bins
    phases = data_slice.get_all_phases()  # gets all training phases as list of data slices
    one_phase = phases[0]
    trial = one_phase.get_nth_trial(0)  # gets first trial in first training phase
    list_of_trials = data_slice.get_trials(
        slice(0, 200000))  # returns a list of trials in the first 200000 ms of session
    sub_list_of_trials = list_of_trials[0:10]  # slices first 10 entries in list of trials

    nth_trial_list = phases.get_nth_trial_in_each_phase(1)  # gets list of all second trials in list of phases
    all_trials_in_second_phase = phases[2].get_all_trials()  # returns a list of all trials in second training phase

    all_trials_starting_with_well_1 = all_trials_in_second_phase.filter_trials_by_well(
        start_well=1)  # returns all trials in second phase that start with a lick at well 1
    return data_slice, smaller_data_slice, phases, trial, list_of_trials, sub_list_of_trials, nth_trial_list, all_trials_in_second_phase


def save_test_objects():
    data_slice, smaller_data_slice, phases, trial, list_of_trials, sub_list_of_trials, nth_trial_list, all_trials_in_second_phase = test_objects()
    save_as_pickle(config["paths"]["data_slice"], data_slice)
    save_as_pickle(config["paths"]["smaller_data_slice"], smaller_data_slice)
    save_as_pickle(config["paths"]["phases"], phases)
    save_as_pickle(config["paths"]["trial"], trial)
    save_as_pickle(config["paths"]["list_of_trials"], list_of_trials)
    save_as_pickle(config["paths"]["sub_list_of_trials"], sub_list_of_trials)
    save_as_pickle(config["paths"]["nth_trial_list"], nth_trial_list)
    save_as_pickle(config["paths"]["all_trials_in_second_phase"], all_trials_in_second_phase)


def compare_test_objects():
    data_slice, smaller_data_slice, phases, trial, list_of_trials, sub_list_of_trials, nth_trial_list, all_trials_in_second_phase = test_objects()
    data_slice_from_file =load_pickle(config["paths"]["data_slice"])
    smaller_data_slice_from_file =load_pickle(config["paths"]["smaller_data_slice"])
    phases_from_file =load_pickle(config["paths"]["phases"])
    trial_from_file =load_pickle(config["paths"]["trial"])
    list_of_trials_from_file =load_pickle(config["paths"]["list_of_trials"])
    sub_list_of_trials_from_file =load_pickle(config["paths"]["sub_list_of_trials"])
    nth_trial_list_from_file =load_pickle(config["paths"]["nth_trial_list"])
    all_trials_in_second_phase_from_file =load_pickle(config["paths"]["all_trials_in_second_phase"])

    if data_slice == data_slice_from_file:
        print("data_slice test succeeded")
    else:
        print("data_slice test failed")
    if smaller_data_slice== smaller_data_slice_from_file:
        print("smaller_data_slice test succeeded")
    else:
        print("smaller_data_slice test failed")
    if phases== phases_from_file:
        print("phases test succeeded")
    else:
        print("phases test failed")
    if trial== trial_from_file:
        print("trial test succeeded")
    else:
        print("trial test failed")
    if list_of_trials== list_of_trials_from_file:
        print("list_of_trials test succeeded")
    else:
        print("list_of_trials test failed")
    if sub_list_of_trials== sub_list_of_trials_from_file:
        print("sub_list_of_trials test succeeded")
    else:
        print("sub_list_of_trials test failed")
    if nth_trial_list== nth_trial_list_from_file:
        print("nth_trial_list test succeeded")
    else:
        print("nth_trial_list test failed")
    if all_trials_in_second_phase== all_trials_in_second_phase_from_file:
        print("all_trials_in_second_phase test succeeded")
    else:
        print("all_trials_in_second_phase test failed")
    pass

