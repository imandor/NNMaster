from guizero import App, Text, TextBox, Combo, Box, PushButton
from src.database_api_beta import Net_data,Filter,hann
import GuiNetworkStyle
from src.network_functions import run_network_process, initiate_network, run_network,initiate_lickwell_network
from src.preprocessing import lickwells_io

def select_Parameter(object): # new objects (sample _experiments) can be defined here
    if (object == "lickwell__experiment_pfc_future"):
        # tbxNetwork_shape.value = 11
        tbx_epochs.value = 10
        tbx_dropout.value = 0.65
        cbo_evaluate_training.value = "False"
        tbx_slice_size.value = 200
        tbx_stride.value = 200
        tbx_y_step.value = 3
        tbx_win_size.value = 200
        tbx_search_radius.value = 200
        tbx_k_cross_validation.value = 10
        tbx_valid_ratio.value = 0.1
        tbx_testing_ratio.value = 0
        tbx_time_shift_steps.value = 1
        cbo_early_stopping.value = "False"
        tbx_model_path.value = "G:/master_datafiles/trained_networks/MLP_PFC_lickwell__example/"
        tbx_raw_data_path.value = "G:/master_datafiles/raw_data/PFC/"
        tbx_filtered_data_path.value = "session_hc_lw.pkl"
        tbx_metric.value = "discrete"
        cbo_shuffle_data.value = "True"
        tbx_shuffle_factor.value = 10
        tbx_lw_classifications.value = 5
        cboLw_normalize.value = "True"
        cbo_lw_differentiate_false_licks.value = "False"
        tbx_num_wells.value = 5
        tbx_initial_timeshift.value = 1

    if (object == "lickwell__experiment_pfc_memory"):
        # tbxNetwork_shape.value = 11
        tbx_dropout.value = 0.65
        tbx_epochs.value = 10
        cbo_evaluate_training.value = "False"
        tbx_slice_size.value = 200
        tbx_stride.value = 200
        tbx_y_step.value = 3
        tbx_win_size.value = 200
        tbx_search_radius.value = 200
        tbx_k_cross_validation.value = 10
        tbx_valid_ratio.value = 0.1
        tbx_testing_ratio.value = 0
        tbx_time_shift_steps.value = 1
        cbo_early_stopping.value = "False"
        tbx_model_path.value = "G:/master_datafiles/trained_networks/MLP_PFC_lickwell__example/"
        tbx_raw_data_path.value = "G:/master_datafiles/raw_data/PFC/"
        tbx_filtered_data_path.value = "session_pfc_lw"
        tbx_metric.value = "discrete"
        cbo_shuffle_data.value = "True"
        tbx_shuffle_factor.value = 10
        tbx_lw_classifications.value = 5
        cboLw_normalize.value = "True"
        cbo_lw_differentiate_false_licks.value = "False"
        tbx_num_wells.value = 5
        tbx_initial_timeshift.value = -1




    if (object == "position__experiment_pfc"):
        # tbxNetwork_shape.value = 11
        tbx_epochs.value = 30
        tbx_dropout.value = 0.65
        cbo_evaluate_training.value = "False"
        tbx_slice_size.value = 1000
        tbx_stride.value = 100
        tbx_y_step.value = 3
        tbx_win_size.value = 200
        tbx_search_radius.value = 200
        tbx_k_cross_validation.value = 10
        tbx_valid_ratio.value = 0.1
        tbx_testing_ratio.value = 0.1
        tbx_time_shift_steps.value = 41
        cbo_early_stopping.value = "False"
        tbx_model_path.value = "G:/master_datafiles/trained_networks/MLP_PFC__example/"
        tbx_raw_data_path.value = "G:/master_datafiles/raw_data/PFC/"
        tbx_filtered_data_path.value = "session_pfc.pkl"
        tbx_metric.value = "map"
        cbo_shuffle_data.value = "True"
        tbx_shuffle_factor.value = 10
        tbx_lw_classifications.value = None
        cboLw_normalize.value = "False"
        cbo_lw_differentiate_false_licks.value = "False"
        tbx_num_wells.value = 5
        tbx_initial_timeshift.value = -10000


def tbx_model_path_ex():
    explain.value = "where trained network is saved to"


def txt_raw_data_path_ex():
    explain.value = "where unprocessed session is saved"


def txt_dropout_ex():
    explain.value = "dropout of network"


def txt_filtered_data_path_ex():
    explain.value = "path where preprocessed data is saved to (reload in order to save processing time)"


def txt_stride_ex():
    explain.value = "stride [ms]with which output samples are generated (making it smaller than y_slice_size creates overlapping samples"


def txtY_slice_size_ex():
    explain.value = "size of each output sample in [ms]"


def txtNetwork_type_ex():
    explain.value = "is saved to file in order to easier identify which network type was used for model"


def txt_epochs_ex():
    explain.value = "how many epochs the network is trained"


def txtNetwork_shape_ex():
    explain.value = "currently not used, originally showed how many samples are in each batch"


def txt_from_raw_data_ex():
    explain.value = "if True, loads session from raw data path, if False loads session from filtered data path (which is much faster)"


def txt_evaluate_training_ex():
    explain.value = "if set to True, the network evaluates training performance at runtime (slows down training)"


def txt_search_radius_ex():
    explain.value = "convolution search radius"


def txt_step_size_ex():
    explain.value = "convolution step size"


def txt_time_shift_steps_ex():
    explain.value = "after each step, the network resets with new timeshift and reassigns new ys  depending  on the current timeshift. Determines how many steps are performed at runtime (see time_shift_iter)"


def txt_shuffle_data_ex():
    explain.value = " if data is to be shuffled before assigning to training and testing (see shuffle-factor parameter)"


def txt_shuffle_factor_ex():
    explain.value = " shuffling occurs batch-wise (to minimize overlap  if stride is lower than y_slice_size). Indicates how many samples are to be shuffled in a batch."


def txt_time_shift_iter_ex():
    explain.value = "determines, how much the time is shifted after each time_shift_step"


def txt_initial_timeshift_ex():
    explain.value ="initial time_shift for position decoding. For well-decoding,I appropiated this parameter to determine if the previous or n_ext well is to be decoded (sorry). +1 indicates n_ext well, -1 previous well"

def txt_metric_iter_ex():
    explain.value ="after how many epochs the network performance is to be evaluated"

def txtBatch_size_ex():
    explain.value = "how many samples are to be given into the network at once (google batch_size)"

def txt_slice_size_ex():
    explain.value ="how many [ms] of neural spike data are in each sample"

def txt_x_max_ex():
    explain.value ="determines shape of the track the rat is located on [cm]"

def txt_y_max_ex():
    explain.value ="determines shape of the track the rat is located on [cm]"

def txt_x_min_ex():
    explain.value ="determines shape of the track the rat is located on [cm]"

def txt_y_min_ex():
    explain.value ="determines shape of the track the rat is located on [cm]"

def txt_x_step_ex():
    explain.value ="determines, what shape the position bins for the samples are [cm] (3*3cm per bin default)"

def txt_y_step_ex():
    explain.value ="determines, what shape the position bins for the samples are [cm] (3*3cm per bin default)"

def txt_early_stopping_ex():
    explain.value ="if True, checks if network performance degrades after a certain amount of steps (see network) and stops training early if yes"

def txt_naive_test_ex():
    explain.value ="if True, doesn't reassign y_values after each time_shift step. Necessary to determine what part of the network performance  is due to similarities between different time_shift step neural data"

def txt_valid_ratio_ex():
    explain.value ="ratio between training and validation data"

def txt_testing_ratio_ex():
    explain.value ="ration between training and testing data"

def txt_k_cross_validation_ex():
    explain.value ="if more than 1, the data is split into k different sets and results are averaged over all set-performances"

def txt_load_model_ex():
    explain.value ="if True, the saved model is loaded for training instead of a new one.Is set to True if naive testing is True and time-shift is != 0"

def txt_train_model_ex():
    explain.value ="if True, the model is not trained during runtime. Is set to True if naive testing is True and time-shift is != 0"

def txt_keep_neuron_ex():
    explain.value ="if -1 do nothing. If >1 removes neurons from data until equal to count. If == 1 removes fraction of neuron (neuron_kept_factor)"

def txt_neurons_kept_factor_ex():
    explain.value ="if less than one, a corresponding fraction of  neurons are randomly removed from session before training (see keep_neuron)"

def txt_lw_classifications_ex():
    explain.value ="for well decoding: how many classes _exist"

def txt_lw_normalize_ex():
    explain.value ="lickwell_data: if True, training and validation data is normalized (compl_ex rules)"

def txt_lw_differentiate_false_licks_ex():
    explain.value ="Not used anymore due to too small sample sizes and should currently  give an error if True. if True, the network specifically trains to  distinguish between correct and false licks. Should work with minimal code editing if ever necessary to implement"

def txt_num_wells_ex():
    explain.value ="number of lickwells in data set. Really only in object because of lw_differentiate_false_licks, but there  is no reason to remove it either"

def txt_metric_ex():
    explain.value ="with what metric the network is to be evaluated (depending on study)"

def txt_valid_licks_ex():
    explain.value ="List of licks which are valid for well-decoding study"

def txt_filter_tetrodes_ex():
    explain.value = "removes tetrodes from raw session data before creating slice object. Useful if some of the tetrodes are e.g. hippocampal and others for pfc"

def txt_phases_ex():
    explain.value ="contains list of training phases"

def txt_phase_change_ids_ex():
    explain.value ="contains list of phase change lick_ids"

def txt_number_of_bins_ex():
    explain.value ="number of win sized bins going into the input"

def txt_start_time_by_lick_id_ex():
    explain.value ="list of tuples (lick_id,start time) which describe the time at which a lick officially starts relative to the lick time described in the lick object. Defaults to zero but can be changed if a different range is to be observed"

def txtBehavior_component_filter_ex():
    explain.value ="filters session data, string"

def txt_win_size_ex():
    explain.value ="Convolution window size"

def not_show__ex():
    explain.value = ""

def guiRun_network():
    nd = Net_data(
        model_path = tbx_model_path.value,
        raw_data_path = tbx_raw_data_path.value,
        dropout=float(tbx_dropout.value),
        filtered_data_path=tbx_filtered_data_path.value,
        stride=int(tbx_stride.value),
        epochs=int(tbx_epochs.value),
        from_raw_data=bool(cbo_from_raw_data.value == "True"),
        evaluate_training=bool(cbo_evaluate_training.value == "True"),
        session_filter=Filter(func=hann, search_radius=int(tbx_search_radius.value), step_size=int(tbx_step_size.value)),
        time_shift_steps=int(tbx_time_shift_steps.value),
        shuffle_data=bool(cbo_shuffle_data.value == "True"),
        shuffle_factor=int(tbx_shuffle_factor.value),
        time_shift_iter=int(tbx_time_shift_iter.value),
        initial_timeshift=int(tbx_initial_timeshift.value),
        metric_iter=int(tbx_metric_iter.value),
        batch_size=int(tbx_batch_size.value),
        slice_size=int(tbx_slice_size.value),
        x_max=int(tbx_x_max.value),
        y_max=int(tbx_y_max.value),
        x_min=int(tbx_x_min.value),
        y_min=int(tbx_y_min.value),
        x_step=int(tbx_x_step.value),
        y_step=int(tbx_y_step.value),
        early_stopping=bool(cbo_early_stopping.value == "True"),
        naive_test=bool(cbo_naive_test.value == "True"),
        valid_ratio=float(tbx_valid_ratio.value),
        testing_ratio=float(tbx_testing_ratio.value),
        k_cross_validation=int(tbx_k_cross_validation.value),
        load_model=bool(cbo_load_model.value == "True"),
        train_model=bool(cbo_train_model.value == "True"),
        keep_neurons=-1,
        neurons_kept_factor=float(tbx_neurons_kept_factor.value),
        lw_classifications=None if not tbx_lw_classifications.value.isnumeric() else int(tbx_lw_classifications.value),
        lw_normalize=bool(cboLw_normalize.value=="True"),
        lw_differentiate_false_licks=bool(cbo_lw_differentiate_false_licks.value == "True"),
        num_wells=int(tbx_num_wells.value),
        metric=tbx_metric.value,
        valid_licks= None if tbx_valid_licks.value == 'None' else tbx_valid_licks.value,
        filter_tetrodes=None if tbx_filter_tetrodes.value == 'None' else tbx_filter_tetrodes.value,
        phases= None if tbx_phases.value == 'None' else tbx_phases.value,
        phase_change_ids = None if tbx_phase_change_ids.value == 'None' else tbx_phase_change_ids.value,
        number_of_bins=int(tbx_number_of_bins.value),
        start_time_by_lick_id= None if not tbx_start_time_by_lick_id.value.isnumeric() else int(tbx_start_time_by_lick_id.value),
        behavior_component_filter = None if cbo_behavior_component.value == 'None' else cbo_behavior_component.value
    )
    if tbx_metric.value == "map":
        session = initiate_network(nd)
        run_network(nd, session)
    else:
        session = initiate_lickwell_network(nd)  # Initialize session
        X, y, nd, session, = lickwells_io(session, nd, _excluded_wells=[1], shift=nd.initial_timeshift,
                                          target_is_phase=False,
                                          lickstart=0, lickstop=5000)

    pass




if __name__ == '__main__':
    style = GuiNetworkStyle.GuiNetworkStyle
    app1 = App(title="Network")
    titlebox = Box(app1, width="fill", align="top")
    app = Box(app1, height="fill", width="fill", layout="grid")
    message = Text(titlebox, text="GUI", grid=[1, 1])
    message.text_color = style.title_Color
    message.text_size = style.title_Size

    explain = Text(titlebox)
    explain.text_color = style.explain_Color
    explain.text_size = style.explain_Size


    txt__example = Text(app, text="_examples", grid=[1, 4])
    txt__example.text_size = style.title_Size
    txt__example.text_color = style.title_Color
    cbo__example = Combo(app,
                        options=["lickwell__experiment_pfc_future", "lickwell__experiment_pfc_memory", "position__experiment_pfc", "Object4"],
                        grid=[2, 4], command=select_Parameter)
    txt_header1 = Text(app, text="Parameters", grid=[1, 5])
    txt_header1.text_size = style.title_Size
    txt_header1.text_color = style.title_Color

    txt_header2 = Text(app, text="Essential", grid=[2, 5])
    txt_header2.text_size = style.title_Size
    txt_header2.text_color = style.title_Color

    txt_model_path = Text(app, text="Model_Path:", grid=[1, 6], align="left")
    txt_model_path.text_size = style.parameter_Size
    txt_model_path.text_color = style.parameter_Color
    tbx_model_path = TextBox(app, grid=[2, 6], align="left", width=30)
    txt_model_path.when_mouse_enters = tbx_model_path_ex
    txt_model_path.when_mouse_leaves = not_show__ex

    txt_raw_data_path = Text(app, text="Raw_data_Path:", grid=[1, 7], align="left")
    txt_raw_data_path.text_size = style.parameter_Size
    txt_raw_data_path.text_color = style.parameter_Color
    tbx_raw_data_path = TextBox(app, grid=[2, 7], align="left", width=30)
    tbx_raw_data_path.when_mouse_enters = txt_raw_data_path_ex

    txt_filtered_data_path = Text(app, align="left", text="filtered_data_path:", grid=[1, 8])
    txt_filtered_data_path.text_size = style.parameter_Size
    txt_filtered_data_path.text_color = style.parameter_Color
    tbx_filtered_data_path = TextBox(app, align="left", grid=[2, 8], width=30)
    tbx_filtered_data_path.when_mouse_enters = txt_filtered_data_path_ex

    txt_from_raw_data = Text(app, text="from_raw_data:", align="left", grid=[1, 9])
    txt_from_raw_data.text_size = style.parameter_Size
    txt_from_raw_data.text_color = style.parameter_Color
    cbo_from_raw_data = Combo(app, options=["True", "False"], align="left", grid=[2, 9], selected="True")
    cbo_from_raw_data.when_mouse_enters = txt_from_raw_data_ex

    txt_initial_timeshift = Text(app, text="initial_timeshift:", align="left", grid=[1, 10])
    txt_initial_timeshift.text_size = style.parameter_Size
    txt_initial_timeshift.text_color = style.parameter_Color
    tbx_initial_timeshift = TextBox(app, align="left", grid=[2, 10], width=30)
    tbx_initial_timeshift.value = 0
    tbx_initial_timeshift.when_mouse_enters= txt_initial_timeshift_ex

    txt_metric = Text(app, text="metric:", align="left", grid=[1, 11])
    txt_metric.text_size = style.parameter_Size
    txt_metric.text_color = style.parameter_Color
    tbx_metric = TextBox(app, align="left", grid=[2, 11], width=30)
    tbx_metric.value = "map"
    tbx_metric.when_mouse_enters= txt_metric_ex

    txt_filter_tetrodes = Text(app, text="filter_tetrodes:", align="left", grid=[1, 12])
    txt_filter_tetrodes.text_size = style.parameter_Size
    txt_filter_tetrodes.text_color = style.parameter_Color
    tbx_filter_tetrodes = TextBox(app, align="left", grid=[2, 12], width=30)
    tbx_filter_tetrodes.value = None
    tbx_filter_tetrodes.when_mouse_enters= txt_filter_tetrodes_ex

    txt_header3 = Text(app, text="Important", grid=[2, 13])
    txt_header3.text_size = style.title_Size
    txt_header3.text_color = style.title_Color

    txt_stride = Text(app, text="stride:", align="left", grid=[1, 14])
    txt_stride.text_size = style.parameter_Size
    txt_stride.text_color = style.parameter_Color
    tbx_stride = TextBox(app, align="left", grid=[2, 14], width=30)
    txt_stride.when_mouse_enters= txt_stride_ex
    tbx_stride.value = 100

    txt_epochs = Text(app, text="epochs:", align="left", grid=[1, 15])
    txt_epochs.text_size = style.parameter_Size
    txt_epochs.text_color = style.parameter_Color
    tbx_epochs = TextBox(app, align="left", grid=[2, 15], width=30)
    tbx_epochs.value = 20
    tbx_epochs.when_mouse_enters = txt_epochs_ex

    txt_search_radius = Text(app, text="search_radius:", align="left", grid=[1, 16])
    txt_search_radius.text_size = style.parameter_Size
    txt_search_radius.text_color = style.parameter_Color
    tbx_search_radius = TextBox(app, align="left", grid=[2, 16], width=30)
    tbx_search_radius.value = 200
    tbx_search_radius.when_mouse_enters= txt_search_radius_ex

    txt_step_size = Text(app, text="step_size:", align="left", grid=[1, 17])
    txt_step_size.text_size = style.parameter_Size
    txt_step_size.text_color = style.parameter_Color
    tbx_step_size = TextBox(app, align="left", grid=[2, 17], width=30)
    tbx_step_size.value = 100
    tbx_step_size.when_mouse_enters= txt_step_size_ex

    txt_time_shift_steps = Text(app, text="time_shift_steps:", align="left", grid=[1, 18])
    txt_time_shift_steps.text_size = style.parameter_Size
    txt_time_shift_steps.text_color = style.parameter_Color
    tbx_time_shift_steps = TextBox(app, align="left", grid=[2, 18], width=30)
    tbx_time_shift_steps.value = 1
    tbx_time_shift_steps.when_mouse_enters= txt_time_shift_steps_ex

    txt_shuffle_data = Text(app, text="shuffle_data:", align="left", grid=[1, 19])
    txt_shuffle_data.text_size = style.parameter_Size
    txt_shuffle_data.text_color = style.parameter_Color
    cbo_shuffle_data = Combo(app, options=["True", "False"], align="left", grid=[2, 19], selected="True")
    cbo_shuffle_data.when_mouse_enters = txt_shuffle_data_ex



    txt_shuffle_factor = Text(app, text="shuffle_factor:", align="left", grid=[1, 20])
    txt_shuffle_factor.text_size = style.parameter_Size
    txt_shuffle_factor.text_color = style.parameter_Color
    tbx_shuffle_factor = TextBox(app, align="left", grid=[2, 20], width=30)
    tbx_shuffle_factor.value = 10
    tbx_shuffle_factor.when_mouse_enters = txt_shuffle_factor_ex

    txt_time_shift_iter = Text(app, text="time_shift_iter:", align="left", grid=[1, 21])
    txt_time_shift_iter.text_size = style.parameter_Size
    txt_time_shift_iter.text_color = style.parameter_Color
    tbx_time_shift_iter = TextBox(app, align="left", grid=[2, 21], width=30)
    tbx_time_shift_iter.value = 500
    tbx_time_shift_iter.when_mouse_enters = txt_time_shift_iter_ex



    txt_metric_iter = Text(app, text="metric_iter:", align="left", grid=[1, 23])
    txt_metric_iter.text_size = style.parameter_Size
    tbx_metric_iter = TextBox(app, align="left", grid=[2, 23], width=30)
    tbx_metric_iter.value = 1
    tbx_metric_iter.when_mouse_enters= txt_metric_iter_ex

    txt_batch_size = Text(app, text="batch_size:", align="left", grid=[1, 24])
    txt_batch_size.text_size = style.parameter_Size
    txt_batch_size.text_color = style.parameter_Color
    tbx_batch_size = TextBox(app, align="left", grid=[2, 24], width=30)
    tbx_batch_size.value = 50
    tbx_batch_size.when_mouse_enters= txtBatch_size_ex

    txt_slice_size = Text(app, text="slice_size:", align="left", grid=[1, 25])
    txt_slice_size.text_size = style.parameter_Size
    txt_slice_size.text_color = style.parameter_Color
    tbx_slice_size = TextBox(app, align="left", grid=[2, 25], width=30)
    tbx_slice_size.value = 1000
    tbx_slice_size.when_mouse_enters = txt_slice_size_ex

    txt_early_stopping = Text(app, text="early_stopping:", align="left", grid=[1, 26])
    txt_early_stopping.text_size = style.parameter_Size
    txt_early_stopping.text_color = style.parameter_Color
    cbo_early_stopping = Combo(app, options=["True", "False"], align="left", grid=[2, 26], selected="False")
    cbo_early_stopping.when_mouse_enters= txt_early_stopping_ex

    txt_valid_ratio = Text(app, text="valid_ratio:", align="left", grid=[1, 27])
    txt_valid_ratio.text_size = style.parameter_Size
    txt_valid_ratio.text_color = style.parameter_Color
    tbx_valid_ratio = TextBox(app, align="left", grid=[2, 27], width=30)
    tbx_valid_ratio.value = 0.1
    tbx_valid_ratio.when_mouse_enters=txt_valid_ratio_ex

    txt_testing_ratio = Text(app, text="testing_ratio:", align="left", grid=[1, 28])
    txt_testing_ratio.text_size = style.parameter_Size
    txt_testing_ratio.text_color = style.parameter_Color
    tbx_testing_ratio = TextBox(app, align="left", grid=[2, 28], width=30)
    tbx_testing_ratio.value = 0.1
    tbx_testing_ratio.when_mouse_enters= txt_testing_ratio_ex

    txt_k_cross_validation = Text(app, text="k_cross_validation:", align="left", grid=[1, 29])
    txt_k_cross_validation.text_size = style.parameter_Size
    txt_k_cross_validation.text_color = style.parameter_Color
    tbx_k_cross_validation = TextBox(app, align="left", grid=[2, 29], width=30)
    tbx_k_cross_validation.value = 1
    tbx_k_cross_validation.when_mouse_enters= txt_k_cross_validation_ex

    txt_number_of_bins = Text(app, text="number_of_bins:", align="left", grid=[1, 30])
    txt_number_of_bins.text_size = style.parameter_Size
    txt_number_of_bins.text_color = style.parameter_Color
    tbx_number_of_bins = TextBox(app, align="left", grid=[2, 30], width=30)
    tbx_number_of_bins.value = 10
    tbx_number_of_bins.when_mouse_enters= txt_number_of_bins_ex

    txt_dropout = Text(app, text="dropOut:", grid=[1, 31], align="left")
    txt_dropout.text_size = style.parameter_Size
    txt_dropout.text_color = style.parameter_Color
    tbx_dropout = TextBox(app, grid=[2, 31], align="left", width=30)
    tbx_dropout.when_mouse_enters= txt_dropout_ex

    txt_win_size = Text(app, text="behavior_component_filter:", align="left", grid=[1, 32])
    txt_win_size.text_size = style.parameter_Size
    txt_win_size.text_color = style.parameter_Color
    tbx_win_size = TextBox(app, align="left", grid=[2, 32], width=30)
    tbx_win_size.value = 100
    tbx_win_size.when_mouse_enters = txt_win_size_ex
    
    txt_header4 = Text(app, text="Specific _experiments", grid=[4, 5])
    txt_header4.text_size = style.title_Size
    txt_header4.text_color = style.title_Color

    txt_behavior_component = Text(app, text="behavior component filter", grid=[3, 6])
    txt_behavior_component.text_size = style.parameter_Size
    txt_behavior_component.text_color = style.parameter_Color
    cbo_behavior_component = Combo(app,
                                   options=["None","at lickwell", "not at lickwell", "correct trials", "incorrect trials","move", "rest"],
                                   grid=[4, 6], command=select_Parameter)

    txt_naive_test = Text(app, text="naive_test:", align="left", grid=[3, 7])
    txt_naive_test.text_size = style.parameter_Size
    txt_naive_test.text_color = style.parameter_Color
    cbo_naive_test = Combo(app, options=["True", "False"], align="left", grid=[4, 7], selected="False")
    cbo_naive_test.when_mouse_enters= txt_naive_test_ex

    txt_load_model = Text(app, text="load_model:", align="left", grid=[3, 8])
    txt_load_model.text_size = style.parameter_Size
    txt_load_model.text_color = style.parameter_Color
    cbo_load_model = Combo(app, options=["True", "False"], align="left", grid=[4, 8], selected="False")
    cbo_load_model.when_mouse_enters= txt_load_model_ex

    txt_train_model = Text(app, text="train_model:", align="left", grid=[3, 9])
    txt_train_model.text_size = style.parameter_Size
    txt_train_model.text_color = style.parameter_Color
    cbo_train_model = Combo(app, options=["True", "False"], align="left", grid=[4, 9], selected="True")
    cbo_train_model.when_mouse_enters= txt_train_model_ex

    txt_keep_neuron = Text(app, text="keep_neurons:", align="left", grid=[3, 10])
    txt_keep_neuron.text_size = style.parameter_Size
    txt_keep_neuron.text_color = style.parameter_Color
    tbx_keep_neuron = TextBox(app, align="left", grid=[4, 10], width=30)
    tbx_keep_neuron.value = -1
    tbx_keep_neuron.when_mouse_enters= txt_keep_neuron_ex

    txt_neurons_kept_factor = Text(app, text="neurons_kept_factor:", align="left", grid=[3, 11])
    txt_neurons_kept_factor.text_size = style.parameter_Size
    txt_neurons_kept_factor.text_color = style.parameter_Color
    tbx_neurons_kept_factor = TextBox(app, align="left", grid=[4, 11], width=30)
    tbx_neurons_kept_factor.value = 1.0
    tbx_neurons_kept_factor.when_mouse_enters= txt_neurons_kept_factor_ex

    txt_start_time_by_lick_id = Text(app, text="start_time_by_lick_id:", align="left", grid=[3, 12])
    txt_start_time_by_lick_id.text_size = style.parameter_Size
    txt_start_time_by_lick_id.text_color = style.parameter_Color
    tbx_start_time_by_lick_id = TextBox(app, align="left", grid=[4, 12], width=30)
    tbx_start_time_by_lick_id.value = None
    tbx_start_time_by_lick_id.when_mouse_enters= txt_start_time_by_lick_id_ex

    txt_header5 = Text(app, text="no changes necessary", grid=[4, 13])
    txt_header5.text_size = style.title_Size
    txt_header5.text_color = style.title_Color

    txt_evaluate_training = Text(app, text="evaluate_training:", align="left", grid=[3, 14])
    txt_evaluate_training.text_size = style.parameter_Size
    txt_evaluate_training.text_color = style.parameter_Color
    cbo_evaluate_training = Combo(app, options=["True", "False"], align="left", grid=[4, 14], selected="False")
    cbo_evaluate_training.when_mouse_enters = txt_evaluate_training_ex

    txt_x_max = Text(app, text="x_max:", align="left", grid=[3, 15])
    txt_x_max.text_size = style.parameter_Size
    txt_x_max.text_color = style.parameter_Color
    tbx_x_max = TextBox(app, align="left", grid=[4, 15], width=30)
    tbx_x_max.value = 240
    tbx_x_max.when_mouse_enters= txt_x_max_ex

    txt_y_max = Text(app, text="y_max:", align="left", grid=[3, 16])
    txt_y_max.text_size = style.parameter_Size
    txt_y_max.text_color = style.parameter_Color
    tbx_y_max = TextBox(app, align="left", grid=[4, 16], width=30)
    tbx_y_max.value = 190
    tbx_y_max.when_mouse_enters=txt_y_max_ex

    txt_x_min = Text(app, text="x_min:", align="left", grid=[3, 17])
    txt_x_min.text_size = style.parameter_Size
    txt_x_min.text_color = style.parameter_Color
    tbx_x_min = TextBox(app, align="left", grid=[4, 17], width=30)
    tbx_x_min.value = 0
    tbx_x_min.when_mouse_enters= txt_x_min_ex

    txt_y_min = Text(app, text="y_min:", align="left", grid=[3, 18])
    txt_y_min.text_size = style.parameter_Size
    txt_y_min.text_color = style.parameter_Color
    tbx_y_min = TextBox(app, align="left", grid=[4, 18], width=30)
    tbx_y_min.value = 100
    tbx_y_min.when_mouse_enters= txt_y_min_ex

    txt_x_step = Text(app, text="x_step:", align="left", grid=[3, 19])
    txt_x_step.text_size = style.parameter_Size
    txt_x_step.text_color = style.parameter_Color
    tbx_x_step = TextBox(app, align="left", grid=[4, 19], width=30)
    tbx_x_step.value = 3
    tbx_x_step.when_mouse_enters= txt_x_step_ex

    txt_y_step = Text(app, text="y_step:", align="left", grid=[3, 20])
    txt_y_step.text_size = style.parameter_Size
    txt_y_step.text_color = style.parameter_Color
    tbx_y_step = TextBox(app, align="left", grid=[4, 20], width=30)
    tbx_y_step.value = 3
    tbx_y_max.when_mouse_enters= txt_y_step_ex

    txt_lw_classifications = Text(app, text="lw_classifications:", align="left", grid=[3, 21])
    txt_lw_classifications.text_size = style.parameter_Size
    txt_lw_classifications.text_color = style.parameter_Color
    tbx_lw_classifications = TextBox(app, align="left", grid=[4, 21], width=30)
    tbx_lw_classifications.value = None
    tbx_lw_classifications.when_mouse_enters = txt_lw_classifications_ex

    txt_lw_normalize = Text(app, text="lw_normalize:", align="left", grid=[3, 22])
    txt_lw_normalize.text_size = style.parameter_Size
    txt_lw_normalize.text_color = style.parameter_Color
    cboLw_normalize = Combo(app, options=["True", "False"], align="left", grid=[4, 22], selected="False")
    cboLw_normalize.when_mouse_enters= txt_lw_normalize_ex

    txt_lw_differentiate_false_licks = Text(app, text="lw_differentiate_false_licks:", align="left", grid=[3, 23])
    txt_lw_differentiate_false_licks.text_size = style.parameter_Size
    txt_lw_differentiate_false_licks.text_color = style.parameter_Color
    cbo_lw_differentiate_false_licks = Combo(app, options=["True", "False"], align="left", grid=[4, 23], selected="False")
    cbo_lw_differentiate_false_licks.when_mouse_enters= txt_lw_differentiate_false_licks_ex

    txt_num_wells = Text(app, text="num_wells:", align="left", grid=[3, 24])
    txt_num_wells.text_size = style.parameter_Size
    txt_num_wells.text_color = style.parameter_Color
    tbx_num_wells = TextBox(app, align="left", grid=[4, 24], width=30)
    tbx_num_wells.value = 5
    tbx_num_wells.when_mouse_enters= txt_num_wells_ex

    txt_valid_licks = Text(app, text="valid_licks:", align="left", grid=[3, 25])
    txt_valid_licks.text_size = style.parameter_Size
    txt_valid_licks.text_color = style.parameter_Color
    tbx_valid_licks = TextBox(app, align="left", grid=[4, 25], width=30)
    tbx_valid_licks.value = None
    tbx_valid_licks.when_mouse_enters= txt_valid_licks_ex

    txt_phases = Text(app, text="phases:", align="left", grid=[3, 26])
    txt_phases.text_size = style.parameter_Size
    txt_phases.text_color = style.parameter_Color
    tbx_phases = TextBox(app, align="left", grid=[4, 26], width=30)
    tbx_phases.value = None
    tbx_phases.when_mouse_enters= txt_phases_ex

    txt_phase_change_ids = Text(app, text="phase_change_ids:", align="left", grid=[3, 27])
    txt_phase_change_ids.text_size = style.parameter_Size
    txt_phase_change_ids.text_color = style.parameter_Color
    tbx_phase_change_ids = TextBox(app, align="left", grid=[4, 27], width=30)
    tbx_phase_change_ids.value = None
    tbx_phase_change_ids.when_mouse_enters= txt_phase_change_ids_ex
    # txtY_slice_size = Text(app, text="y_slice_size:", align="left", grid=[1, 10])
    # txtY_slice_size.text_size = style.parameter_Size
    # txtY_slice_size.text_color = style.parameter_Color
    # tbxY_slice_size = TextBox(app, align="left", grid=[2, 10], width=30)
    # tbxY_slice_size.when_mouse_enters = txtY_slice_size_ex
    # tbxY_slice_size.value = 100

    # txtNetwork_type = Text(app, text="network_type:", align="left", grid=[1, 11])
    # txtNetwork_type.text_size = style.parameter_Size
    # txtNetwork_type.text_color = style.parameter_Color
    # tbxNetwork_type = TextBox(app, align="left", grid=[2, 11], width=30)
    # tbxNetwork_type.value = "MLP"
    # tbxNetwork_type.when_mouse_enters = txtNetwork_type_ex


    #
    # txtNetwork_shape = Text(app, text="network_shape:", align="left", grid=[1, 13])
    # txtNetwork_shape.text_size = style.parameter_Size
    # txtNetwork_shape.text_color = style.parameter_Color
    # tbxNetwork_shape = TextBox(app, align="left", grid=[2, 13], width=30)
    # tbxNetwork_shape.value = 10
    # tbxNetwork_shape.when_mouse_enters = txtNetwork_shape_ex









    # txtBehavior_component_filter = Text(app, text="behavior_component_filter:", align="left", grid=[3, 30])
    # txtBehavior_component_filter.text_size = style.parameter_Size
    # txtBehavior_component_filter.text_color = style.parameter_Color
    # tbxBehavior_component_filter = TextBox(app, align="left", grid=[4, 30], width=30)
    # tbxBehavior_component_filter.value = None
    # tbxBehavior_component_filter.when_mouse_enters= txtBehavior_component_filter_ex



    button = PushButton(app, grid=[4,28],command=guiRun_network,text="run program")
    app1.display()
