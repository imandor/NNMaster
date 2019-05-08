from guizero import App, Text, TextBox, Combo, Box, PushButton
from src.database_api_beta import Net_data,Filter,hann
import GuiNetworkStyle
from src.network_functions import run_network_process, initiate_network, run_network,initiate_lickwell_network
from src.preprocessing import lickwells_io

def select_Parameter(object): # new objects (sample experiments) can be defined here
    if (object == "lickwell_experiment_pfc_future"):
        # tbxNetwork_shape.value = 11
        tbxEpochs.value = 10
        cboEvaluate_training.value = "False"
        tbxSlice_size.value = 200
        tbxStride.value = 200
        tbxY_step.value = 200
        tbxWin_size.value = 200
        tbxSearch_radius.value = 200
        tbxK_cross_validation.value = 10
        tbxValid_ratio.value = 0.1
        tbxTesting_ratio.value = 0
        tbxTime_shift_steps.value = 1
        cboEarly_stopping.value = "False"
        tbxModel_path.value = "G:/master_datafiles/trained_networks/MLP_PFC_lickwell_example/"
        tbxRaw_data_path.value = "G:/master_datafiles/raw_data/PFC/"
        tbxFiltered_data_path.value = "session_hc_lw.pkl"
        tbxMetric.value = "discrete"
        cboShuffle_data.value = "True"
        tbxShuffle_factor.value = 1
        tbxLw_classifications.value = 5
        cboLw_normalize.value = "True"
        cboLw_differentiate_false_licks.value = "False"
        tbxNum_wells.value = 5
        tbxInitial_timeshift.value = 1

    if (object == "lickwell_experiment_pfc_memory"):
        # tbxNetwork_shape.value = 11
        tbxEpochs.value = 10
        cboEvaluate_training.value = "False"
        tbxSlice_size.value = 200
        tbxStride.value = 200
        tbxY_step.value = 200
        tbxWin_size.value = 200
        tbxSearch_radius.value = 200
        tbxK_cross_validation.value = 10
        tbxValid_ratio.value = 0.1
        tbxTesting_ratio.value = 0
        tbxTime_shift_steps.value = 1
        cboEarly_stopping.value = "False"
        tbxModel_path.value = "G:/master_datafiles/trained_networks/MLP_PFC_lickwell_example/"
        tbxRaw_data_path.value = "G:/master_datafiles/raw_data/PFC/"
        tbxFiltered_data_path.value = "session_pfc_lw"
        tbxMetric.value = "discrete"
        cboShuffle_data.value = "True"
        tbxShuffle_factor.value = 1
        tbxLw_classifications.value = 5
        cboLw_normalize.value = "True"
        cboLw_differentiate_false_licks.value = "False"
        tbxNum_wells.value = 5
        tbxInitial_timeshift.value = -1




    if (object == "position_experiment_pfc"):
        # tbxNetwork_shape.value = 11
        tbxEpochs.value = 30
        tbxDropout.value = 0.65
        cboEvaluate_training.value = "False"
        tbxSlice_size.value = 200
        tbxStride.value = 200
        tbxY_step.value = 200
        tbxWin_size.value = 200
        tbxSearch_radius.value = 200
        tbxK_cross_validation.value = 10
        tbxValid_ratio.value = 0.1
        tbxTesting_ratio.value = 0
        tbxTime_shift_steps.value = 41
        cboEarly_stopping.value = "False"
        tbxModel_path.value = "G:/master_datafiles/trained_networks/MLP_PFC_example/"
        tbxRaw_data_path.value = "G:/master_datafiles/raw_data/PFC/"
        tbxFiltered_data_path.value = "session_pfc.pkl"
        tbxMetric.value = "map"
        cboShuffle_data.value = "True"
        tbxShuffle_factor.value = 1
        tbxLw_classifications.value = 5
        cboLw_normalize.value = "True"
        cboLw_differentiate_false_licks.value = "False"
        tbxNum_wells.value = 5
        tbxInitial_timeshift.value = -10000


def tbxModel_pathEx():
    explain.value = "where trained network is saved to"


def txtRaw_data_pathEx():
    explain.value = "where unprocessed session is saved"


def txtDropoutEx():
    explain.value = "dropout of network"


def txtFiltered_data_pathEx():
    explain.value = ""


def txtStrideEx():
    explain.value = "stride [ms]with which output samples are generated (making it smaller than y_slice_size creates overlapping samples"


def txtY_slice_sizeEx():
    explain.value = "size of each output sample in [ms]"


def txtNetwork_typeEx():
    explain.value = "MLP"


def txtEpochsEx():
    explain.value = "how many epochs the network is trained"


def txtNetwork_shapeEx():
    explain.value = "currently not used, originally showed how many samples are in each batch"


def txtFrom_raw_dataEx():
    explain.value = "if True, loads session from raw data path, if False loads session from filtered data path (which is much faster)"


def txtEvaluate_trainingEx():
    explain.value = "if set to True, the network evaluates training performance at runtime (slows down training)"


def txtSearch_radiusEx():
    explain.value = ""


def txtStep_sizeEx():
    explain.value = ""


def txtTime_shift_stepsEx():
    explain.value = "after each step, the network resets with new timeshift and reassigns new ys  depending  on the current timeshift. Determines how many steps are performed at runtime (see time_shift_iter)"


def txtShuffle_dataEx():
    explain.value = " if data is to be shuffled before assigning to training and testing (see shuffle-factor parameter)"


def txtShuffle_factorEx():
    explain.value = " shuffling occurs batch-wise (to minimize overlap  if stride is lower than y_slice_size). Indicates how many samples are to be shuffled in a batch."


def txtTime_shift_iterEx():
    explain.value = "determines, how much the time is shifted after each time_shift_step"


def txtInitial_timeshiftEx():
    explain.value ="initial time_shift for position decoding. For well-decoding,I appropiated this parameter to determine if the previous or next well is to be decoded (sorry). +1 indicates next well, -1 previous well"

def txtMetric_iterEx():
    explain.value ="after how many epochs the network performance is to be evaluated"

def txtBatch_sizeEx():
    explain.value = "how many samples are to be given into the network at once (google batch_size)"

def txtSlice_sizeEx():
    explain.value ="how many [ms] of neural spike data are in each sample"

def txtX_maxEx():
    explain.value ="determines shape of the track the rat is located on [cm]"

def txtY_maxEx():
    explain.value ="determines shape of the track the rat is located on [cm]"

def txtX_minEx():
    explain.value ="determines shape of the track the rat is located on [cm]"

def txtY_minEx():
    explain.value ="determines shape of the track the rat is located on [cm]"

def txtX_stepEx():
    explain.value ="determines, what shape the position bins for the samples are [cm] (3*3cm per bin default)"

def txtY_stepEx():
    explain.value ="determines, what shape the position bins for the samples are [cm] (3*3cm per bin default)"

def txtEarly_stoppingEx():
    explain.value ="if True, checks if network performance degrades after a certain amount of steps (see network) and stops training early if yes"

def txtNaive_testEx():
    explain.value ="if True, doesn't reassign y_values after each time_shift step. Necessary to determine what part of the network performance  is due to similarities between different time_shift step neural data"

def txtValid_ratioEx():
    explain.value ="ratio between training and validation data"

def txtTesting_ratioEx():
    explain.value ="ration between training and testing data"

def txtK_cross_validationEx():
    explain.value ="if more than 1, the data is split into k different sets and results are averaged over all set-performances"

def txtLoad_modelEx():
    explain.value ="if True, the saved model is loaded for training instead of a new one.Is set to True if naive testing is True and time-shift is != 0"

def txtTrain_modelEx():
    explain.value ="if True, the model is not trained during runtime. Is set to True if naive testing is True and time-shift is != 0"

def txtKeep_neuronEx():
    explain.value =""

def txtNeurons_kept_factorEx():
    explain.value ="if less than one, a corresponding fraction of  neurons are randomly removed from session before training"

def txtLw_classificationsEx():
    explain.value ="for well decoding: how many classes exist"

def txtLw_normalizeEx():
    explain.value ="lickwell_data: if True, training and validation data is normalized (complex rules)"

def txtLw_differentiate_false_licksEx():
    explain.value ="Not used anymore due to too small sample sizes and should currently  give an error if True. if True, the network specifically trains to  distinguish between correct and false licks. Should work with minimal code editing if ever necessary to implement"

def txtNum_wellsEx():
    explain.value ="number of lickwells in data set. Really only in object because of lw_differentiate_false_licks, but there  is no reason to remove it either"

def txtMetricEx():
    explain.value ="with what metric the network is to be evaluated (depending on study)"

def txtValid_licksEx():
    explain.value ="List of licks which are valid for well-decoding study"

def txtFilter_tetrodesEx():
    explain.value = "removes tetrodes from raw session data before creating slice object. Useful if some of the tetrodes are e.g. hippocampal and others for pfc"

def txtPhasesEx():
    explain.value ="contains list of training phases"

def txtPhase_change_idsEx():
    explain.value ="contains list of phase change lick_ids"

def txtNumber_of_binsEx():
    explain.value ="number of win sized bins going into the input"

def txtStart_time_by_lick_idEx():
    explain.value ="list of tuples (lick_id,start time) which describe the time at which a lick officially starts relative to the lick time described in the lick object. Defaults to zero but can be changed if a different range is to be observed"

def txtBehavior_component_filterEx():
    explain.value ="filters session data, string"

def txtWin_sizeEx():
    explain.value =""

def notShowEx():
    explain.value = ""

def run_network():
    nd = Net_data(
        model_path = tbxModel_path.value,
        raw_data_path = tbxRaw_data_path.value,
        dropout=float(tbxDropout.value),
        filtered_data_path=tbxFiltered_data_path.value,
        stride=int(tbxStride.value),
        epochs=int(tbxEpochs.value),
        from_raw_data=bool(cboFrom_raw_data.value=="True"),
        evaluate_training=bool(cboEvaluate_training.value=="True"),
        session_filter=Filter(func=hann, search_radius=int(tbxSearch_radius.value), step_size=int(tbxWin_size.value)),
        time_shift_steps=int(tbxTime_shift_steps.value),
        shuffle_data=bool(cboShuffle_data.value=="True"),
        shuffle_factor=int(tbxShuffle_factor.value),
        time_shift_iter=int(tbxTime_shift_iter.value),
        initial_timeshift=int(tbxInitial_timeshift.value),
        metric_iter=int(tbxMetric_iter.value),
        batch_size=int(tbxBatch_size.value),
        slice_size=int(tbxSlice_size.value),
        x_max=int(tbxX_max.value),
        y_max=int(tbxY_max.value),
        x_min=int(tbxX_min.value),
        y_min=int(tbxY_min.value),
        x_step=int(tbxX_step.value),
        y_step=int(tbxY_step.value),
        early_stopping=bool(cboEarly_stopping.value=="True"),
        naive_test=bool(cboNaive_test.value=="True"),
        valid_ratio=float(tbxValid_ratio.value),
        testing_ratio=float(tbxTesting_ratio.value),
        k_cross_validation=int(tbxK_cross_validation.value),
        load_model=bool(cboLoad_model.value=="True"),
        train_model=bool(cboTrain_model.value=="True"),
        keep_neurons=bool(tbxKeep_neuron.value=="True"),
        neurons_kept_factor=float(tbxNeurons_kept_factor.value),
        lw_classifications=int(tbxLw_classifications.value),
        lw_normalize=bool(cboLw_normalize.value=="True"),
        lw_differentiate_false_licks=bool(cboLw_differentiate_false_licks.value=="True"),
        num_wells=int(tbxNum_wells.value),
        metric=tbxMetric.value,
        valid_licks=tbxValid_licks.value,
        filter_tetrodes=bool(tbxFilter_tetrodes.value=="True"),
        phases=tbxPhases.value,
        phase_change_ids=tbxPhase_change_ids.value,
        number_of_bins=int(tbxNumber_of_bins.value=="True"),
        start_time_by_lick_id=int(tbxStart_time_by_lick_id.value=="True"),
        behavior_component_filter=tbxBehavior_component_filter.value,
    )
    if tbxMetric.value == "map":
        session = initiate_network(nd)
        run_network(nd, session)
    else:
        session = initiate_lickwell_network(nd)  # Initialize session
        X, y, nd, session, = lickwells_io(session, nd, excluded_wells=[1], shift=nd.initial_timeshift,
                                          target_is_phase=True,
                                          lickstart=0, lickstop=5000)

    pass

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




# Combo Select one Object
txtObject = Text(app, text="examples", grid=[1, 4])
txtObject.text_size = style.title_Size
txtObject.text_color = style.title_Color
cboObject = Combo(app,
                  options=["lickwell_experiment_pfc_future", "lickwell_experiment_pfc_memory", "position_experiment_pfc", "Object4"],
                  grid=[2, 4], command=select_Parameter)

txtObjectEx = Text(app, text="Todo....", grid=[3, 4])
txtObjectEx.text_color = style.explain_Color
txtObjectEx.text_size = style.explain_Size

# model_path,  raw_data_path, dropout with TextBox Enable = true

txtModel_path = Text(app, text="Model_Path:", grid=[1, 5], align="left")
txtModel_path.text_size = style.parameter_Size
txtModel_path.text_color = style.parameter_Color
tbxModel_path = TextBox(app, grid=[2, 5], align="left", width=30)
txtModel_path.when_mouse_enters = tbxModel_pathEx
txtModel_path.when_mouse_leaves = notShowEx

txtRaw_data_path = Text(app, text="Raw_data_Path:", grid=[1, 6], align="left")
txtRaw_data_path.text_size = style.parameter_Size
txtRaw_data_path.text_color = style.parameter_Color
tbxRaw_data_path = TextBox(app, grid=[2, 6], align="left", width=30)
tbxRaw_data_path.when_mouse_enters = txtRaw_data_pathEx


txtDropout = Text(app, text="dropOut:", grid=[1, 7], align="left")
txtDropout.text_size = style.parameter_Size
txtDropout.text_color = style.parameter_Color
tbxDropout = TextBox(app, grid=[2, 7], align="left", width=30)
tbxDropout.when_mouse_enters= txtDropoutEx

txtFiltered_data_path = Text(app, align="left", text="filtered_data_path:", grid=[1, 8])
txtFiltered_data_path.text_size = style.parameter_Size
txtFiltered_data_path.text_color = style.parameter_Color
tbxFiltered_data_path = TextBox(app, align="left", grid=[2, 8], width=30)
tbxFiltered_data_path.when_mouse_enters = txtFiltered_data_pathEx

txtStride = Text(app, text="stride:", align="left", grid=[1, 9])
txtStride.text_size = style.parameter_Size
txtStride.text_color = style.parameter_Color
tbxStride = TextBox(app, align="left", grid=[2, 9], width=30)
txtStride.when_mouse_enters= txtStrideEx
tbxStride.value = 100


cboObject = Combo(app,
                  options=["at lickwell", "not at lickwell", "correct trials", "incorrect trials","move", "rest"],
                  grid=[2, 4], command=select_Parameter)
# txtY_slice_size = Text(app, text="y_slice_size:", align="left", grid=[1, 10])
# txtY_slice_size.text_size = style.parameter_Size
# txtY_slice_size.text_color = style.parameter_Color
# tbxY_slice_size = TextBox(app, align="left", grid=[2, 10], width=30)
# tbxY_slice_size.when_mouse_enters = txtY_slice_sizeEx
# tbxY_slice_size.value = 100

# txtNetwork_type = Text(app, text="network_type:", align="left", grid=[1, 11])
# txtNetwork_type.text_size = style.parameter_Size
# txtNetwork_type.text_color = style.parameter_Color
# tbxNetwork_type = TextBox(app, align="left", grid=[2, 11], width=30)
# tbxNetwork_type.value = "MLP"
# tbxNetwork_type.when_mouse_enters = txtNetwork_typeEx

txtEpochs = Text(app, text="epochs:", align="left", grid=[1, 12])
txtEpochs.text_size = style.parameter_Size
txtEpochs.text_color = style.parameter_Color
tbxEpochs = TextBox(app, align="left", grid=[2, 12], width=30)
tbxEpochs.value = 20
tbxEpochs.when_mouse_enters = txtEpochsEx
#
# txtNetwork_shape = Text(app, text="network_shape:", align="left", grid=[1, 13])
# txtNetwork_shape.text_size = style.parameter_Size
# txtNetwork_shape.text_color = style.parameter_Color
# tbxNetwork_shape = TextBox(app, align="left", grid=[2, 13], width=30)
# tbxNetwork_shape.value = 10
# tbxNetwork_shape.when_mouse_enters = txtNetwork_shapeEx

txtFrom_raw_data = Text(app, text="from_raw_data:", align="left", grid=[1, 14])
txtFrom_raw_data.text_size = style.parameter_Size
txtFrom_raw_data.text_color = style.parameter_Color
cboFrom_raw_data = Combo(app, options=["True", "False"], align="left", grid=[2, 14], selected="True")
cboFrom_raw_data.when_mouse_enters = txtFrom_raw_dataEx

txtEvaluate_training = Text(app, text="evaluate_training:", align="left", grid=[1, 15])
txtEvaluate_training.text_size = style.parameter_Size
txtEvaluate_training.text_color = style.parameter_Color
cboEvaluate_training = Combo(app, options=["True", "False"], align="left", grid=[2, 15], selected="False")
cboEvaluate_training.when_mouse_enters = txtEvaluate_trainingEx

txtSearch_radius = Text(app, text="search_radius:", align="left", grid=[1, 16])
txtSearch_radius.text_size = style.parameter_Size
txtSearch_radius.text_color = style.parameter_Color
tbxSearch_radius = TextBox(app, align="left", grid=[2, 16], width=30)
tbxSearch_radius.value = 200
tbxSearch_radius.when_mouse_enters= txtSearch_radiusEx

# txtStep_size = Text(app, text="step_size:", align="left", grid=[1, 17])
# txtStep_size.text_size = style.parameter_Size
# txtStep_size.text_color = style.parameter_Color
# tbxStep_size = TextBox(app, align="left", grid=[2, 17], width=30)
# tbxStep_size.value = 100
# tbxStep_size.when_mouse_enters= txtStep_sizeEx

txtTime_shift_steps = Text(app, text="time_shift_steps:", align="left", grid=[1, 18])
txtTime_shift_steps.text_size = style.parameter_Size
txtTime_shift_steps.text_color = style.parameter_Color
tbxTime_shift_steps = TextBox(app, align="left", grid=[2, 18], width=30)
tbxTime_shift_steps.value = 1
tbxTime_shift_steps.when_mouse_enters= txtTime_shift_stepsEx

txtShuffle_data = Text(app, text="shuffle_data:", align="left", grid=[1, 19])
txtShuffle_data.text_size = style.parameter_Size
txtShuffle_data.text_color = style.parameter_Color
cboShuffle_data = Combo(app, options=["True", "False"], align="left", grid=[2, 19], selected="True")
cboShuffle_data.when_mouse_enters = txtShuffle_dataEx



txtShuffle_factor = Text(app, text="shuffle_factor:", align="left", grid=[1, 20])
txtShuffle_factor.text_size = style.parameter_Size
txtShuffle_factor.text_color = style.parameter_Color
tbxShuffle_factor = TextBox(app, align="left", grid=[2, 20], width=30)
tbxShuffle_factor.value = 500
tbxShuffle_factor.when_mouse_enters = txtShuffle_factorEx

txtTime_shift_iter = Text(app, text="time_shift_iter:", align="left", grid=[1, 21])
txtTime_shift_iter.text_size = style.parameter_Size
txtTime_shift_iter.text_color = style.parameter_Color
tbxTime_shift_iter = TextBox(app, align="left", grid=[2, 21], width=30)
tbxTime_shift_iter.value = 500
tbxTime_shift_iter.when_mouse_enters = txtTime_shift_iterEx

txtInitial_timeshift = Text(app, text="initial_timeshift:", align="left", grid=[1, 22])
txtInitial_timeshift.text_size = style.parameter_Size
txtInitial_timeshift.text_color = style.parameter_Color
tbxInitial_timeshift = TextBox(app, align="left", grid=[2, 22], width=30)
tbxInitial_timeshift.value = 0
tbxInitial_timeshift.when_mouse_enters= txtInitial_timeshiftEx

txtMetric_iter = Text(app, text="metric_iter:", align="left", grid=[1, 23])
txtMetric_iter.text_size = style.parameter_Size
tbxMetric_iter = TextBox(app, align="left", grid=[2, 23], width=30)
tbxMetric_iter.value = 1
tbxMetric_iter.when_mouse_enters= txtMetric_iterEx

txtBatch_size = Text(app, text="batch_size:", align="left", grid=[1, 24])
txtBatch_size.text_size = style.parameter_Size
txtBatch_size.text_color = style.parameter_Color
tbxBatch_size = TextBox(app, align="left", grid=[2, 24], width=30)
tbxBatch_size.value = 50
tbxBatch_size.when_mouse_enters= txtBatch_sizeEx

txtSlice_size = Text(app, text="slice_size:", align="left", grid=[1, 25])
txtSlice_size.text_size = style.parameter_Size
txtSlice_size.text_color = style.parameter_Color
tbxSlice_size = TextBox(app, align="left", grid=[2, 25], width=30)
tbxSlice_size.value = 5000
tbxSlice_size.when_mouse_enters = txtSlice_sizeEx

txtX_max = Text(app, text="x_max:", align="left", grid=[1, 26])
txtX_max.text_size = style.parameter_Size
txtX_max.text_color = style.parameter_Color
tbxX_max = TextBox(app, align="left", grid=[2, 26], width=30)
tbxX_max.value = 240
tbxX_max.when_mouse_enters= txtX_maxEx

txtY_max = Text(app, text="y_max:", align="left", grid=[1, 27])
txtY_max.text_size = style.parameter_Size
txtY_max.text_color = style.parameter_Color
tbxY_max = TextBox(app, align="left", grid=[2, 27], width=30)
tbxY_max.value = 190
tbxY_max.when_mouse_enters=txtY_maxEx

txtX_min = Text(app, text="x_min:", align="left", grid=[3, 5])
txtX_min.text_size = style.parameter_Size
txtX_min.text_color = style.parameter_Color
tbxX_min = TextBox(app, align="left", grid=[4, 5], width=30)
tbxX_min.value = 0
tbxX_min.when_mouse_enters= txtX_minEx

txtY_min = Text(app, text="y_min:", align="left", grid=[3, 6])
txtY_min.text_size = style.parameter_Size
txtY_min.text_color = style.parameter_Color
tbxY_min = TextBox(app, align="left", grid=[4, 6], width=30)
tbxY_min.value = 100
tbxY_min.when_mouse_enters= txtY_minEx

txtX_step = Text(app, text="x_step:", align="left", grid=[3, 7])
txtX_step.text_size = style.parameter_Size
txtX_step.text_color = style.parameter_Color
tbxX_step = TextBox(app, align="left", grid=[4, 7], width=30)
tbxX_step.value = 3
tbxX_step.when_mouse_enters= txtX_stepEx

txtY_step = Text(app, text="y_step:", align="left", grid=[3, 8])
txtY_step.text_size = style.parameter_Size
txtY_step.text_color = style.parameter_Color
tbxY_step = TextBox(app, align="left", grid=[4, 8], width=30)
tbxY_max.value = 3
tbxY_max.when_mouse_enters= txtY_stepEx

txtEarly_stopping = Text(app, text="early_stopping:", align="left", grid=[3, 9])
txtEarly_stopping.text_size = style.parameter_Size
txtEarly_stopping.text_color = style.parameter_Color
cboEarly_stopping = Combo(app, options=["True", "False"], align="left", grid=[4, 9], selected="False")
cboEarly_stopping.when_mouse_enters= txtEarly_stoppingEx

txtNaive_test = Text(app, text="naive_test:", align="left", grid=[3, 10])
txtNaive_test.text_size = style.parameter_Size
txtNaive_test.text_color = style.parameter_Color
cboNaive_test = Combo(app, options=["True", "False"], align="left", grid=[4, 10], selected="False")
cboNaive_test.when_mouse_enters= txtNaive_testEx

txtValid_ratio = Text(app, text="valid_ratio:", align="left", grid=[3, 11])
txtValid_ratio.text_size = style.parameter_Size
txtValid_ratio.text_color = style.parameter_Color
tbxValid_ratio = TextBox(app, align="left", grid=[4, 11], width=30)
tbxValid_ratio.value = 0.1
tbxValid_ratio.when_mouse_enters=txtValid_ratioEx

txtTesting_ratio = Text(app, text="testing_ratio:", align="left", grid=[3, 12])
txtTesting_ratio.text_size = style.parameter_Size
txtTesting_ratio.text_color = style.parameter_Color
tbxTesting_ratio = TextBox(app, align="left", grid=[4, 12], width=30)
tbxTesting_ratio.value = 0.1
tbxTesting_ratio.when_mouse_enters= txtTesting_ratioEx

txtK_cross_validation = Text(app, text="k_cross_validation:", align="left", grid=[3, 13])
txtK_cross_validation.text_size = style.parameter_Size
txtK_cross_validation.text_color = style.parameter_Color
tbxK_cross_validation = TextBox(app, align="left", grid=[4, 13], width=30)
tbxK_cross_validation.value = 1
tbxK_cross_validation.when_mouse_enters= txtK_cross_validationEx

txtLoad_model = Text(app, text="load_model:", align="left", grid=[3, 14])
txtLoad_model.text_size = style.parameter_Size
txtLoad_model.text_color = style.parameter_Color
cboLoad_model = Combo(app, options=["True", "False"], align="left", grid=[4, 14], selected="False")
cboLoad_model.when_mouse_enters= txtLoad_modelEx

txtTrain_model = Text(app, text="train_model:", align="left", grid=[3, 15])
txtTrain_model.text_size = style.parameter_Size
txtTrain_model.text_color = style.parameter_Color
cboTrain_model = Combo(app, options=["True", "False"], align="left", grid=[4, 15], selected="True")
cboTrain_model.when_mouse_enters= txtTrain_modelEx

txtKeep_neuron = Text(app, text="keep_neuron:", align="left", grid=[3, 16])
txtKeep_neuron.text_size = style.parameter_Size
txtKeep_neuron.text_color = style.parameter_Color
tbxKeep_neuron = TextBox(app, align="left", grid=[4, 16], width=30)
tbxKeep_neuron.value = -1
tbxKeep_neuron.when_mouse_enters= txtKeep_neuronEx

txtNeurons_kept_factor = Text(app, text="neurons_kept_factor:", align="left", grid=[3, 17])
txtNeurons_kept_factor.text_size = style.parameter_Size
txtNeurons_kept_factor.text_color = style.parameter_Color
tbxNeurons_kept_factor = TextBox(app, align="left", grid=[4, 17], width=30)
tbxNeurons_kept_factor.value = 1.0
tbxNeurons_kept_factor.when_mouse_enters= txtNeurons_kept_factorEx

txtLw_classifications = Text(app, text="lw_classifications:", align="left", grid=[3, 18])
txtLw_classifications.text_size = style.parameter_Size
txtLw_classifications.text_color = style.parameter_Color
tbxLw_classifications = TextBox(app, align="left", grid=[4, 18], width=30)
tbxLw_classifications.value = None
tbxLw_classifications.when_mouse_enters = txtLw_classificationsEx

txtLw_normalize = Text(app, text="lw_normalize:", align="left", grid=[3, 19])
txtLw_normalize.text_size = style.parameter_Size
txtLw_normalize.text_color = style.parameter_Color
cboLw_normalize = Combo(app, options=["True", "False"], align="left", grid=[4, 19], selected="False")
cboLw_normalize.when_mouse_enters= txtLw_normalizeEx

txtLw_differentiate_false_licks = Text(app, text="lw_differentiate_false_licks:", align="left", grid=[3, 20])
txtLw_differentiate_false_licks.text_size = style.parameter_Size
txtLw_differentiate_false_licks.text_color = style.parameter_Color
cboLw_differentiate_false_licks = Combo(app, options=["True", "False"], align="left", grid=[4, 20], selected="False")
cboLw_differentiate_false_licks.when_mouse_enters= txtLw_differentiate_false_licksEx

txtNum_wells = Text(app, text="num_wells:", align="left", grid=[3, 21])
txtNum_wells.text_size = style.parameter_Size
txtNum_wells.text_color = style.parameter_Color
tbxNum_wells = TextBox(app, align="left", grid=[4, 21], width=30)
tbxNum_wells.value = 5
tbxNum_wells.when_mouse_enters= txtNum_wellsEx

txtMetric = Text(app, text="metric:", align="left", grid=[3, 22])
txtMetric.text_size = style.parameter_Size
txtMetric.text_color = style.parameter_Color
tbxMetric = TextBox(app, align="left", grid=[4, 22], width=30)
tbxMetric.value = "map"
tbxMetric.when_mouse_enters= txtMetricEx

txtValid_licks = Text(app, text="valid_licks:", align="left", grid=[3, 23])
txtValid_licks.text_size = style.parameter_Size
txtValid_licks.text_color = style.parameter_Color
tbxValid_licks = TextBox(app, align="left", grid=[4, 23], width=30)
tbxValid_licks.value = None
tbxValid_licks.when_mouse_enters= txtValid_licksEx

txtFilter_tetrodes = Text(app, text="filter_tetrodes:", align="left", grid=[3, 24])
txtFilter_tetrodes.text_size = style.parameter_Size
txtFilter_tetrodes.text_color = style.parameter_Color
tbxFilter_tetrodes = TextBox(app, align="left", grid=[4, 24], width=30)
tbxFilter_tetrodes.value = None
tbxFilter_tetrodes.when_mouse_enters= txtFilter_tetrodesEx

txtPhases = Text(app, text="phases:", align="left", grid=[3, 26])
txtPhases.text_size = style.parameter_Size
txtPhases.text_color = style.parameter_Color
tbxPhases = TextBox(app, align="left", grid=[4, 26], width=30)
tbxPhases.value = None
tbxPhases.when_mouse_enters= txtPhasesEx

txtPhase_change_ids = Text(app, text="phase_change_ids:", align="left", grid=[3, 27])
txtPhase_change_ids.text_size = style.parameter_Size
txtPhase_change_ids.text_color = style.parameter_Color
tbxPhase_change_ids = TextBox(app, align="left", grid=[4, 27], width=30)
tbxPhase_change_ids.value = None
tbxPhase_change_ids.when_mouse_enters= txtPhase_change_idsEx

txtNumber_of_bins = Text(app, text="number_of_bins:", align="left", grid=[3, 28])
txtNumber_of_bins.text_size = style.parameter_Size
txtNumber_of_bins.text_color = style.parameter_Color
tbxNumber_of_bins = TextBox(app, align="left", grid=[4, 28], width=30)
tbxNumber_of_bins.value = 10
tbxNumber_of_bins.when_mouse_enters= txtNumber_of_binsEx

txtStart_time_by_lick_id = Text(app, text="start_time_by_lick_id:", align="left", grid=[3, 29])
txtStart_time_by_lick_id.text_size = style.parameter_Size
txtStart_time_by_lick_id.text_color = style.parameter_Color
tbxStart_time_by_lick_id = TextBox(app, align="left", grid=[4, 29], width=30)
tbxStart_time_by_lick_id.value = None
tbxStart_time_by_lick_id.when_mouse_enters= txtStart_time_by_lick_idEx

txtBehavior_component_filter = Text(app, text="behavior_component_filter:", align="left", grid=[3, 30])
txtBehavior_component_filter.text_size = style.parameter_Size
txtBehavior_component_filter.text_color = style.parameter_Color
tbxBehavior_component_filter = TextBox(app, align="left", grid=[4, 30], width=30)
tbxBehavior_component_filter.value = None
tbxBehavior_component_filter.when_mouse_enters= txtBehavior_component_filterEx

txtWin_size = Text(app, text="behavior_component_filter:", align="left", grid=[3, 31])
txtWin_size.text_size = style.parameter_Size
txtWin_size.text_color = style.parameter_Color
tbxWin_size = TextBox(app, align="left", grid=[4, 31], width=30)
tbxWin_size.value = 100
tbxWin_size.when_mouse_enters = txtWin_sizeEx
button = PushButton(app, grid=[4,32],command=run_network,text="execute")
app1.display()
