from appJar import gui
import tmp.sample_experiments as se
from src.database_api_beta import Slice, Filter, hann, Net_data

from src.preprocessing import lickwells_io
from src.network_functions import run_network_process, initiate_lickwell_network, run_lickwell_network
from src.metrics import print_metric_details,print_lickwell_metrics


def press(button):
    if button == "Cancel":
        app.stop()
    else:
        option = app.getOptionBox("menu")
        if option == "Future lickwell prediction hippocampus":
            experiment = se.lickwell_experiment_pfc_future
        if option == "Future lickwell prediction prefrontal cortex":
            experiment = se.lickwell_experiment_pfc_memory
        if option == "Last lickwell prediction hippocampus":
            experiment = se.lickwell_experiment_hc_future
        if option == "Last lickwell prediction prefrontal cortex":
            experiment = se.lickwell_experiment_hc_memory


        if app.getCheckBox("preprocessed session") is True:
            experiment.nd.session_from_raw = False
        else:
            experiment.nd.session_from_raw = True
        if experiment.study == "lickwell":
            nd = experiment.nd
            session = initiate_lickwell_network(experiment.nd)  # Initialize session
            X, y, metadata, nd = lickwells_io(session, nd, excluded_wells=[1], shift=nd.initial_timeshift,
                                              normalize=nd.lw_normalize,
                                              differentiate_false_licks=nd.lw_differentiate_false_licks)

            run_lickwell_network(nd, session, X, y, metadata)
            path = nd.model_path + "output/"
            print_metric_details(path, nd.initial_timeshift)

if __name__ == '__main__':

    options = ["Future lickwell prediction hippocampus",
               "Future lickwell prediction prefrontal cortex",
                "Last lickwell prediction hippocampus",
               "Last lickwell prediction prefrontal cortex"




               ]

    app = gui()
    app.addLabel("title", "Sample experiments")
    app.setLabelBg("title", "red")
    app.addLabelOptionBox("menu", options)
    app.addCheckBox("preprocessed session")
    app.addButtons(["Run program", "Cancel"], press)
    app.go()