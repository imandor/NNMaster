



class Network_Paths():
    
    def __init__(self, model_path,filtered_data_path,raw_data_path,filter_tetrodes=None,switch_x_y=False):
        self.model_path=model_path
        self.raw_data_path=raw_data_path
        self.filtered_data_path=filtered_data_path
        self.filter_tetrodes=filter_tetrodes
        self.switch_x_y=switch_x_y
        
        
        
# prefrontal cortex
pfc_dmf = Network_Paths(
    model_path="G:/master_datafiles/trained_networks/pfc/",
    raw_data_path="G:/master_datafiles/raw_data/PFC/",
    filtered_data_path = "session_pfc",
)

# hippocampus
#
hc_dmf = Network_Paths(
    model_path="G:/master_datafiles/trained_networks/hc/",
    raw_data_path="G:/master_datafiles/raw_data/HC/",
    filtered_data_path = "session_hc",
)


# Combination data sets
c_dmf = Network_Paths(
    model_path="G:/master_datafiles/trained_networks/c/",
    raw_data_path="G:/master_datafiles/raw_data/C/",
    filtered_data_path = "session_hc",
    switch_x_y=True
)

chc_dmf = Network_Paths(
    model_path="G:/master_datafiles/trained_networks/chc/",
    raw_data_path="G:/master_datafiles/raw_data/C/",
    filtered_data_path = "session_chc",
    switch_x_y=True,
    filter_tetrodes=range(13,28)
)

cpfc_dmf = Network_Paths(
    model_path="G:/master_datafiles/trained_networks/cpfc/",
    raw_data_path="G:/master_datafiles/raw_data/C/",
    filtered_data_path = "session_cpfc",
    switch_x_y=True,
    filter_tetrodes = range(0, 13)
)

pfc_lw = Network_Paths(
    model_path="G:/master_datafiles/trained_networks/pfc_lw/",
    raw_data_path= "G:/master_datafiles/raw_data/PFC/",
    filtered_data_path = "session_pfc_lw.pkl",
)

hc_lw = Network_Paths(
    model_path="G:/master_datafiles/trained_networks/hc_lw/",
    raw_data_path= "G:/master_datafiles/raw_data/HC/",
    filtered_data_path = "session_hc_lw.pkl",
)
