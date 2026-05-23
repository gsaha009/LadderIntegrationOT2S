# Splitting Ladder ROOT file to separate Optical Groups
# Author : Gourab Saha, IPHC

# References:
#  https://gitlab.cern.ch/otsdaq/potatoconverters/-/blob/master/Histogrammer.py?ref_type=heads (PotatoConverter)
#  ChatGPT (OpenAI)

# How to Run:
# python main.py -c Configs_split_ladder_root_file/file.yaml -s (if splitting is required) -og <-1: default -- all OGs>


import os
import re
import sys
import yaml
import ROOT
import time
import numpy as np
import pandas as pd
import argparse
from datetime import datetime



def load_cfg(cfg):
    cfgdict = None
    with open(cfg, 'r') as f:
        cfgdict = yaml.safe_load(f)
    return cfgdict


def copy_dir(src_dir, dest_dir, skipdir = False):
    for key in src_dir.GetListOfKeys():
        name = key.GetName()
        obj = key.ReadObj()

        if obj.InheritsFrom("TDirectory"):
            if skipdir == True:
                continue

            dest_dir.mkdir(name)
            subdir = dest_dir.GetDirectory(name)
            copy_dir(obj, subdir, skipdir = skipdir)
        else:
            dest_dir.cd()
            obj.Write(name, ROOT.TObject.kOverwrite)

                
            
def add_modID(TDir, modID, OG):
    ID_obj = ROOT.TObjString(modID)
    TDir.Delete(f"D_B(0)_NameId_OpticalGroup({OG});*")
    TDir.WriteObject(ID_obj, f"D_B(0)_NameId_OpticalGroup({OG})")


# ------------------------------------------------------------ #
#                    Histogrammer Class                        #
# ------------------------------------------------------------ #
TIME_FORMAT_1 = '%Y-%m-%d %H:%M:%S'
TIME_FORMAT_2 = '%d.%m.%Y %H:%M:%S'
convert_timestamp_to_float = lambda timestamp,timeformat: datetime.strptime(timestamp, timeformat).timestamp()
convert_timestamps_to_float = lambda timestamps,timeformat: [convert_timestamp_to_float(ts, timeformat) for ts in timestamps]


class Histogrammer():
    def __init__(self):
        self.markerStyle  = 20
        self.markerSize   = 0.6
        self.markerSizeIV = 1
        self.lineWidth    = 2
        self.timeDivision = 503

        self.GRAPHS_EXTENSION_TIME = 300 #5 minutes before test starts


    def setMonitorGraphStyle(self, graph):
        graph.SetMarkerStyle(self.markerStyle)
        graph.SetMarkerSize(self.markerSize)
        graph.SetLineColor(ROOT.kBlue)
        graph.SetLineWidth(self.lineWidth)
        graph.SetLineStyle(ROOT.kSolid)
        # Configure x-axis to display time correctly
        graph.GetXaxis().SetTimeDisplay(1)
        graph.GetXaxis().SetNdivisions(self.timeDivision)
        graph.GetXaxis().SetTimeFormat(TIME_FORMAT_1)
        graph.GetXaxis().SetTimeOffset(0)


    def setIVGraphStyle(self, graph, color=ROOT.kBlue):
        graph.SetMarkerStyle(self.markerStyle)
        graph.SetMarkerSize(self.markerSizeIV)
        graph.SetLineColor(color)
        graph.SetLineWidth(self.lineWidth)
        graph.SetLineStyle(ROOT.kSolid)
        

    def makeGraph(self, x, y, max_points=None):
        x = np.array(x, dtype='float64')
        y = np.array(y, dtype='float64')

        if x.size == 0:
            raise Exception('There are no values to make a Graph between the start and stop test times!')
        
        return ROOT.TGraph(x.size, x, y)

    
    def makeMonitorLVCurrent(self, timestamps, values, moduleID=None):
        timestamps = convert_timestamps_to_float(timestamps,TIME_FORMAT_1)
        graph = self.makeGraph(timestamps, values)
        graph.SetName("LV_Current")
        graph.SetTitle(f"LV Current - {moduleID};Local Time;Current [A]")
        self.setMonitorGraphStyle(graph)
        return graph
        
    def makeMonitorLVVoltage(self, timestamps, values, moduleID=None):
        timestamps = convert_timestamps_to_float(timestamps,TIME_FORMAT_1)
        graph = self.makeGraph(timestamps, values)
        graph.SetName("LV_Voltage")
        graph.SetTitle(f"LV Voltage - {moduleID};Local Time;Voltage [V]")
        self.setMonitorGraphStyle(graph)
        return graph

    def makeMonitorHVCurrent(self, timestamps, values, moduleID=None):
        if np.sum(np.array(values, dtype='float64') < 0) > np.sum(np.array(values, dtype='float64') > 0):
            values = -values
        timestamps = convert_timestamps_to_float(timestamps,TIME_FORMAT_1)
        graph = self.makeGraph(timestamps, values)
        graph.SetName("HV_Current")
        graph.SetTitle(f"HV Current - {moduleID};Local Time;Current [-µA]")
        self.setMonitorGraphStyle(graph)
        return graph

    def makeMonitorHVVoltage(self, timestamps, values, moduleID=None):
        if np.sum(np.array(values, dtype='float64') < 0) > np.sum(np.array(values, dtype='float64') > 0):
            values = -values
        timestamps = convert_timestamps_to_float(timestamps,TIME_FORMAT_1)
        graph = self.makeGraph(timestamps, values)
        graph.SetName("HV_Voltage")
        graph.SetTitle(f"HV Voltage - {moduleID};Local Time;Voltage [-V]")
        self.setMonitorGraphStyle(graph)
        return graph

    def makeMonitorEnvTemp(self, timestamps, values, moduleID=None):
        timestamps = convert_timestamps_to_float(timestamps,TIME_FORMAT_1)
        graph = self.makeGraph(timestamps, values)
        graph.SetName("ENV_Temperature")
        graph.SetTitle(f"Environment Temperature - {moduleID};Local Time;Temperature [#circC]")
        self.setMonitorGraphStyle(graph)
        return graph

    def makeMonitorEnvHumidity(self, timestamps, values, moduleID=None):
        timestamps = convert_timestamps_to_float(timestamps,TIME_FORMAT_1)
        graph = self.makeGraph(timestamps, values)
        graph.SetName("ENV_Humidity")
        graph.SetTitle(f"Environment Humidity - {moduleID};Local Time;Relative Humidity [%]")
        self.setMonitorGraphStyle(graph)
        return graph

    def makeMonitorEnvDewP(self, timestamps, values, moduleID=None):
        timestamps = convert_timestamps_to_float(timestamps,TIME_FORMAT_1)
        graph = self.makeGraph(timestamps, values)
        graph.SetName("ENV_Dewpoint")
        graph.SetTitle(f"Environment Dew Point - {moduleID};Local Time;Dew Point [#circC]")
        self.setMonitorGraphStyle(graph)
        return graph

    def makeMonitorCarrTemp(self, timestamps, values, moduleID=None):
        timestamps = convert_timestamps_to_float(timestamps,TIME_FORMAT_1)
        graph = self.makeGraph(timestamps, values)
        graph.SetName("CARRIER_Temperature")
        graph.SetTitle(f"Carrier Temperature - {moduleID};Local Time;Temperature [#circC]")
        self.setMonitorGraphStyle(graph)
        return graph

    def makeMonitorChilTemp(self, timestamps, values, moduleID=None):
        timestamps = convert_timestamps_to_float(timestamps,TIME_FORMAT_1)
        graph = self.makeGraph(timestamps, values)
        graph.SetName("CHILLER_Set_Temperature")
        graph.SetTitle(f"Chiller Set Point Temperature - {moduleID};Local Time;Temperature [#circC]")
        self.setMonitorGraphStyle(graph)
        return graph

    
    def makeIVCurve(self, voltages, currents, moduleID=None):
        if np.sum(voltages < 0) > np.sum(voltages > 0):
            voltages = -voltages

        if np.sum(currents < 0) > np.sum(currents > 0):
            currents = -currents

        graph = ROOT.TGraph(len(voltages), voltages, currents)
        graph.SetName("IV_Current");
        graph.SetTitle(f"IV Curve - {moduleID};Voltage [-V];Current [-uA]")
        self.setIVGraphStyle(graph)
        return graph


    
# ------------------------------------------------------------ #
#                        Monitor Class                         #
# ------------------------------------------------------------ #

class Monitor:
    def __init__(self, file = None, nHeaders=0):
        self.file = file
        self.nHeaders = nHeaders # 0: default, None: keep all
    
    def loadcsv(self):
        return pd.read_csv(self.file, header=self.nHeaders)

    
class MonitorPS(Monitor):
    def __init__(self, csvfile = None, ps_channels = None):
        super().__init__(file = csvfile, nHeaders=None)
        self.PS_channels = ps_channels
        
    def getpattern(self):
        pattern = (
            r"(?:TimeStamp:(?P<timestamp>\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2}:\d{2}),)?"
            r"\s*CAEN-SY5527_(?P<vtype>\w+)#(?P<channel>\d+)\s+"
            r"Voltage:(?P<voltage>[\d.]+)\s+"
            r"Current:(?P<current>[\d.]+)"
        )
        rh_pattern   = r"RH\s*:\s*(?P<RH>[-+]?\d+(?:\.\d+)?)"
        temp_pattern = r"TEMP\s*:\s*(?P<TEMP>[-+]?\d+(?:\.\d+)?)"
        dewp_pattern = r"DEWP\s*:\s*(?P<DEWP>[-+]?\d+(?:\.\d+)?)"

        return pattern,rh_pattern,temp_pattern,dewp_pattern

    
    def extract(self):
        df = self.loadcsv()
        pattern,rh_pattern,temp_pattern,dewp_pattern = self.getpattern()
        out_dict = {
            f'OpticalGroup_{i-1}' : {
                'HV': {
                    'timestamp':[],
                    'voltage': [],
                    'current': []
                },
                'LV': {
                    'timestamp':[],
                    'voltage': [],
                    'current': []
                },
            } for i in self.PS_channels
        }
        env_dict = {
            'timestamp' : [],
            'relhumidity': [],
            'temperature': [],
            'dewpoint': []
        }
        
        last_timestamp = None
        
        for _, row in df.iterrows():
            s = ",".join("" if pd.isna(x) else str(x) for x in row)

            match = re.search(pattern, s)
            if match:
                if match.group("timestamp") is not None:
                    last_timestamp = match.group("timestamp")
                
                vtype = match.group("vtype")
                channel = int(match.group("channel"))
        
                if channel > int(self.PS_channels[-1]) :
                    continue

                voltage = float(match.group("voltage"))
                current = float(match.group("current"))
                
                out_dict[f'OpticalGroup_{channel-1}'][vtype]['timestamp'].append(last_timestamp)
                out_dict[f'OpticalGroup_{channel-1}'][vtype]['voltage'].append(voltage)
                out_dict[f'OpticalGroup_{channel-1}'][vtype]['current'].append(current)

            rh_match = re.search(rh_pattern, s)
            if rh_match:
                rh = float(rh_match.group("RH"))
                env_dict['timestamp'].append(last_timestamp)
                env_dict['relhumidity'].append(rh)

            temp_match = re.search(temp_pattern, s)
            if temp_match:
                temp = float(temp_match.group("TEMP"))
                env_dict['temperature'].append(temp)

            dewp_match = re.search(dewp_pattern, s)
            if dewp_match:
                dewp = float(dewp_match.group("DEWP"))
                env_dict['dewpoint'].append(dewp)
                            
        return out_dict,env_dict

    
#class MonitorEnv(Monitor):
#    def __init__(self, csvfile = None):
#        super().__init__(file = csvfile)
#            
#    def extract(self):
#        df = self.loadcsv()
#        cols = {c.lower().strip(): c for c in df.columns}
#        # Identify columns
#        time_col = next((cols[c] for c in cols if "time" in c), None)
#        temp_col = next((cols[c] for c in cols if "temp" in c), None)
#        hum_col  = next((cols[c] for c in cols if "humid" in c), None)
#
#        if not time_col or not temp_col or not hum_col:
#            raise RuntimeError(f"Missing required columns in {df.columns}")
#
#        timestamp   = pd.to_datetime(df[time_col],
#                                     format="%d/%m/%Y %H:%M:%S").dt.strftime("%Y-%m-%d %H:%M:%S").to_numpy()
#        temperature = df[temp_col].astype(float).to_numpy()
#        humidity    = df[hum_col].astype(float).to_numpy()
#
#        return {
#            "timestamp": timestamp,
#            "temperature": temperature,
#            "humidity": humidity,
#        }
        
    
class MonitorIV(Monitor):
    def __init__(self, csvfile = None):
        super().__init__(file = csvfile)

    def extract(self):
        df = self.loadcsv()
        V = df['Voltage'].astype(float).to_numpy() 
        I = df['Current'].astype(float).to_numpy() 
        
        return {
            'voltage': V,
            'current': I
        }


# Helpers
def write_string(rdir, name, value):
    rdir.WriteObject(ROOT.TObjString(f"{value}"), name)

    
def read_TGraph(gr):
    N = gr.GetN()
    x = gr.GetX()
    y = gr.GetY()
    data = [[x[i], y[i]] for i in range(N)]
    return data

def get_first_entry_from_TGraph(gr):
    data = np.array(read_TGraph(gr))[:,1]
    return float(data[0])

def get_last_entry_from_TGraph(gr):
    data = np.array(read_TGraph(gr))[:,1]
    return float(data[-1])


def get_timestamps_at_configuration(logf):
    cmd = rf"""awk '/\|I\| Initialized/ {{t=$1" "$2}} END{{print t}}' "{logf}" """
    timestamp_unconf = os.popen(cmd).read().strip().rstrip(":")
    cmd = rf"""awk '/\|I\| Configured/ {{t=$1" "$2}} END{{print t}}' "{logf}" """
    timestamp_conf = os.popen(cmd).read().strip().rstrip(":")

    return timestamp_unconf,timestamp_conf
    

def get_idx_at_configuration(PS_ts, logf):
    # Search for Initialized in the log to get the LV current while unconfigured
    # Search for Configured in the log to get the LV current while configured
    ts_unconf,ts_conf = get_timestamps_at_configuration(logf)
    # these are the TSs
    # Now check in PS_ts for the respective closest time stamps
    # if not exactly equal, take the one not less than the logf
    ts_unconf_float = convert_timestamp_to_float(ts_unconf, TIME_FORMAT_2)
    ts_conf_float = convert_timestamp_to_float(ts_conf, TIME_FORMAT_2)
    
    PS_ts_float = np.array(convert_timestamps_to_float(PS_ts, TIME_FORMAT_1))
    PS_ts_index = np.arange(PS_ts_float.shape[0])

    mask_unconf = PS_ts_float >= ts_unconf_float
    mask_conf   = PS_ts_float >= ts_conf_float

    PS_ts_unconf_idx = None
    PS_ts_conf_idx   = None
    
    if np.sum(mask_unconf) > 0:
        PS_ts_unconf_idx = PS_ts_index[mask_unconf][0]
    else:
        print(f"W A R N I N G : time stamp of module unconfigured from Ph2ACF log is out of range of the PS time stamps --> UNUSUAL")
        print("setting it to the 1st index : dummy")
        PS_ts_unconf_idx = PS_ts_index[0]

    if np.sum(mask_conf) > 0:
        PS_ts_conf_idx = PS_ts_index[mask_conf][0]
    else:
        print(f"W A R N I N G : time stamp of module configured from Ph2ACF log is out of range of the PS time stamps --> UNUSUAL")
        print("setting it to the 1st index : dummy")
        PS_ts_conf_idx = PS_ts_index[0]
        
    return PS_ts_unconf_idx, PS_ts_conf_idx
    

# +++++++++++++++++++++++++++++++++++++++++++++++++++++++ #
# ooooooooooooooooooooo Main Function ooooooooooooooooooo #
# +++++++++++++++++++++++++++++++++++++++++++++++++++++++ #

def main(args):

    REAL_START = time.perf_counter()
    CPU_START  = time.time()
    
    print(f"\nLadder --> Module level splitting ===>> {args.split}\n")
    if args.split == False:
        print("Just to add ModuleIDs per OpticalGroup")
        print("Use -s in the cmdline to enable splitting\n")
    
    CFG_FILE = args.config
    CFG = load_cfg(CFG_FILE)

    OUTDIR = CFG.get('OUTPUT_PATH')
    if OUTDIR is not None:
        if not os.path.exists(OUTDIR):
            os.mkdir(OUTDIR)

    MODULE_POS = CFG.get("MODULE_POS")
    
    ladID    = CFG.get('LADDER')
    coolTemp = CFG.get('COOLING_TEMP')


    # Open input file in Read mode
    infilename = CFG.get("INPUT_FILE")
    infile = ROOT.TFile(infilename, "READ")
    print(f"Ph2ACF Ladder Output file as the input here : {infilename}\n")

    TDir_Infile_Det = infile.Get("Detector")
    
    test_type      = TDir_Infile_Det.GetKey("CalibrationName_Detector").ReadObj()
    ph2acf_version = TDir_Infile_Det.GetKey('GitTag_Detector').ReadObj()
    ph2acf_commit  = TDir_Infile_Det.GetKey('GitCommitHash_Detector').ReadObj()
    host_pc        = f"{TDir_Infile_Det.GetKey('Username_Detector').ReadObj()}@{TDir_Infile_Det.GetKey('HostName_Detector').ReadObj()}"
    start_time     = f"{TDir_Infile_Det.GetKey('CalibrationStartTimestamp_Detector').ReadObj()}"
    fmt_start_time = datetime.strptime(start_time, "%Y-%m-%d %H:%M:%S").strftime("%Y-%m-%d_%Hh%Mm%Ss")
    stop_time      = f"{TDir_Infile_Det.GetKey('CalibrationStopTimestamp_Detector').ReadObj()}"

    TDir_Infile_Brd= infile.Get("Detector/Board_0")
    board_ip       = TDir_Infile_Brd.GetKey('D_NameId_Board_(0)').ReadObj()
    
    
    print(f"Calibration     : {test_type}")    
    print(f"Ph2-ACF version : {ph2acf_version}")
    print(f"Ph2-ACF commit  : {ph2acf_commit}")
    print(f"Host PC         : {host_pc}")
    print(f"START           : {start_time}")
    print(f"STOP            : {stop_time}\n")
    print(f"Board IP        : {board_ip}")
    print(f"Ladder ID       : {ladID}")
    print(f"Cooling Temp    : {coolTemp}\n")


    
    # Creating an output file in Write mode and then copy the contents
    outfile_ladder = None
    outfile_ladder_name = f"Ladder_{ladID}_{fmt_start_time}_{coolTemp}_{test_type}_{ph2acf_version}.root"

    print(f"💾 ==> Copying file content from infile to an another file : {outfile_ladder_name}\n")
    outfile_ladder = ROOT.TFile.Open(f"{OUTDIR}/{outfile_ladder_name}", "RECREATE")
    det_out_ladder = outfile_ladder.mkdir("Detector")
    copy_dir(infile.Get("Detector"), det_out_ladder)
    outfile_ladder.Write() # Output file is written here
    infile.Close()         # Input file is closed 

    # Getting the Detector and Board_0 TDirectories from the output file
    TDir_Detector_Aux = outfile_ladder.Get("Detector")
    TDir_Board_Aux    = outfile_ladder.Get("Detector/Board_0")


    dqm_file_name = CFG.get("DQM_FILE")
    dqm_file = ROOT.TFile(dqm_file_name, "READ")
    print(f"DQM --> Ph2ACF Ladder Monitor-Histogram file : {dqm_file_name}\n")

    monitor_dqm_dir = outfile_ladder.mkdir('MonitorDQM')
    monitor_dqm_det_dir = monitor_dqm_dir.mkdir('Detector')
    print(f"📝 ==> Adding MonitorDQM Histograms")
    copy_dir(dqm_file.Get("Detector"), monitor_dqm_det_dir)


    # ---------------------------------------------------------------------------- #
    #                                   Summary                                    #
    #        Fill with fStrings with info taken from several TDirectories          #
    # ---------------------------------------------------------------------------- #
    print(f"🧩 ==> Initiate Summary Directory")
    summary_dir = outfile_ladder.mkdir('Summary')
    summary_det_dir = summary_dir.mkdir('Detector')
    summary_det_brd_dir = summary_det_dir.mkdir('Board_0')
    for og in MODULE_POS:
        summary_det_brd_dir.mkdir(og)
    

    # ---------------------------------------------------------------------------- #
    #                     Monitor Power Supply and Environment                     #
    # ---------------------------------------------------------------------------- #

    # Create Monitor TDirectory
    print(f"🧩 ==> Initiate Monitor Directory")
    monitor_ps_dir = outfile_ladder.mkdir('Monitor')
    monitor_ps_det_dir = monitor_ps_dir.mkdir('Detector')
    monitor_ps_det_brd_dir = monitor_ps_det_dir.mkdir('Board_0')

    # Monitor PowerSupply and ColdBox Env : Read csv file, extract data and plot TGraph
    print(f" ... Read PS csv and add in Monitor")
    monitor_ps_file = CFG.get('PS_MONITOR_FILE')
    ps_channels = CFG.get('PS_CHANNELS')
    channels_connected = np.arange(24)[ps_channels[0]:ps_channels[1]] + 1
    
    monitor_ps_obj = MonitorPS(csvfile = monitor_ps_file,
                               ps_channels = channels_connected)
    PS_dict,ENV_dict = monitor_ps_obj.extract()


    #from IPython import embed; embed()
    # coldbox env
    #env_monitor_file = CFG.get('ENV_MONITOR_FILE')
    #monitor_env_obj = MonitorEnv(csvfile = env_monitor_file)
    #ENV_dict = monitor_env_obj.extract()

    
    dqm_vtrxoff_file_name = CFG.get("DQM_VTRX_OFF_FILE")
    dqm_vtrxoff_file = ROOT.TFile(dqm_vtrxoff_file_name, "READ")
    print(f"DQM --> Ph2ACF Ladder Monitor-Histogram file (VTRX Off) : {dqm_vtrxoff_file_name}\n")

    Ph2ACF_Log = CFG.get("PH2ACF_LOG_FILE")
    print(f"Ph2ACF Log --> {Ph2ACF_Log}")
    
    hist_obj = Histogrammer()

    # get timestamp from Ph2ACF log
    _timestamps = ENV_dict['timestamp']
    PS_ts_unconf_idx, PS_ts_conf_idx = get_idx_at_configuration(_timestamps, Ph2ACF_Log)
    
    for OG in MODULE_POS:
        LV_current_dict = PS_dict[OG]['LV']
        timestamp = LV_current_dict['timestamp']
        voltage = LV_current_dict['voltage']
        current = LV_current_dict['current']

        summary_og_dir = summary_det_brd_dir.GetDirectory(OG)
        write_string(summary_og_dir, "LV Current at start of Module Test (A)", current[0]) # add to summary
        write_string(summary_og_dir, "LV Current at stop of Module Test (A)", current[-1]) # add to summary
        
        
        gr_LV_c = hist_obj.makeMonitorLVCurrent(timestamp, current, moduleID=MODULE_POS[OG])
        gr_LV_v = hist_obj.makeMonitorLVVoltage(timestamp, voltage, moduleID=MODULE_POS[OG])


        # LV Current module unconfigured (A) / Configured (A)
        # -- from PS data, right after module rebooting, check the timestamp from HV Voltage and pick LV Current at the same
        LV_I_unconf = current[PS_ts_unconf_idx]
        LV_I_conf   = current[PS_ts_conf_idx]
        
        write_string(summary_og_dir, "LV Current module unconfigured (A)", LV_I_unconf) # add to summary
        write_string(summary_og_dir, "LV Current module configured (A)", LV_I_conf) # add to summary
        

        HV_current_dict = PS_dict[OG]['HV']
        timestamp = HV_current_dict['timestamp']
        voltage = HV_current_dict['voltage']
        current = HV_current_dict['current']

        write_string(summary_og_dir, "HV Current at start of Module Test (uA)", current[0]) # add to summary
        write_string(summary_og_dir, "HV Current at stop of Module Test (uA)", current[-1]) # add to summary
        
        gr_HV_c = hist_obj.makeMonitorHVCurrent(timestamp, current, moduleID=MODULE_POS[OG])
        gr_HV_v = hist_obj.makeMonitorHVVoltage(timestamp, voltage, moduleID=MODULE_POS[OG])

        if not monitor_ps_det_brd_dir.GetDirectory(OG):
            monitor_ps_det_brd_dir.mkdir(OG)

        monitor_ps_det_brd_og_dir = monitor_ps_det_brd_dir.GetDirectory(OG)

        monitor_ps_det_brd_og_dir.WriteObject(gr_LV_c, gr_LV_c.GetName())
        monitor_ps_det_brd_og_dir.WriteObject(gr_LV_v, gr_LV_v.GetName())
        monitor_ps_det_brd_og_dir.WriteObject(gr_HV_c, gr_HV_c.GetName())
        monitor_ps_det_brd_og_dir.WriteObject(gr_HV_v, gr_HV_v.GetName())

        
        env_timestamp   = ENV_dict['timestamp']
        env_temperature = ENV_dict['temperature']
        env_humidity    = ENV_dict['relhumidity']
        env_dewpoint    = ENV_dict['dewpoint']

        write_string(summary_og_dir, "Environment T at start of Module Test (C)", env_temperature[0]) # add to summary
        write_string(summary_og_dir, "Environment T at stop of Module Test (C)", env_temperature[-1]) # add to summary
        write_string(summary_og_dir, "RH at start of Module Test (%)", env_humidity[0]) # add to summary
        write_string(summary_og_dir, "RH at stop of Module Test (%)", env_humidity[-1]) # add to summary
                
        gr_env_T  = hist_obj.makeMonitorEnvTemp(env_timestamp, env_temperature, moduleID=MODULE_POS[OG])
        gr_env_H  = hist_obj.makeMonitorEnvHumidity(env_timestamp, env_humidity, moduleID=MODULE_POS[OG])
        gr_env_DP = hist_obj.makeMonitorEnvDewP(env_timestamp, env_dewpoint, moduleID=MODULE_POS[OG])
        
        
        monitor_ps_det_brd_og_dir.WriteObject(gr_env_T,  gr_env_T.GetName())
        monitor_ps_det_brd_og_dir.WriteObject(gr_env_H,  gr_env_H.GetName())
        monitor_ps_det_brd_og_dir.WriteObject(gr_env_DP, gr_env_DP.GetName())


        gr_carr_T  = hist_obj.makeMonitorCarrTemp(env_timestamp, env_temperature, moduleID=MODULE_POS[OG])
        monitor_ps_det_brd_og_dir.WriteObject(gr_carr_T,  gr_carr_T.GetName())

        # Set Chiller SP here
        gr_chil_sp_T = hist_obj.makeMonitorChilTemp(env_timestamp,
                                                    0 - np.ones_like(env_temperature)*30.0,
                                                    moduleID=MODULE_POS[OG])
        monitor_ps_det_brd_og_dir.WriteObject(gr_chil_sp_T,  gr_chil_sp_T.GetName())
        
        
        # get sensor temperature at the start of IV for each OG
        OG_No = OG.split('_')[-1]
        hstemp = dqm_vtrxoff_file.Get(f'Detector/Board_0/{OG}/D_B(0)_LpGBT_DQM_SensorTemp_OpticalGroup({OG_No})')
        temp_start_IV = get_first_entry_from_TGraph(hstemp)
        hstemp2 = dqm_file.Get(f'Detector/Board_0/{OG}/D_B(0)_LpGBT_DQM_SensorTemp_OpticalGroup({OG_No})')
        temp_stop_IV = get_first_entry_from_TGraph(hstemp2)

        #print(temp_start_IV, temp_stop_IV)
        
        write_string(summary_og_dir, "Sensor T at start of IV (C)", temp_start_IV) # add to summary
        write_string(summary_og_dir, "Sensor T at stop of IV (C)", temp_stop_IV) # add to summary
        write_string(summary_og_dir, "Sensor T at start of Module Test (C)", temp_stop_IV) # add to summary
        write_string(summary_og_dir, "Sensor T at stop of Module Test (C)", get_last_entry_from_TGraph(hstemp2)) # add to summary
        
        
    # ---------------------------------------------------------------------------- #
    #                  Monitor History : Keep it same as Monitor                   #
    # ---------------------------------------------------------------------------- #
    
    print(f"🧩 ==> Initiate MonitorHistory Directory")           
    monitor_his_ps_dir = outfile_ladder.mkdir('MonitorHistory')
    monitor_his_ps_det_dir = monitor_his_ps_dir.mkdir('Detector')
    monitor_his_ps_det_brd_dir = monitor_his_ps_det_dir.mkdir('Board_0')
    print(f" ... Copy Monitor in MonitorHistory")
    for OG in MODULE_POS:
        monitor_his_ps_det_brd_og_dir = monitor_his_ps_det_brd_dir.mkdir(OG)
        copy_dir(monitor_ps_det_brd_dir.GetDirectory(OG), monitor_his_ps_det_brd_og_dir)
        
    

    # ---------------------------------------------------------------------------- #
    #                                       IV                                     #
    # ---------------------------------------------------------------------------- #

    print(f"🧩 ==> Initiate IV Directory")
    IV_dir = outfile_ladder.mkdir('IV')
    IV_det_dir = IV_dir.mkdir('Detector')
    IV_det_brd_dir = IV_det_dir.mkdir('Board_0')
    
    
    IV_csv_dir = CFG.get('IV_FILE_DIR')
    print(f" ... Read IV csv per channel and add in IV Dir per OG")
    for ch in channels_connected:
        OG = f'OpticalGroup_{ch-1}'
        key = MODULE_POS[OG]
        file = f'{IV_csv_dir}/{key}_HV#{ch}.csv'
        #print(file)
        if not os.path.exists(file):
            raise Exception(f'{file} not found')
        monitor_IV_obj = MonitorIV(csvfile = file)
        IV_dict = monitor_IV_obj.extract()

        voltage = IV_dict['voltage']
        current = IV_dict['current']

        summary_og_dir = summary_det_brd_dir.GetDirectory(OG)
        write_string(summary_og_dir, "HV Current at start of IV (uA)", current[0]) # add to summary
        write_string(summary_og_dir, "HV Current at stop of IV (uA)", current[-1]) # add to summary 

        
        gr_IV_c = hist_obj.makeIVCurve(voltage, current, moduleID=key)
        
        if not IV_det_brd_dir.GetDirectory(OG):
            IV_det_brd_dir.mkdir(OG)

        IV_det_brd_og_dir = IV_det_brd_dir.GetDirectory(OG)

        IV_det_brd_og_dir.WriteObject(gr_IV_c, gr_IV_c.GetName())




    # ---------------------------------------------------------------------------- #
    #                                     INFO                                     #
    # ---------------------------------------------------------------------------- #


    _info = CFG.get('EXTRA_INFO')
    print(f"🧩 ==> Initiate Info Directory")
    info_dir = outfile_ladder.mkdir("Info")
    info_dir.WriteObject(ROOT.TObjString(f"{_info['Setup']}"), "Setup")
    info_dir.WriteObject(ROOT.TObjString(test_type), "Tasks")
    info_dir.WriteObject(ROOT.TObjString(TDir_Detector_Aux.GetKey('GitTag_Detector').ReadObj()), "Version")
    info_dir.WriteObject(ROOT.TObjString(TDir_Detector_Aux.GetKey('GitCommitHash_Detector').ReadObj()), "Commit")
    info_dir.WriteObject(ROOT.TObjString(TDir_Board_Aux.GetKey('D_NameId_Board_(0)').ReadObj()), "Board_IP")
    info_dir.WriteObject(ROOT.TObjString(f"{TDir_Detector_Aux.GetKey('Username_Detector').ReadObj()}@{TDir_Detector_Aux.GetKey('HostName_Detector').ReadObj()}"), "UserHost_PC")
    info_dir.WriteObject(ROOT.TObjString(TDir_Detector_Aux.GetKey('CalibrationStartTimestamp_Detector').ReadObj()), "StartDateTime")
    info_dir.WriteObject(ROOT.TObjString(TDir_Detector_Aux.GetKey('CalibrationStopTimestamp_Detector').ReadObj()), "StopDateTime")
    info_dir.WriteObject(ROOT.TObjString(f"{_info['RunNo']}"), "LocalRunNumber")
    info_dir.WriteObject(ROOT.TObjString(_info['Location']), "Location")
    info_dir.WriteObject(ROOT.TObjString(_info['Operator']), "Operator")
    info_dir.WriteObject(ROOT.TObjString(_info['ResultFolder']), "Result_Folder")
    info_dir.WriteObject(ROOT.TObjString(_info['RunType']), "Run_Type")
    info_dir.WriteObject(ROOT.TObjString(_info['StationName']), "Station_Name")
    info_dir.WriteObject(ROOT.TObjString(ladID), "Ladder_ID")
    info_dir.WriteObject(ROOT.TObjString(f"{_info['LadderSlot']}"), "Ladder_Slot")
    info_dir.WriteObject(ROOT.TObjString(coolTemp), "Cooling_Setpoint")
    info_dir.WriteObject(ROOT.TObjString(_info['Comment']), "Comment")


    
    OGs = args.opticalgroup
    if OGs == [-1]:
        OGs = list(range(12))

    print(f"\n>> OpticalGroup indices : {OGs}\n\n")

    
    info_obj = outfile_ladder.Get("Info")
    board_dir = outfile_ladder.Get("Detector/Board_0")
    dqm_board_dir = dqm_file.Get("Detector/Board_0")

    

    
    # Loop over the Optical Groups
    for OG in OGs:
        name = f'OpticalGroup_{OG}'
        obj = board_dir.Get(name)
        
        if not obj:
            print(f"⚠️  OpticalGroup_{OG} not found in Board_0")
            continue
        if not isinstance(obj, ROOT.TDirectory):
            raise RuntimeError(f"‼️  OpticalGroup_{OG} is not a TDirectory")

        dqm_obj = dqm_board_dir.Get(name)

        if not dqm_obj:
            print(f"⚠️  OpticalGroup_{OG} not found in Board_0 in MonitorHistogram file")
            continue
        if not isinstance(dqm_obj, ROOT.TDirectory):
            raise RuntimeError(f"‼️  OpticalGroup_{OG} is not a TDirectory")


        monitor_ps_det_brd_og_obj = monitor_ps_det_brd_dir.GetDirectory(f"OpticalGroup_{OG}")

        IV_det_brd_og_obj = IV_det_brd_dir.GetDirectory(f"OpticalGroup_{OG}")
        summary_det_brd_og_obj = summary_det_brd_dir.GetDirectory(f"OpticalGroup_{OG}")
        
        modID = MODULE_POS[name]
        print(f"📝 ==> Adding ModuleID : {modID} to D_B(0)_NameId_OpticalGroup({OG})")
        add_modID(obj, modID, OG)



        obj.Write("", ROOT.TObject.kOverwrite)
        
        """
        # to add histograms from OGs in MonitorHistogrm file to Main file
        dqm_dir = obj.mkdir("MonitorDQM")
        copy_dir(dqm_obj, dqm_dir, skipdir=True)
        obj.Write("", ROOT.TObject.kOverwrite)
        print(f"📝 ==> Adding TGraphs from Monitor Histogram file to D_B(0)_NameId_OpticalGroup({OG})\n")
        """
        
        if args.split == True:        
            print(f"⚙️  ==> Assigning Name to the ROOT file for {name}")
            lpgbt_id = obj.GetKey(f"D_B(0)_LpGBTFuseId_OpticalGroup({OG})").ReadObj()
            vtrx_id  = obj.GetKey(f"D_B(0)_VTRxFuseId_OpticalGroup({OG})").ReadObj()
            print(f" ... LpGBTID    : {lpgbt_id}")
            print(f" ... VTRxID     : {vtrx_id}")
            
            #outfilename = f"Results__{test_type}__{modID}__{name}__Ladder_{ladID}__CO2_{coolTemp}.root"
            outfilename = f"{modID}_{fmt_start_time}_{coolTemp}_{test_type}_{ph2acf_version}.root"
            print(f" >>> File Name  : {outfilename}")

            outfilename = f"{OUTDIR}/{outfilename}"
            outfile = ROOT.TFile(outfilename, "RECREATE")
            
            det_out   = outfile.mkdir("Detector")
            board_out = det_out.mkdir("Board_0")
            group_out = board_out.mkdir(name)

            print(f"💾 ==> Copying everything from OpticalGroup TDirectory to {name} ROOT file")
            copy_dir(obj, group_out)

            print(f"💾 ==> Copying TObjStrings from Board_0 TDirectory to {name} ROOT file")
            copy_dir(TDir_Board_Aux, board_out, skipdir=True)
            
            print(f"💾 ==> Copying TObjStrings from Detector TDirectory to {name} ROOT file")
            copy_dir(TDir_Detector_Aux, det_out, skipdir=True)


            #new
            dqm_out = outfile.mkdir("MonitorDQM")
            dqm_det_out = dqm_out.mkdir("Detector")
            dqm_det_brd_out = dqm_det_out.mkdir("Board_0")
            dqm_det_brd_og_out = dqm_det_brd_out.mkdir(f"OpticalGroup_{OG}")
            copy_dir(dqm_obj, dqm_det_brd_og_out)

            monitor_ps_out = outfile.mkdir("Monitor")
            copy_dir(monitor_ps_det_brd_og_obj, monitor_ps_out)

            monitor_his_out = outfile.mkdir("MonitorHistory")
            copy_dir(monitor_ps_det_brd_og_obj, monitor_his_out)
                        
            IV_out = outfile.mkdir("IV")
            copy_dir(IV_det_brd_og_obj, IV_out)

            summary_out = outfile.mkdir("Summary")
            copy_dir(summary_det_brd_og_obj, summary_out)

            info_out  = outfile.mkdir("Info")
            copy_dir(info_obj, info_out)
            info_out.WriteObject(ROOT.TObjString(modID), "Module_ID")

            
            outfile.Write("", ROOT.TObject.kOverwrite)
            outfile.Close()
            print(f"✅ Wrote {outfilename}\n")

        
    print(f"✅ Wrote {outfile_ladder_name} with ModuleIDs as NameId_OpticalGroup \n")
    outfile_ladder.Close()

    REAL_STOP = time.perf_counter()
    CPU_STOP  = time.time()

    print(f"Real time : {round(REAL_STOP - REAL_START, 2)} seconds")
    print(f"CPU time  : {round(CPU_STOP - CPU_START, 2)} seconds")



    
if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Sparsing Ph2ACF ROOT file with more than one Optical Group')
    parser.add_argument("-c",
                        "--config", 
                        type        = str,
                        required    = True,
                        help="Config file") 

    parser.add_argument("-s",
                        "--split",
                        action      = "store_true",
                        default     = False,
                        help="Split the ROOT file?")

    parser.add_argument("-og",
                        "--opticalgroup",
                        type=int,
                        nargs="+",
                        default=[-1],
                        required=False,
                        help="list of OGs; use -1 to select all (0–11)")
    

    args= parser.parse_args()
    
    main(args)
