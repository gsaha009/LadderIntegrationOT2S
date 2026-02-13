# Splitting Ladder ROOT file to separate Optical Groups
# Author : Gourab Saha, IPHC

# How to Run:
# python main.py Configs_split_ladder_root_file/file.yaml


import os
import sys
import yaml
import ROOT
import time
import argparse

def load_cfg(cfg):
    cfgdict = None
    with open(cfg, 'r') as f:
        cfgdict = yaml.safe_load(f)
    return cfgdict


def copy_dir(src_dir, dest_dir, skipdir = False):
    for key in src_dir.GetListOfKeys():
        name = key.GetName()
        obj = key.ReadObj()

        #if isinstance(obj, ROOT.TDirectory):
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

    test_type = infile.Get("Detector").GetKey("CalibrationName_Detector").ReadObj()
    
    # Creating an output file in Write mode and then copy the contents
    outfile_ladder = None
    outfile_ladder_name = f"Results__{test_type}__Ladder_{ladID}__CO2_{coolTemp}.root"

    print(f"💾 ==> Copying file content from infile to an another file : {outfile_ladder_name}\n")
    outfile_ladder = ROOT.TFile.Open(f"{OUTDIR}/{outfile_ladder_name}", "RECREATE")
    det_out_ladder = outfile_ladder.mkdir("Detector")
    copy_dir(infile.Get("Detector"), det_out_ladder)
    outfile_ladder.Write() # Output file is written here
    infile.Close()         # Input file is closed 

    # Getting the Detector and Board_0 TDirectories from the output file
    TDir_Detector_Aux = outfile_ladder.Get("Detector")
    TDir_Board_Aux    = outfile_ladder.Get("Detector/Board_0")

    print(f"Calibration     : {test_type}")    
    print(f"Ph2-ACF version : {TDir_Detector_Aux.GetKey('GitTag_Detector').ReadObj()}")
    print(f"Ph2-ACF commit  : {TDir_Detector_Aux.GetKey('GitCommitHash_Detector').ReadObj()}")
    print(f"Host PC         : {TDir_Detector_Aux.GetKey('Username_Detector').ReadObj()}@{TDir_Detector_Aux.GetKey('HostName_Detector').ReadObj()}")
    print(f"START           : {TDir_Detector_Aux.GetKey('CalibrationStartTimestamp_Detector').ReadObj()}")
    print(f"STOP            : {TDir_Detector_Aux.GetKey('CalibrationStopTimestamp_Detector').ReadObj()}\n")
    print(f"Board IP        : {TDir_Board_Aux.GetKey('D_NameId_Board_(0)').ReadObj()}")
    print(f"Ladder ID       : {ladID}")
    print(f"Cooling Temp    : {coolTemp}\n")

    dqm_file_name = CFG.get("DQM_FILE")
    dqm_file = ROOT.TFile(dqm_file_name, "READ")
    print(f"DQM --> Ph2ACF Ladder Monitor-Histogram file : {dqm_file_name}\n")

    monitor_dqm_dir = outfile_ladder.mkdir('MonitorDQM')
    monitor_dqm_det_dir = monitor_dqm_dir.mkdir('Detector')
    print(f"📝 ==> Adding MonitorDQM Histograms")
    copy_dir(dqm_file.Get("Detector"), monitor_dqm_det_dir)

    monitor_ps_dir = outfile_ladder.mkdir('MonitorPS')
    monitor_ps_det_dir = monitor_ps_dir.mkdir('Detector')
    monitor_ps_det_brd_dir = monitor_ps_det_dir.mkdir('Board_0')
    
    monitor_env_dir = outfile_ladder.mkdir('MonitorEnv')

    
    _info = CFG.get('EXTRA_INFO')
    
    info_dir = outfile_ladder.mkdir("Info")
    info_dir.WriteObject(ROOT.TObjString(f"{_info['Setup']}"), "Setup")
    info_dir.WriteObject(ROOT.TObjString(TDir_Detector_Aux.GetKey('GitTag_Detector').ReadObj()), "Version")
    info_dir.WriteObject(ROOT.TObjString(TDir_Detector_Aux.GetKey('GitCommitHash_Detector').ReadObj()), "Commit")
    info_dir.WriteObject(ROOT.TObjString(test_type), "Tasks")
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



    IV_dir = outfile_ladder.mkdir('IV')
    IV_det_dir = IV_dir.mkdir('Detector')
    IV_det_brd_dir = IV_det_dir.mkdir('Board_0')
    
    
    OGs = args.opticalgroup
    if OGs == [-1]:
        OGs = list(range(12))

    print(f"OpticalGroup indices : {OGs}\n\n")

    
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


        monitor_ps_det_brd_og_obj = monitor_ps_det_brd_dir.mkdir(f"OpticalGroup_{OG}")

        IV_det_brd_og_obj = IV_det_brd_dir.mkdir(f"OpticalGroup_{OG}")

        
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
            
            outfilename = f"Results__{test_type}__{modID}__{name}__Ladder_{ladID}__CO2_{coolTemp}.root"
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

            monitor_ps_out = outfile.mkdir("MonitorPS")
            monitor_ps_det_out = monitor_ps_out.mkdir("Detector")
            monitor_ps_det_brd_out = monitor_ps_det_out.mkdir("Board_0")
            monitor_ps_det_brd_og_out = monitor_ps_det_brd_out.mkdir(f"OpticalGroup_{OG}")
            copy_dir(monitor_ps_det_brd_og_obj, monitor_ps_det_brd_og_out)
            
            
            IV_out = outfile.mkdir("IV")
            IV_det_out = IV_out.mkdir("Detector")
            IV_det_brd_out = IV_det_out.mkdir("Board_0")
            IV_det_brd_og_out = IV_det_brd_out.mkdir(f"OpticalGroup_{OG}")
            copy_dir(IV_det_brd_og_obj, IV_det_brd_og_out)

            monitor_env_out = outfile.mkdir("MonitorEnv")
            copy_dir(monitor_env_dir, monitor_env_out)
            

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
