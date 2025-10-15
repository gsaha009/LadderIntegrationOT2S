# Splitting Ladder ROOT file to separate Optical Groups
# Author : Gourab Saha, IPHC

# How to Run:
# python main.py <cfg_file.yaml>


import os
import sys
import yaml
import ROOT

def load_cfg(cfg):
    cfgdict = None
    with open(cfg, 'r') as f:
        cfgdict = yaml.safe_load(f)
    return cfgdict


def copy_dir(src_dir, dest_dir):
    for key in src_dir.GetListOfKeys():
        name = key.GetName()
        obj = key.ReadObj()

        if isinstance(obj, ROOT.TDirectory):
            dest_dir.mkdir(name)
            subdir = dest_dir.GetDirectory(name)
            copy_dir(obj, subdir)
        else:
            dest_dir.cd()
            obj.Write(name)


            
def main():
    args = sys.argv
    if len(args) < 2:
        raise RuntimeError("Config file is missing")

    CFG_FILE = args[1]
    CFG = load_cfg(CFG_FILE)
    
    infilename = CFG.get("INPUT_FILE")
    print(f"Ladder output : {infilename}")

    OUTDIR = CFG.get('OUTPUT_PATH')
    MODULE_POS = CFG.get("MODULE_POS")
    
    infile = ROOT.TFile(infilename, "READ")
    for key in infile.Get("Detector/Board_0").GetListOfKeys():
        name = key.GetName()
        obj = key.ReadObj()

        if not isinstance(obj, ROOT.TDirectory):
            continue
        if not "OpticalGroup" in name:
            continue

        # make outfile name
        outfilename = f"Results__{MODULE_POS[name]}__{name}__Ladder_{CFG.get('LADDER_NO')}__{CFG.get('LADDER_POS')}__{CFG.get('COOLING_TEMP')}.root"
        
        if OUTDIR is not None:
            if not os.path.exists(OUTDIR):
                os.mkdir(OUTDIR)
            outfilename = f"{OUTDIR}/{outfilename}"
                
        outfile = ROOT.TFile(outfilename, "RECREATE")

        det_out = outfile.mkdir("Detector")
        board_out = det_out.mkdir("Board_0")
        group_out = board_out.mkdir(name)
        
        copy_dir(obj, group_out)

        outfile.Write()
        outfile.Close()
        print(f"✅ Wrote {outfilename}")

    infile.Close()


if __name__ == "__main__":
    main()
