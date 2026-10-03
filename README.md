# LadderIntegrationOT2S

# 📊 Ph2_ACF 2S Module Quick Plotting Toolkit

Generate quick plots from **Ph2_ACF** quick/full test results for **2S modules** — whether tested individually in a **single-module box** (pre-integration) or integrated to a **ladder** and tested inside IPHC **cold-box** setup (post-integration).

## ✨ Features
- **Fast plotting** of Ph2_ACF quick/full test data
- Supports both **single module** and **ladder cold-box** configurations
- Flexible analysis levels:
  - **CBC-level** inspection
  - **Module-level** summary
  - **Cross-setup comparison** (single-module vs. ladder)

## 🛠 Requirements

Two options :
  - Use `cvmfs` : The easiest way
  - Or, use `Anaconda` to setup the environment : Easy, but messy

if `cvmfs` is not accessible, a bit longer process to be followed
```bash
# conda installation
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh
# use the `yml` and build the environment
conda env create -f environment.yml
```

Or, if `cvmfs` is there, then life would be easier as mentioned above
```bash
# first clone the repo
git clone https://github.com/gsaha009/LadderIntegrationOT2S.git
source setup.sh
```

## 🤖 Now, lets start ...

  - Clone the main repository [for now, checkout `dev` branch]
  ```bash
  git clone --recurse-submodules https://github.com/gsaha009/LadderIntegrationOT2S.git
  git checkout dev
  ```
  - How to run?
    - The main script is `main.py`
    - two mandatory inputs are `Ph2ACF Calibration Results directory` and `Ph2ACF VTRXoff Results directory`
      ```bash
      python3 main.py -i path/to/2SfullTest -ivtrx path/to/vtrxoff -s <#to split the merged potato compatible ROOT file> -qccfg "bla.yaml" <#default is "qc_config.yaml">
      ```
  - IMPORTANT:
    - The `PreInt Results` should be downloaded before hand, and the best strategy would be to check if the files are correct.
    - Perhaps, we should not add the `downloading Preint results` task to the maste GUI controller.
    - The files should come from `2SfullTest` and `Potato-converted` from `DCA`.
    - If one/two files are found missing from `DCA`, it would be better to test the modules in the single module box.
    - We need to coume-up with some smart idea, however, if we finally decide to overlook the pre-int results, the `preint check` can be made optional.


## The final output structure

```bash
TB2SLadders__pos_1__TB2S_Ladder_minus_6CP_9__pos_2__TB2S_Ladder_plus_5CP_4__+15C__2SfullTestAnalysis__2026-10-03_23_01_28
├── Pos_1__TB2S_Ladder_minus_6CP_9
│   ├── AnalysisResults
│   │   ├── Configs
│   │   │   ├── configPostInt.yaml
│   │   │   └── configPreInt.yaml
│   │   └── Results
│   │       └── version_2026-10-03_23_01_28
│   │           ├── Files
│   │           │   ├── TB2S_Ladder__pos_1__minus_6CP_9.root
│   │           │   ├── TB2S_Ladder__pos_1__minus_6CP_9__grade.csv
│   │           │   └── TB2S_Ladder__pos_1__minus_6CP_9__preint_postint_diff.csv
│   │           └── Plots
│   │               ├── Plot_BrokenChannels_hybrid0_allModules_compare.png
│   │               ├── Plot_BrokenChannels_hybrid1_allModules_compare.png
│   │               ├── Plot_CMN_frac_module_compare.png
│   │               ├── ...
│   │               ├── PostInt__Pos_1__minus_6CP_9
│   │               │   ├── 2S_18_5_BRN-01001
│   │               │   │   ├── CBCLevel
│   │               │   │   │   ├── Hist_DelayVsThr_Hybrid0_2S_18_5_BRN-01001_PostInt__Pos_1__minus_6CP_9.png
│   │               │   │   │   ├── Hist_DelayVsThr_Hybrid1_2S_18_5_BRN-01001_PostInt__Pos_1__minus_6CP_9.png
│   │               │   │   │   ├── ...
│   │               │   │   │   └── SCurve_2S_18_5_BRN-01001.png
│   │               │   │   ├── Hist_StripNoise_bothHybrids_bothSensors_2S_18_5_BRN-01001_PostInt__Pos_1__minus_6CP_9.png
│   │               │   │   ├── LpGBT_EyeOpening.png
│   │               │   │   ├── ...
│   │               │   │   ├── Plot_StripNoise_bothHybrids_TopSensor_2S_18_5_BRN-01001_PostInt__Pos_1__minus_6CP_9.png
│   │               │   │   └── VTRx_LightYeildScan.png
│   │               │   ├── 2S_18_6_NCP-10003
│   │               │   │   ├── CBCLevel
│   │               │   │   │   ├── Hist_DelayVsThr_Hybrid0_2S_18_6_NCP-10003_PostInt__Pos_1__minus_6CP_9.png
│   │               │   │   │   ├── ...
│   │               │   │   │   └── SCurve_2S_18_6_NCP-10003.png
│   │               │   │   ├── Hist_StripNoise_bothHybrids_bothSensors_2S_18_6_NCP-10003_PostInt__Pos_1__minus_6CP_9.png
│   │               │   │   ├── LpGBT_EyeOpening.png
│   │               │   │   ├── ...
│   │               │   │   └── VTRx_LightYeildScan.png
│   │               │   ├── 2S_18_6_NCP-10006
│   │               │   │   ├── CBCLevel
│   │               │   │   │   ├── Hist_DelayVsThr_Hybrid0_2S_18_6_NCP-10006_PostInt__Pos_1__minus_6CP_9.png
│   │               │   │   │   ├── Hist_DelayVsThr_Hybrid1_2S_18_6_NCP-10006_PostInt__Pos_1__minus_6CP_9.png
│   │               │   │   │   ├── ...
│   │               │   │   │   └── SCurve_2S_18_6_NCP-10006.png
│   │               │   │   ├── Hist_StripNoise_bothHybrids_bothSensors_2S_18_6_NCP-10006_PostInt__Pos_1__minus_6CP_9.png
│   │               │   │   ├── LpGBT_EyeOpening.png
│   │               │   │   ├── ...
│   │               │   │   └── VTRx_LightYeildScan.png
│   │               │   ├── ...
│   │               │   ├── 2S_18_6_NCP-10023
│   │               │   │   ├── CBCLevel
│   │               │   │   │   ├── Hist_DelayVsThr_Hybrid0_2S_18_6_NCP-10023_PostInt__Pos_1__minus_6CP_9.png
│   │               │   │   │   ├── ...
│   │               │   │   │   └── SCurve_2S_18_6_NCP-10023.png
│   │               │   │   ├── Hist_StripNoise_bothHybrids_bothSensors_2S_18_6_NCP-10023_PostInt__Pos_1__minus_6CP_9.png
│   │               │   │   ├── LpGBT_EyeOpening.png
│   │               │   │   ├── ...
│   │               │   │   └── VTRx_LightYeildScan.png
│   │               │   ├── Eye_Opening_Scan_Pow_0p33.png
│   │               │   ├── Eye_Opening_Scan_Pow_0p67.png
│   │               │   ├── ...
│   │               │   ├── Sensor_Temp_OGs.png
│   │               │   └── VTRx_LightYield_Ladder.png
│   │               └── PreInt__Pos_1__minus_6CP_9
│   │                   ├── 2S_18_5_BRN-01001
│   │                   │   ├── CBCLevel
│   │                   │   │   ├── Hist_DelayVsThr_Hybrid0_2S_18_5_BRN-01001_PreInt__Pos_1__minus_6CP_9.png
│   │                   │   │   ├── ...
│   │                   │   │   └── SCurve_2S_18_5_BRN-01001.png
│   │                   │   ├── Hist_StripNoise_bothHybrids_bothSensors_2S_18_5_BRN-01001_PreInt__Pos_1__minus_6CP_9.png
│   │                   │   ├── ...
│   │                   │   └── VTRx_LightYeildScan.png
│   │                   ├── 2S_18_6_NCP-10023
│   │                   │   ├── CBCLevel
│   │                   │   │   ├── Hist_DelayVsThr_Hybrid0_2S_18_6_NCP-10023_PreInt__Pos_1__minus_6CP_9.png
│   │                   │   │   ├── ...
│   │                   │   │   └── SCurve_2S_18_6_NCP-10023.png
│   │                   │   ├── Hist_StripNoise_bothHybrids_bothSensors_2S_18_6_NCP-10023_PreInt__Pos_1__minus_6CP_9.png
│   │                   │   ├── ...
│   │                   │   └── VTRx_LightYeildScan.png
│   │                   ├── Eye_Opening_Scan_Pow_0p33.png
│   │                   ├── ...
│   │                   ├── Sensor_Temp_OGs.png
│   │                   └── VTRx_LightYield_Ladder.png
│   ├── ROOTfilesForDCA
│   │   ├── Configs
│   │   │   └── config_minus_6CP_9.yaml
│   │   └── Results
│   │       └── v_2026-10-03_23_01_28
│   │           ├── 2S_18_5_BRN-01001_2026-10-02_10h58m20s_+15C_2SfullTest_v6-29.root
│   │           ├── 2S_18_6_NCP-10003_2026-10-02_10h58m20s_+15C_2SfullTest_v6-29.root
│   │           ├── ...
│   │           ├── 2S_18_6_NCP-10023_2026-10-02_10h58m20s_+15C_2SfullTest_v6-29.root
│   │           └── TB2S_Ladder_minus_6CP_9_2026-10-02_10h58m20s_+15C_2SfullTest_v6-29.root
│   └── Summary
│       ├── quality_grade.pdf
│       └── quality_grade.png
├── Pos_2__TB2S_Ladder_plus_5CP_4
│   ├── AnalysisResults
│   │   ├── Configs
│   │   │   ├── configPostInt.yaml
│   │   │   └── configPreInt.yaml
│   │   └── Results
│   │       └── version_2026-10-03_23_01_28
│   │           ├── Files
│   │           │   ├── TB2S_Ladder__pos_2__plus_5CP_4.root
│   │           │   ├── TB2S_Ladder__pos_2__plus_5CP_4__grade.csv
│   │           │   └── TB2S_Ladder__pos_2__plus_5CP_4__preint_postint_diff.csv
│   │           └── Plots
│   │               ├── Plot_BrokenChannels_hybrid0_allModules_compare.png
│   │               ...
└── TB2SLadders__pos_1__TB2S_Ladder_minus_6CP_9__pos_2__TB2S_Ladder_plus_5CP_4__+15C__2SfullTestAnalysis__2026-10-03_23_01_28.log

```

## SubModules

| Path | Repository | Purpose |
|------|------------|---------|
| `modules/2SLadderCMNAna` | [GitHub Repo](https://github.com/gsaha009/2SLadderCMNAna) | CMNoise fitter |
| `modules/EventContainer` | [GitLab Repo](https://gitlab.cern.ch/gsaha/eventcontainer.git) | Read OTPhysics Root file |