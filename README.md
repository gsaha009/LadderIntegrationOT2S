# LadderIntegrationOT2S

# 📊 Ph2_ACF 2S Module Quick Plotting Toolkit

Generate quick plots from **Ph2_ACF** quick/full test results for **2S modules** — whether tested individually in a **single-module box** or integrated into a **ladder** within a **cold-box** setup.

## ✨ Features
- **Fast plotting** of Ph2_ACF quick/full test data
- Supports both **single module** and **ladder cold-box** configurations
- Flexible analysis levels:
  - 🔬 **CBC-level** inspection
  - 📦 **Module-level** summary
  - ⚖ **Cross-setup comparison** (single-module vs. ladder / ladder-1 vs. ladder-2)

## 📂 Typical Workflow
1. **Collect** test results from Ph2_ACF quick/full tests : Main requirement.
2. Then one/two `yaml` files e.g. in **Configs**, is/are needed to run the main function.
3. The **main** script first load the Data from ROOT file to a dictionary, that one can dump to a separate `yaml` if needed.
4. The **Plotter** takes the dictionary as input on-the-fly and starts producing the plots.

## 🛠 Requirements

Two options :
- Use `cvmfs` : The easiest way
- Or, use `Anaconda` to setup the environment : Easy but messy

Install requirements:

if `cvmfs` is not accessible, a bit longer process to be followed
```bash
# conda installation
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh
# use the `yml` and build the environment
conda env create -f environment.yml
# and, then clone the repo
```

But, if `cvmfs` is there, then life would be easier as mentioned above
```bash
# first clone the repo
git clone https://github.com/gsaha009/LadderIntegrationOT2S.git
source setup.sh
# and everything will be ready
```

Again, to clone the repository, if not done e.g. for w/o `cvmfs`:
```bash
git clone https://github.com/gsaha009/LadderIntegrationOT2S.git
```

## Structure of the code:
- `Configs` :
  -  Here you can find some `yaml` files, which are the config files with information about input, output, histogram names, module_IDs etc.
  - `tests.yml` config with some switches for test
- `DataLoader.py` :
  - It reads the ROOT files, and saves some information like noise, nHits, pedestal etc. in a dictionary. It needs to be modified if one wants to add more info to the dictionaries.
- `Plotter.py` :
  - Plotter reads the dictionaries to produce the plots. 
- `Fitter.py` :
  - Several models can be used to fit the plots e.g. nHits
- `main.py` :
  - the main function, reads the config, loads the ROOT files, produced the dictionaries and do the plots.

## How to RUN
```bash
python main.py -cs Configs/bla.yaml Configs/foo.yaml -t <Output_Tag>
# max 2 configs can be used if `compare` switch is True in `tests.yaml`, or one config if it is False
```