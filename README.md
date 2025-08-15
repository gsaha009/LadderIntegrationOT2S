# LadderIntegrationOT2S

# 📊 Ph2_ACF 2S Module Quick Plotting Toolkit

Generate quick, insightful plots from **Ph2_ACF** quick/full test results for **2S modules** — whether tested individually in a **single-module box** or integrated into a **ladder** within a **cold-box** setup.

## ✨ Features
- **Fast plotting** of Ph2_ACF quick/full test data
- Supports both **single module** and **ladder cold-box** configurations
- Flexible analysis levels:
  - 🔬 **CBC-level** inspection
  - 📦 **Module-level** summary
  - ⚖ **Cross-setup comparison** (single-module vs. ladder / ladder-1 vs. ladder-2)

## 📂 Typical Workflow
1. **Collect** test results from Ph2_ACF quick/full tests.
2. **Run the scripts** with your dataset.
3. **Explore plots** for:
   - Per-chip performance metrics
   - Per-module aggregated statistics
   - Comparative performance between setups

## 🛠 Requirements
- Use `Anaconda` and the [`conda_requirements.txt`]() file to setup the environment


Install requirements:
```bash
# conda installation
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh

# use the `yml` and build the environment
conda env create -f environment.yml

# Set up the plotter
git clone https://github.com/gsaha009/LadderIntegrationOT2S.git
```

## Structure of the code:
- `Configs` :
  -  Here you can find some `yaml` files, which are the config files with information about input, output, histogram names, module_IDs etc.
  - `tests.yml` config with some switches
- `DataLoader.py` :
- `Plotter.py` :
- `Fitter.py` :
- `main.py` :


1. `cd LadderIntegrationOT2S/Configs`  