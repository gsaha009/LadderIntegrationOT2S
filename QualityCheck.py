import os
import yaml
import ROOT
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import mplhep as hep
from matplotlib.colors import ListedColormap, BoundaryNorm

import logging
logger = logging.getLogger('main')


def graph_to_valerr(g):
    n = g.GetN()

    arr = np.recarray(n, dtype=[("val", float), ("err", float)])

    for i in range(n):
        arr.val[i] = g.GetY()[i]
        arr.err[i] = g.GetEY()[i]

    return arr


def load_ladder_graphs(root_file, ladder="Ladder", periods=("PreInt", "PostInt")):
    out = {}

    f = ROOT.TFile.Open(root_file)
    if not f or f.IsZombie():
        raise OSError(f"Could not open ROOT file: {root_file}")

    ladder_dir = f.Get(ladder)
    if not ladder_dir:
        f.Close()
        raise KeyError(f"Directory not found: {ladder}")

    for period in periods:
        period_dir = ladder_dir.Get(period)
        if not period_dir:
            print(f"Warning: directory not found: {ladder}/{period}")
            continue

        for key in period_dir.GetListOfKeys():
            name = key.GetName()
            obj = period_dir.Get(name)

            if not obj.InheritsFrom("TGraphErrors"):
                continue

            arr = graph_to_valerr(obj)

            if len(arr) != 12:
                print(f"Warning: {period}/{name} has {len(arr)} points, not 12")

            out.setdefault(name, {})[period] = arr

    f.Close()
    return out



class QualityCheck:
    def __init__(self,
                 ladderfile : str,
                 qc_config : str,
                 OGs : dict,
                 **kwargs):
        self.dataOG = OGs
        self.qc_config_data = self.config_to_dict(qc_config = qc_config)
        self.ladder_data = self.root_to_array(ladderfile = ladderfile)

    def config_to_dict(self, qc_config = None):
        with open(qc_config, 'r') as f:
            data = yaml.safe_load(f)
        return data
            
    def root_to_array(self, ladderfile = None):
        data = load_ladder_graphs(ladderfile)
        return data

    def analyze(self):
        """
        
        """
        grade_dict = {}
        prepostcomp_dict = {}
        for field, cut_info in self.qc_config_data.items():
            
            logger.info(f"Feature : {field}")
            
            range_to_use = cut_info['range']
            prepostcomp  = cut_info['prepostcomp']

            # get the field from self.ladder_data
            preint_field_from_ladder_data  = self.ladder_data[field]['PreInt'].val
            postint_field_from_ladder_data = self.ladder_data[field]['PostInt'].val

            
            # pre-post int difference
            prepostdiff = np.full(12, np.nan)
            if prepostcomp:
                prepostdiff = np.round(postint_field_from_ladder_data - preint_field_from_ladder_data, 3)

            prepostcomp_dict[field] = prepostdiff.tolist()

            # apply cuts to get grade
            grade = np.full(12, 'F')
            if range_to_use:
                for grade_key, cut_range in range_to_use.items():
                    cut_range_low  = cut_range[0]
                    cut_range_high = cut_range[1]
                    mask = (postint_field_from_ladder_data > cut_range_low) & (postint_field_from_ladder_data < cut_range_high)
                    grade = np.where(mask, grade_key, grade)
                
            else:
                if prepostcomp:
                    grade = np.where(prepostdiff <= 0.0, 'A', 'F')
                else:
                    raise Exception('both range and prepostcomp can be set at null ... check qc_config !')
                
            grade_dict[field] = grade.tolist()    


        return grade_dict,prepostcomp_dict


    def get_df(self, grade_dict, prepostcomp_dict):
        #OGs = ['feature'] + [f"OG{OG.split('_')[-1]} [{modID}]" for OG, modID in self.dataOG.items()]
        OGs = [f"OG{OG.split('_')[-1]} [{modID}]" for OG, modID in self.dataOG.items()]
        df_grade   = pd.DataFrame.from_dict(grade_dict, orient="index")
        df_grade.columns = OGs
        df_prepost = pd.DataFrame.from_dict(prepostcomp_dict, orient="index")
        df_prepost.columns = OGs
        
        return df_grade, df_prepost


    def plot_qc(self, data = None, outdir = ""):        
        #hep.style.use("CMS")
        grade_map = {
            "A": 1,
            "B": 2,
            "C": 3,
            "F": 4,
        }
        dataval = data.map(grade_map.get).to_numpy(dtype=float)
        cmap = ListedColormap([
            "#2ca25f",   # A
            "#FFD54F",   # B
            "#F57C00",   # C
            "#EF5350",   # F
        ])
        norm = BoundaryNorm([0.5, 1.5, 2.5, 3.5, 4.5],cmap.N)
        
        fig, ax = plt.subplots(figsize=(15, 8))
        im = ax.imshow(
            dataval,
            cmap=cmap,
            norm=norm,
            aspect="auto"
        )

        xlabels = [col.split()[0] for col in data.columns]
        ax.set_xticks(np.arange(len(data.columns)))
        ax.set_xticklabels(
            xlabels,
            rotation=90,
            ha="center",
            fontsize=14,
            fontweight="bold"
        )

        ax.set_yticks(np.arange(len(data.index)))
        ax.set_yticklabels(
            data.index,
            rotation=0,
            fontsize=13
        )

        for i in range(data.shape[0]):
            for j in range(data.shape[1]):
                grade = data.iloc[i, j]
                ax.text(
                    j,
                    i,
                    str(grade),
                    ha="center",
                    va="center",
                    fontsize=13,
                    fontweight="bold",
                    color="black"
                )

        ax.set_xticks(np.arange(-0.5, len(data.columns), 1),
                      minor=True)

        ax.set_yticks(np.arange(-0.5, len(data.index), 1),
                      minor=True)

        ax.grid(which="minor",
                color="white",
                linewidth=2)

        ax.tick_params(which="minor",
                       bottom=False,
                       left=False)

        cbar = fig.colorbar(
            im,
            ax=ax,
            ticks=[1, 2, 3, 4],
            pad=0.025,
            fraction=0.035
        )

        cbar.ax.set_yticklabels(["A", "B", "C", "F"],
                                fontsize=13,
                                fontweight="bold")
        
        cbar.set_label("Grade",
                       fontsize=14,
                       fontweight="bold")

        ax.set_title("Quality Grade",
                     fontsize=26,
                     fontweight="bold",
                     pad=55)

        ax.text(0.5,
                1.015,
                "A = Good   |   B = Warning   |   C = Critical  |  F = RIP",
                transform=ax.transAxes,
                ha="center",
                va="bottom",
                fontsize=13)

        #hep.cms.label("Work in Progress",
        #              data=True,
        #              com=None,
        #              ax=ax,
        #              loc=0)
        
        ax.set_xlabel("")
        ax.set_ylabel("")

        plt.tight_layout()

        plt.savefig(f"{outdir}/quality_grade.pdf", bbox_inches="tight")
        plt.savefig(f"{outdir}/quality_grade.png", dpi=300, bbox_inches="tight")



    def plot_prepost(self, data = None, outdir = ""):
        pass
