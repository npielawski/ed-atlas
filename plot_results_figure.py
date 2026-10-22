"""Figure 5: scores of the mortality and diagnosis (chapter) models on SwED, Uppsala and MIMIC-IV.

Reads models/metrics_diagnoses.csv (columns: dataset, diagnosis, metric, score, ci_low, ci_high).
The mortality scores and the number of data points per chapter are entered below.
"""
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import numpy as np
import pandas as pd
from collections import OrderedDict

df = pd.concat([
    pd.read_csv("models/metrics_diagnoses.csv", dtype={"diagnosis": str}),
    pd.DataFrame({
        "dataset": ["SwED", "Akkis", "MIMIC"] * 7,
        "diagnosis": [" mortality"] * 7 * 3,
        "metric": ["NLL"]*3 + ["Brier"]*3 + ["ROC"]*3 + ["AP"]*3 + ["AP_chance_level"]*3 + ["uniform_ECE"]*3 + ["quantile_ECE"]*3,
        "score": [
            0.698, 0.702, 0.698,
            0.017, 0.018, 0.021,
            88.530, 88.310, 82.480,
            18.590, 12.870, 10.680,
            1.91, 1.94, 2.26,
            0., 0.002, 0.008,
            0., 0.002, 0.007
        ],
        "ci_low": [
            0.698, 0.701, 0.698,
            0.017, 0.018, 0.021,
            88.520, 88.190, 82.24,
            18.580, 12.876, 10.53,
            1.91, 1.94, 2.26,
            0., 0.002, 0.008,
            0., 0.002, 0.007
        ],
        "ci_high": [
            0.698, 0.702, 0.698,
            0.017, 0.018, 0.021,
            88.530, 88.390, 82.69,
            18.600, 12.895, 10.83,
            1.91, 1.94, 2.26,
            0., 0.003, 0.009,
            0., 0.002, 0.008
        ],
    })
])
df["dataset"] = df.dataset.map({"SwED": "train", "Akkis": "uppsala", "MIMIC": "mimic"})
df = df[(df["diagnosis"].str.len() == 2) | (df["diagnosis"] == " mortality")]
df["dset_index"] = (df["dataset"] == "test").astype(int)
df = df.sort_values(["diagnosis", "dset_index"])
TERM_NAMES = {
    " mortality": "mortality",
    "01": "A00-B99",
    "02": "C00-D48", 
    "03": "D50-D89",
    "04": "E00-E90", 
    "05": "F00-F99",
    "06": "G00-G99", 
    "07": "H00-H59",
    "08": "H60-H95",
    "09": "I00-I99", 
    "10": "J00-J99",
    "11": "K00-K93", 
    "12": "L00-L99",
    "13": "M00-M99",
    "14": "N00-N99",
    "15": "O00-O99", 
    "16": "P00-P96",
    "17": "Q00-Q99",
    "19": "S00-T98",
    "20": "V01-Y98", 
}
df["name"] = df["diagnosis"].map(TERM_NAMES) + " (" + df["dataset"] + ")"

N_train = [8280387,351637,391851,71821,177419,521853,274260,135160,118765,1078715,553441,714819,227096,995792,591531,233630,217,9883,1824803, 9381]
N_uppsala =  [ 638619, 27254, 28338, 7809, 20247, 45742, 19668, 11465,  5216,  76919, 37794, 63068, 18029, 56132, 41292,  3235,  8, 847,  96399,  113]
N_mimic = [583860, 54468, 49595, 97650, 285677, 196609, 91230, 15930, 4982, 298515, 91221, 167741, 23756, 91453, 100362, 5324, 56, 5241, 97215, 41685]

METRICS_NAME = OrderedDict({
    "NLL": "Negative Log-Likelihood ↓",
    "Brier": "Brier score ↓",
    "ROC": "C-Index ↑",
    "AP": "Average Precision ↑",
    "quantile_ECE": "Expected Calibration Error (quantile-based sampling) ↓",
    "uniform_ECE": "Expected Calibration Error (uniform sampling) ↓",
})

fix, axs = plt.subplots(len(METRICS_NAME)+1, 1, sharex=True, figsize=(12, 2*len(METRICS_NAME)))
ax0, axs = axs[-1], axs[:]

for i, metric in enumerate(METRICS_NAME.keys()):
    ax = axs[i]
    multiplier = 1
    
    db = df.query("metric == @metric").copy()
    
    idx = np.arange(0, len(db))
    loc = np.cumsum(np.array([7., 4., 4.])[idx % 3])
    locs = dict(zip(db["name"].unique().tolist(), loc))

    colors = {"train": "tab:blue", "uppsala": "tab:orange", "mimic": "tab:green"}

    for dset in ["train", "uppsala", "mimic"]:
        shift = {"train": 0, "uppsala": 1, "mimic": 2}[dset]
        for j, (_, row) in enumerate(db.query("dataset == @dset").iterrows()):
            ax.plot(
                [loc[shift::3][j], loc[shift::3][j]],
                [
                    row["score"] - multiplier * (row["score"] - row["ci_low"]),
                    row["score"] + multiplier * (row["ci_high"] - row["score"])
                ],
                c=colors[row["dataset"]]
            )
        ax.scatter(
            loc[shift::3],
            db.query("dataset == @dset")["score"],
            marker="o",
            c="white",
            edgecolors=colors[dset],
            zorder=3,
            s=10,
            label=f"Median {dset} score" if i == 0 else None
        )

    if metric == "ROC" or metric == "AP": mloc = 10.
    elif metric == "Brier": mloc = 0.02
    elif metric == "NLL": mloc = 0.01
    else: mloc = 0.02
    ax.yaxis.set_major_locator(mtick.MultipleLocator(mloc))
    ax.set_xticks(loc)
    ax.set_xticklabels(db["name"].to_list(), rotation=45, ha='right')
    marginy = plt.margins()[1] * (1 - 0)
    #ax.set_ylim(None, 1 + marginy)
    ax.grid(alpha=0.4)
    #ax.text(0, np.dot(ax.get_ylim(), [0.04, 1-0.04]), METRICS_NAME[metric], size=15, alpha=1., ha="left", va="top")
    ax.text(0, np.dot(ax.get_ylim(), [-0.2, 1+0.2]), METRICS_NAME[metric], size=15, alpha=1., ha="left", va="top")

    if metric == "AP":
        db_ = df.query("metric == 'AP_chance_level'").copy()
        for j, (_, row) in enumerate(db_.iterrows()):
            ax.plot(
                [locs[row["name"]]-.5, locs[row["name"]]+.5],
                [row["score"], row["score"]],
                c="k",
                label="Chance-level AP" if j == 0 else None
            )
    if i == 0 or i == 3:
        ax.legend(ncols=3, loc="upper left")   

ax0.bar(loc[0::3], N_train, width=1.5, zorder=3)
ax0.bar(loc[1::3], N_uppsala, width=1.5, zorder=3)
ax0.bar(loc[2::3], N_mimic, width=1.5, zorder=3)
ax0.semilogy()
formatter = mtick.ScalarFormatter()
formatter.set_scientific(False)
ax0.yaxis.set_major_formatter(
    mtick.FuncFormatter(lambda x, p: format(int(x), ","))
)
ax0.yaxis.set_major_locator(mtick.LogLocator(base=10, numticks=11))
ax0.grid(axis="y", alpha=0.4)
ax0.text(0, np.dot(ax0.get_ylim(), [-.2, 20]), "Number of symptoms and diagnoses", size=15, alpha=1., ha="left", va="top")

priority = {'train': 0, 'uppsala': 1, 'mimic': 2}
sorted_labels = sorted(db["name"].to_list(), key=lambda x: (x.split(' (')[0], priority[x.split(' (')[1][:-1]]))
sorted_labels = sorted_labels[-3:] + sorted_labels[:-3]
cleaned_labels = []
for i, name in enumerate(sorted_labels):
    final_name = None
    if i % 3 == 0:
        final_name = name
    else:
        final_name = name[name.find('('):]
    final_name = final_name.replace("train", "SwED")
    final_name = final_name.replace("uppsala", "Uppsala")
    final_name = final_name.replace("mimic", "MIMIC-IV")
    
    cleaned_labels.append(final_name)
ax0.set_xticks(loc)
ax0.set_xticklabels(cleaned_labels, rotation=45, ha='right', rotation_mode="anchor")
for label in ax0.get_xticklabels():
    label.set_fontsize(8)

plt.tight_layout()
plt.savefig("figures/results.svg")
plt.savefig("figures/results.pdf")
