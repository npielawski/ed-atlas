"""Runs the test scripts and collects their scores into models/metrics_diagnoses.csv.

The CSV has the columns that plot_results_figure.py reads (dataset, diagnosis, metric,
score, ci_low, ci_high). Mortality rows have the diagnosis " mortality"; chapters have
their two-digit code. ROC, AP and AP_chance_level are in percent, as printed by the tests.

Besides the metrics of test_toolkit.py, the metric NLL_legacy is written: the NLL as it
was computed before it was corrected (probabilities passed as logits, rows unweighted).
It is the value of the first version of the paper, and is not used by any figure.

Already collected (dataset, diagnosis) pairs are skipped, so the script can be restarted.

    python collect_metrics.py                       # mortality and all chapters, 200 draws
    python collect_metrics.py --what mortality
    python collect_metrics.py --draws all --datasets SwED Akkis
"""
import argparse
import csv
import gc
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
from optax.losses import sigmoid_binary_cross_entropy

import test_toolkit

CHAPTERS = [f"{c:02d}" for c in list(range(1, 18)) + [19, 20]]
MORTALITY = " mortality"
SCRIPTS = {
    "SwED": ("test_mort.py", "test_diag.py"),
    "Akkis": ("test_akkis_mort.py", "test_akkis_diag.py"),
    "MIMIC": ("test_mimic_mort.py", "test_mimic_diag.py"),
}
PERCENT = {"ROC", "AP", "AP_chance_level"}
OUTPUT = "models/metrics_diagnoses.csv"

parser = argparse.ArgumentParser()
parser.add_argument("--what", nargs="+", default=["mortality", "chapters"], choices=["mortality", "chapters"])
parser.add_argument("--datasets", nargs="+", default=list(SCRIPTS), choices=list(SCRIPTS))
parser.add_argument("--draws", default="200", help='number of posterior draws to test, or "all"')
args = parser.parse_args()

# The scripts call compute_metrics once. Wrap it to keep its scores and add NLL_legacy.
captured = []
# Keep the real function on the module: this file can be run again in the same session.
if not hasattr(test_toolkit, "compute_metrics_original"):
    test_toolkit.compute_metrics_original = test_toolkit.compute_metrics
compute_metrics = test_toolkit.compute_metrics_original

def compute_metrics_with_legacy(results):
    scores = compute_metrics(results)
    y_true = results["y_true"].to_numpy()
    legacy = np.array([
        np.asarray(sigmoid_binary_cross_entropy(results[col].to_numpy(), y_true))
        for col in results.select("^y_pred_.*$").columns
    ])
    scores["metrics"]["NLL_legacy"], _ = test_toolkit.summarize_curves_from_metric(legacy, legacy.mean(1))
    captured.append(scores)
    return scores

test_toolkit.compute_metrics = compute_metrics_with_legacy

def done_pairs():
    if not os.path.exists(OUTPUT):
        return set()
    with open(OUTPUT) as f:
        return {(row["dataset"], row["diagnosis"]) for row in csv.DictReader(f)}

def run(script, argv, dataset, diagnosis):
    source = open(script).read()
    if args.draws != "all":
        source = source.replace('N_models = "all"', f"N_models = {int(args.draws)}")
    sys.argv = [script] + argv
    captured.clear()
    exec(compile(source, script, "exec"), {"__name__": "__main__"})
    scores = captured[-1]
    with open(OUTPUT, "a", newline="") as f:
        writer = csv.writer(f)
        for metric, (score, ci_low, ci_high) in scores["metrics"].items():
            k = 100 if metric in PERCENT else 1
            writer.writerow([dataset, diagnosis, metric, k * score, k * ci_low, k * ci_high])
    plt.close("all")
    gc.collect()

os.makedirs("models", exist_ok=True)
if not os.path.exists(OUTPUT):
    with open(OUTPUT, "w", newline="") as f:
        csv.writer(f).writerow(["dataset", "diagnosis", "metric", "score", "ci_low", "ci_high"])

# The tests of Uppsala and MIMIC-IV add their curves to the figure of the previous dataset,
# so the datasets have to be run in this order, and the pickles of SwED must come first.
datasets = [d for d in SCRIPTS if d in args.datasets]
if "mortality" in args.what:
    for dataset in datasets:
        if (dataset, MORTALITY) in done_pairs():
            continue
        print(f"=== {dataset} {MORTALITY.strip()}", flush=True)
        run(SCRIPTS[dataset][0], [], dataset, MORTALITY)
if "chapters" in args.what:
    for chapter in CHAPTERS:
        for dataset in datasets:
            if (dataset, chapter) in done_pairs():
                continue
            print(f"=== {dataset} chapter {chapter}", flush=True)
            run(SCRIPTS[dataset][1], [chapter], dataset, chapter)
print("Done:", OUTPUT)
