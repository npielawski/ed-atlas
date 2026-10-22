from calibration_curve import weighted_calibration_curve
from sklearn import metrics
from tqdm import tqdm
import numpy as np

def summarize_curves_from_metric(curves, metric):
    med_metric = np.quantile(metric, q=0.5, method="closest_observation")
    cilo_metric = np.quantile(metric, q=0.025, method="closest_observation")
    cihi_metric = np.quantile(metric, q=0.975, method="closest_observation")
    # Find the curves
    curve_med_id = np.argwhere(metric == med_metric)[0].item()
    curve_cilo_id = np.argwhere(metric == cilo_metric)[0].item()
    curve_cihi_id = np.argwhere(metric == cihi_metric)[0].item()
    # Return the metrics and the curves
    return (
        (med_metric, cilo_metric, cihi_metric),
        (curves[curve_med_id], curves[curve_cilo_id], curves[curve_cihi_id])
    )

def compute_metrics(results):
    scores = dict(
        metrics=dict(),
        curves=dict(),
    )

    # *** NLL ***
    all_nll = []
    y_true = results["y_true"].to_numpy()
    weight = results["weight"].to_numpy()
    for col in tqdm(results.select("^y_pred_.*$").columns):
        # The predictions are probabilities, not logits.
        p = np.clip(results[col].to_numpy().astype("float64"), 1e-12, 1 - 1e-12)
        all_nll.append(-(y_true * np.log(p) + (1 - y_true) * np.log1p(-p)))
    all_nll = np.array(all_nll)
    nll_per_model = all_nll @ weight / weight.sum()
    metrics_nll, curves_nll = summarize_curves_from_metric(all_nll, nll_per_model)
    scores["metrics"]["NLL"] = metrics_nll
    scores["curves"]["NLL"] = curves_nll

    # *** Brier ***
    brier = (results.select("^y_pred_.*$") - results["y_true"]).to_numpy()**2
    brier_weighted = results["weight"].to_numpy() @ brier / results["weight"].sum()
    brier_loci, brier_med, brier_hici = np.quantile(brier_weighted, [0.025, 0.5, 0.975])
    scores["metrics"]["Brier"] = (brier_med, brier_loci, brier_hici)

    # *** ROC ***
    all_fpr, all_tpr = [], []
    all_auc = []
    for col in tqdm(results.select("^y_pred_.*$").columns):
        fpr, tpr, _ = metrics.roc_curve(results["y_true"], results[col], sample_weight=results["weight"])
        all_fpr.append(fpr)
        all_tpr.append(tpr)
        all_auc.append(
            metrics.roc_auc_score(results["y_true"], results[col], sample_weight=results["weight"])
        )
    all_auc = np.array(all_auc)
    metrics_roc, curves_fpr = summarize_curves_from_metric(all_fpr, all_auc)
    _, curves_tpr = summarize_curves_from_metric(all_tpr, all_auc)
    scores["curves"]["ROC_fpr"] = curves_fpr
    scores["curves"]["ROC_tpr"] = curves_tpr
    scores["metrics"]["ROC"] = metrics_roc

    # *** PRECISION-RECALL CURVES ***
    all_pre, all_rec = [], []
    all_ap = []
    for col in tqdm(results.select("^y_pred_.*$").columns):
        pre, rec, _ = metrics.precision_recall_curve(results["y_true"], results[col], sample_weight=results["weight"])
        all_pre.append(pre)
        all_rec.append(rec)
        all_ap.append(
            metrics.average_precision_score(results["y_true"], results[col], sample_weight=results["weight"])
        )
    all_ap = np.array(all_ap)

    metrics_ap, curves_pre = summarize_curves_from_metric(all_pre, all_ap)
    _, curves_rec = summarize_curves_from_metric(all_rec, all_ap)
    scores["curves"]["AP_pre"] = curves_pre
    scores["curves"]["AP_rec"] = curves_rec
    scores["metrics"]["AP"] = metrics_ap
    AP_chance_level = (results["y_true"] * results["weight"]).sum() / results["weight"].sum()
    scores["metrics"]["AP_chance_level"] = (AP_chance_level, AP_chance_level, AP_chance_level)

    # *** CALIBRATION ERROR
    for cal_type in ["uniform", "quantile"]:
        all_p_true, all_p_pred, bin_totals = [], [], []
        for col in tqdm(results.select("^y_pred_.*$").columns):
            prob_true, prob_pred, bin_total = weighted_calibration_curve(
                results["y_true"],
                results[col],
                sample_weight=results["weight"],
                n_bins=20,
                strategy=cal_type,
                return_bin_total=True,
            )
            all_p_true.append(prob_true)
            all_p_pred.append(prob_pred)
            bin_totals.append(bin_total)
        all_p_true, all_p_pred = np.array(all_p_true), np.array(all_p_pred)
        bin_totals = np.array(bin_totals)

        med_p_true = np.quantile(all_p_true, q=0.5, axis=0, method="linear")
        cilo_p_true = np.quantile(all_p_true, q=0.025, axis=0, method="linear")
        cihi_p_true = np.quantile(all_p_true, q=0.975, axis=0, method="linear")
        scores["curves"][cal_type + "_cal_p_true"] = (med_p_true, cilo_p_true, cihi_p_true)
        med_p_pred = np.quantile(all_p_pred, q=0.5, axis=0, method="linear")
        cilo_p_pred = np.quantile(all_p_pred, q=0.025, axis=0, method="linear")
        cihi_p_pred = np.quantile(all_p_pred, q=0.975, axis=0, method="linear")
        scores["curves"][cal_type + "_cal_p_pred"] = (med_p_pred, cilo_p_pred, cihi_p_pred)

        bin_totals_sum = bin_totals.sum(1, keepdims=True)
        bin_totals_sum[bin_totals_sum == 0] = 1
        ECE = np.einsum(
            'ij,ij->i',
            bin_totals / bin_totals_sum,
            np.nan_to_num(np.abs(all_p_true - all_p_pred), 0)
        )

        metrics_ece, curves_p_true = summarize_curves_from_metric(all_p_true, ECE)
        _, curves_p_pred = summarize_curves_from_metric(all_p_pred, ECE)
        scores["curves"][cal_type + "_cal_p_true"] = curves_p_true
        scores["curves"][cal_type + "_cal_p_pred"] = curves_p_pred
        scores["metrics"][cal_type + "_ECE"] = metrics_ece

    return scores

def plot_results(axs, scores, results, dataset_name, color, *, partial=False):
    ax, ax2 = axs
    med_fpr, _, _ = scores["curves"]["ROC_fpr"]
    med_tpr, _, _ = scores["curves"]["ROC_tpr"]
    mean_roc, _, _ = scores["metrics"]["ROC"]
    med_pre, _, _ = scores["curves"]["AP_pre"]
    med_rec, _, _ = scores["curves"]["AP_rec"]
    mean_ap, _, _ = scores["metrics"]["AP"]
    ap_chance, _, _ = scores["metrics"]["AP_chance_level"]
    uniform_med_p_true, _, _ = scores["curves"]["uniform_cal_p_true"]
    uniform_med_p_pred, _, _ = scores["curves"]["uniform_cal_p_pred"]
    quantile_med_p_true, _, _ = scores["curves"]["quantile_cal_p_true"]
    quantile_med_p_pred, _, _ = scores["curves"]["quantile_cal_p_pred"]
    mean_ece_unif, _, _ = scores["metrics"]["uniform_ECE"]
    mean_ece_quant, _, _ = scores["metrics"]["quantile_ECE"]

    ax[0, 0].plot(
        med_fpr,
        med_tpr,
        color=color,
        label=f"[{dataset_name}] Median classifier (AUC={mean_roc:.02f})"
    )
    if not partial:
        ax[0, 0].plot([0, 1], [0, 1], "k--", label="Chance level (AUC=0.5)")
    ax[0, 0].set_xlabel("False Positive Rate")
    ax[0, 0].set_ylabel("True Positive Rate")
    ax[0, 0].set_title("ROC curve")
    ax[0, 0].legend()
    ax[0, 0].grid(visible=True)

    ax[0, 1].plot(
        med_rec,
        med_pre,
        label=f"[{dataset_name}] Classifier (AP={mean_ap:.02f})",
        color=color,
    )
    if not partial:
        ax[0, 1].plot(
            [0, 1],
            [ap_chance, ap_chance],
            "k--",
            label=f"Chance level (AP={ap_chance:.02f})"
        )
    ax[0, 1].set_xlabel("Recall")
    ax[0, 1].set_ylabel("Precision")
    ax[0, 1].set_title("PR curve")
    ax[0, 1].legend()
    ax[0, 1].grid(visible=True)

    ax[1, 0].plot(
        uniform_med_p_pred,
        uniform_med_p_true,
        color=color,
        marker="o",
        markersize=3,
        label=f"[{dataset_name}] Median classifier (ECE={mean_ece_unif:.02f})"
    )
    if not partial:
        ax[1, 0].plot([0, 1], [0, 1], "k:", label="Perfectly calibrated")
    ax[1, 0].set_xlabel("Mean predicted probability")
    ax[1, 0].set_ylabel("Fraction of positives")
    ax[1, 0].set_title("Calibration curve (uniform)")
    ax[1, 0].legend()
    ax[1, 0].grid(visible=True)

    ax2.hist(
        results["y_mean_pred"],
        range=(0, 1),
        bins=20,
        alpha=0.3,
        color=color,
        density=True,
        label="Distribution of predicted probability"
    )
    ax2.set_ylabel("Density of distribution")

    ax[1, 1].loglog()
    ax[1, 1].plot(
        quantile_med_p_pred,
        quantile_med_p_true,
        color=color,
        marker="o",
        markersize=3,
        label=f"[{dataset_name}] Median classifier (ECE={mean_ece_quant:.02f})"
    )
    if not partial:
        ax[1, 1].plot([0, 1], [0, 1], "k:", label="Perfectly calibrated")
    ax[1, 1].set_xlabel("Mean predicted probability")
    ax[1, 1].set_ylabel("Fraction of positives")
    ax[1, 1].set_title("Calibration curve (quantile)")
    ax[1, 1].legend()
    ax[1, 1].grid(visible=True)
