import argparse
import os
import json
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    confusion_matrix,
    classification_report,
    roc_curve,
    auc,
)
from sklearn.preprocessing import label_binarize
from torch.utils.data import DataLoader
from model import QuotientTransformer, QuotientSpaceDataset, MAX_OBSTACLES

CLASS_NAMES = ["Feasible", "Link 1", "Link 2", "Link 3", "Link 4"]
NUM_CLASSES = 5

def load_model(weights_path, device):
    model = QuotientTransformer(
        d_model=64, n_heads=4, num_layers=2, dropout=0.0  # dropout=0 at eval
    ).to(device)
    state = torch.load(weights_path, map_location=device)
    model.load_state_dict(state)
    model.eval()
    print(f"Loaded weights from: {weights_path}")
    return model

def run_inference(model, loader, device):
    all_targets, all_preds, all_probs = [], [], []
    with torch.no_grad():
        for batch in loader:
            state        = batch["state"].to(device)
            obstacles    = batch["obstacles"].to(device)
            padding_mask = batch["padding_mask"].to(device)
            target       = batch["target"].to(device)

            logits = model(state, obstacles, padding_mask)
            probs  = torch.softmax(logits, dim=1)
            preds  = probs.argmax(dim=1)

            all_targets.extend(target.cpu().numpy().tolist())
            all_preds.extend(preds.cpu().numpy().tolist())
            all_probs.append(probs.cpu().numpy())

    all_probs = np.vstack(all_probs)
    return np.array(all_targets), np.array(all_preds), all_probs

def plot_confusion_matrix(cm_arr, title, filename, normalise=False):
    fig, ax = plt.subplots(figsize=(7, 6))
    fmt = ".2f" if normalise else "d"
    cmap = cm.Blues
    im = ax.imshow(cm_arr, interpolation="nearest", cmap=cmap, vmin=0, vmax=(1.0 if normalise else None))
    plt.colorbar(im, ax=ax)

    tick_marks = np.arange(NUM_CLASSES)
    ax.set_xticks(tick_marks)
    ax.set_yticks(tick_marks)
    ax.set_xticklabels(CLASS_NAMES, rotation=40, ha="right", fontsize=10)
    ax.set_yticklabels(CLASS_NAMES, fontsize=10)

    thresh = cm_arr.max() / 2.0
    for i in range(NUM_CLASSES):
        for j in range(NUM_CLASSES):
            val = cm_arr[i, j]
            txt = f"{val:{fmt}}" if normalise else f"{int(val)}"
            ax.text(j, i, txt, ha="center", va="center", color="white" if val > thresh else "black", fontsize=9)

    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.set_xlabel("Predicted Label", fontsize=11)
    ax.set_ylabel("True Label", fontsize=11)
    plt.tight_layout()
    plt.savefig(filename, dpi=200)
    plt.close()
    print(f"Saved: {filename}")

def plot_roc_curves(targets, probs, filename):
    y_bin = label_binarize(targets, classes=list(range(NUM_CLASSES)))
    fig, ax = plt.subplots(figsize=(8, 6))
    colours = ["steelblue", "tomato", "seagreen", "darkorange", "mediumpurple"]
    for i, (name, col) in enumerate(zip(CLASS_NAMES, colours)):
        fpr, tpr, _ = roc_curve(y_bin[:, i], probs[:, i])
        roc_auc = auc(fpr, tpr)
        ax.plot(fpr, tpr, color=col, lw=2, label=f"{name}  (AUC = {roc_auc:.3f})")
    ax.plot([0, 1], [0, 1], "k--", lw=1)
    ax.set_xlim([0, 1]); ax.set_ylim([0, 1.02])
    ax.set_xlabel("False Positive Rate", fontsize=11)
    ax.set_ylabel("True Positive Rate", fontsize=11)
    ax.set_title("One-vs-Rest ROC Curves", fontsize=13, fontweight="bold")
    ax.legend(loc="lower right", fontsize=9)
    ax.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(filename, dpi=200)
    plt.close()
    print(f"Saved: {filename}")

def plot_calibration(targets, probs, filename, n_bins=10):
    fig, axes = plt.subplots(2, 3, figsize=(13, 8))
    axes = axes.flatten()
    y_bin = label_binarize(targets, classes=list(range(NUM_CLASSES)))

    for i, name in enumerate(CLASS_NAMES):
        ax = axes[i]
        bin_edges = np.linspace(0, 1, n_bins + 1)
        frac_pos, mean_conf = [], []
        for lo, hi in zip(bin_edges[:-1], bin_edges[1:]):
            mask = (probs[:, i] >= lo) & (probs[:, i] < hi)
            if mask.sum() == 0:
                continue
            frac_pos.append(y_bin[mask, i].mean())
            mean_conf.append(probs[mask, i].mean())
        ax.plot([0, 1], [0, 1], "k--", lw=1, label="Perfect")
        ax.plot(mean_conf, frac_pos, "o-", color="steelblue",
                lw=2, ms=5, label="Model")
        ax.set_title(name, fontsize=11)
        ax.set_xlabel("Mean predicted prob.", fontsize=9)
        ax.set_ylabel("Fraction positives", fontsize=9)
        ax.set_xlim([0, 1]); ax.set_ylim([0, 1])
        ax.grid(True, linestyle="--", alpha=0.4)
        ax.legend(fontsize=8)

    axes[-1].set_visible(False)
    fig.suptitle("Calibration / Reliability Diagrams", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(filename, dpi=200)
    plt.close()
    print(f"Saved: {filename}")

def plot_confidence_histogram(targets, preds, probs, filename):
    confidence = probs.max(axis=1)
    correct    = targets == preds
    fig, ax = plt.subplots(figsize=(8, 5))
    bins = np.linspace(0, 1, 30)
    ax.hist(confidence[correct],  bins=bins, alpha=0.65, color="steelblue",
            label=f"Correct  (n={correct.sum():,})")
    ax.hist(confidence[~correct], bins=bins, alpha=0.65, color="tomato",
            label=f"Wrong    (n={(~correct).sum():,})")
    ax.set_xlabel("Max-class Softmax Probability", fontsize=11)
    ax.set_ylabel("Count", fontsize=11)
    ax.set_title("Prediction Confidence Distribution", fontsize=13, fontweight="bold")
    ax.legend(fontsize=10)
    ax.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(filename, dpi=200)
    plt.close()
    print(f"Saved: {filename}")


def breakdown_by_field(targets, preds, dataset, field):
    values = sorted(set(item[field] for item in dataset.data))
    result = {}
    for v in values:
        indices = [i for i, item in enumerate(dataset.data) if item[field] == v]
        t = targets[indices]; p = preds[indices]
        result[v] = 100.0 * accuracy_score(t, p)
    return result


def evaluate(args):
    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    dataset = QuotientSpaceDataset(args.dataset, max_obstacles=MAX_OBSTACLES)
    loader  = DataLoader(dataset, batch_size=256, shuffle=False, num_workers=0)
    model = load_model(args.weights, device)
    total_params = sum(p.numel() for p in model.parameters())
    trainable    = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}  |  Trainable: {trainable:,}")

    print("\nRunning inference on the full dataset …")
    targets, preds, probs = run_inference(model, loader, device)

    overall_acc  = 100.0 * accuracy_score(targets, preds)
    macro_f1     = f1_score(targets, preds, average="macro",     zero_division=0)
    weighted_f1  = f1_score(targets, preds, average="weighted",  zero_division=0)
    macro_prec   = precision_score(targets, preds, average="macro",    zero_division=0)
    macro_rec    = recall_score(targets, preds,    average="macro",    zero_division=0)

    clf_report = classification_report(
        targets, preds,
        target_names=CLASS_NAMES,
        digits=4,
        zero_division=0
    )

    dof_acc  = breakdown_by_field(targets, preds, dataset, "dof")
    obs_acc  = breakdown_by_field(targets, preds, dataset, "num_obstacles")
    cm_abs  = confusion_matrix(targets, preds)
    cm_norm = cm_abs.astype(float) / cm_abs.sum(axis=1, keepdims=True)
    cm_path = os.path.join(args.output_dir, "confusion_matrix.png")
    cm_norm_path = os.path.join(args.output_dir, "confusion_matrix_normalised.png")
    plot_confusion_matrix(cm_abs,  "Confusion Matrix (counts)", cm_path, normalise=False)
    plot_confusion_matrix(cm_norm, "Confusion Matrix (row-normalised)", cm_norm_path, normalise=True)
    roc_path = os.path.join(args.output_dir, "roc_curves.png")
    plot_roc_curves(targets, probs, roc_path)
    cal_path = os.path.join(args.output_dir, "calibration.png")
    plot_calibration(targets, probs, cal_path)
    conf_path = os.path.join(args.output_dir, "confidence_hist.png")
    plot_confidence_histogram(targets, preds, probs, conf_path)

    y_bin = label_binarize(targets, classes=list(range(NUM_CLASSES)))
    auc_per_class = {}
    for i, name in enumerate(CLASS_NAMES):
        fpr, tpr, _ = roc_curve(y_bin[:, i], probs[:, i])
        auc_per_class[name] = auc(fpr, tpr)

    sep   = "=" * 60
    lines = [
        sep,
        "  QUOTIENT-SPACE TRANSFORMER — EVALUATION REPORT",
        sep,
        f"  Dataset file : {args.dataset}",
        f"  Weights file : {args.weights}",
        f"  Total samples: {len(dataset):,}",
        f"  Device       : {device}",
        f"  Model params : {total_params:,}  (trainable: {trainable:,})",
        sep,
        "",
        "OVERALL METRICS",
        f"  Overall Accuracy  : {overall_acc:.4f}%",
        f"  Macro Precision   : {macro_prec:.4f}",
        f"  Macro Recall      : {macro_rec:.4f}",
        f"  Macro F1          : {macro_f1:.4f}",
        f"  Weighted F1       : {weighted_f1:.4f}",
        "",
        "PER-CLASS METRICS (sklearn classification_report)",
        clf_report,
        "PER-CLASS AUC (One-vs-Rest)",
    ]
    for name, a in auc_per_class.items():
        lines.append(f"  {name:<12}: AUC = {a:.4f}")

    lines += [
        "",
        "PER-DOF ACCURACY",
    ]
    for dof, acc in dof_acc.items():
        lines.append(f"  DOF {dof}: {acc:.2f}%")

    lines += [
        "",
        "PER-OBSTACLE-COUNT ACCURACY",
    ]
    for n_obs, acc in obs_acc.items():
        lines.append(f"  {n_obs:2d} obstacles: {acc:.2f}%")

    lines += [
        "",
        "CONFUSION MATRIX (absolute counts)",
        "  Rows = True label, Columns = Predicted label",
        "  Classes: " + "  ".join(f"{c:>10}" for c in CLASS_NAMES),
    ]
    for i, row in enumerate(cm_abs):
        lines.append("  " + f"{CLASS_NAMES[i]:>10}" + "".join(f"{v:11d}" for v in row))

    lines += [
        "",
        "CONFUSION MATRIX (row-normalised)",
        "  Classes: " + "  ".join(f"{c:>10}" for c in CLASS_NAMES),
    ]
    for i, row in enumerate(cm_norm):
        lines.append("  " + f"{CLASS_NAMES[i]:>10}" + "".join(f"{v:11.3f}" for v in row))

    lines += [
        "",
        "SAVED",
        f"  confusion_matrix.png",
        f"  confusion_matrix_normalised.png",
        f"  roc_curves.png",
        f"  calibration.png",
        f"  confidence_hist.png",
        f"  evaluation_report.txt",
        sep,
    ]

    report_text = "\n".join(lines)
    print("\n" + report_text)
    report_path = os.path.join(args.output_dir, "evaluation_report.txt")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report_text)
    print(f"\nReport saved to: {report_path}")

def parse_args():
    p = argparse.ArgumentParser(description="Evaluate a trained Quotient-Space Transformer.")
    p.add_argument("--dataset", default="dataset.json", help="Path to dataset JSON file.")
    p.add_argument("--weights", default="best_model_weights.pth", help="Path to model weights (.pth).")
    p.add_argument("--output_dir", default="eval_results", help="Directory to save evaluation outputs.")
    return p.parse_args()

if __name__ == "__main__":
    evaluate(parse_args())