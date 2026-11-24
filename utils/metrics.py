"""Classification metrics implemented with NumPy (no scikit-learn dependency).

Per class it reports Accuracy (one-vs-rest), Precision, Recall, F1,
Specificity and one-vs-rest ROC-AUC; then the macro averages and the global
accuracy.
"""

from __future__ import annotations

from typing import Dict, List, Sequence

import numpy as np


def confusion_matrix(y_true: Sequence[int], y_pred: Sequence[int], num_classes: int) -> np.ndarray:
    cm = np.zeros((num_classes, num_classes), dtype=np.int64)
    for t, p in zip(y_true, y_pred):
        cm[int(t), int(p)] += 1
    return cm


def roc_auc_binary(y_true: np.ndarray, score: np.ndarray) -> float:
    """Rank-based AUC with average ranks for ties (Mann-Whitney U)."""
    y_true = y_true.astype(np.int64)
    n_pos = int(y_true.sum())
    n_neg = int(len(y_true) - n_pos)
    if n_pos == 0 or n_neg == 0:
        return float("nan")

    order = np.argsort(score, kind="mergesort")
    sorted_scores = score[order]
    ranks = np.empty(len(score), dtype=np.float64)

    i = 0
    n = len(score)
    while i < n:
        j = i
        while j + 1 < n and sorted_scores[j + 1] == sorted_scores[i]:
            j += 1
        ranks[order[i:j + 1]] = (i + j) / 2.0 + 1.0
        i = j + 1

    u = ranks[y_true == 1].sum() - n_pos * (n_pos + 1) / 2.0
    return float(u / (n_pos * n_neg))


def compute_metrics(y_true: Sequence[int], probs: np.ndarray,
                    class_names: Sequence[str]) -> Dict[str, object]:
    """Full metric set from labels and softmax probabilities."""
    y_true = np.asarray(y_true, dtype=np.int64)
    probs = np.asarray(probs, dtype=np.float64)
    y_pred = probs.argmax(axis=1)
    num_classes = probs.shape[1]

    cm = confusion_matrix(y_true, y_pred, num_classes)
    per_class: List[Dict[str, float]] = []

    for c in range(num_classes):
        t_bin = (y_true == c).astype(np.int64)
        p_bin = (y_pred == c).astype(np.int64)
        tp = int(((t_bin == 1) & (p_bin == 1)).sum())
        tn = int(((t_bin == 0) & (p_bin == 0)).sum())
        fp = int(((t_bin == 0) & (p_bin == 1)).sum())
        fn = int(((t_bin == 1) & (p_bin == 0)).sum())

        total = tp + tn + fp + fn
        acc = (tp + tn) / total if total else 0.0
        prec = tp / (tp + fp) if (tp + fp) else 0.0
        rec = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
        spec = tn / (tn + fp) if (tn + fp) else 0.0
        auc = roc_auc_binary(t_bin, probs[:, c])

        per_class.append({
            "class": class_names[c] if c < len(class_names) else str(c),
            "accuracy": acc, "precision": prec, "recall": rec,
            "f1": f1, "specificity": spec, "auc": auc,
            "support": int(t_bin.sum()),
        })

    def macro(key: str) -> float:
        return float(np.nanmean([m[key] for m in per_class]))

    return {
        "overall_accuracy": float((y_pred == y_true).mean()),
        "macro": {
            "accuracy": macro("accuracy"),
            "precision": macro("precision"),
            "recall": macro("recall"),
            "f1": macro("f1"),
            "specificity": macro("specificity"),
            "auc": macro("auc"),
        },
        "per_class": per_class,
        "confusion_matrix": cm.tolist(),
        "class_names": list(class_names),
        "num_samples": int(len(y_true)),
    }


def format_table(metrics: Dict[str, object]) -> str:
    """Plain-text table for console output."""
    header = f"{'class':<24}{'Acc':>9}{'Prec':>9}{'Rec':>9}{'F1':>9}{'Spec':>9}{'AUC':>9}{'N':>7}"
    lines = [header, "-" * len(header)]
    for m in metrics["per_class"]:
        lines.append(
            f"{m['class']:<24}{m['accuracy']:>9.4f}{m['precision']:>9.4f}"
            f"{m['recall']:>9.4f}{m['f1']:>9.4f}{m['specificity']:>9.4f}"
            f"{m['auc']:>9.4f}{m['support']:>7d}"
        )
    macro = metrics["macro"]
    lines.append("-" * len(header))
    lines.append(
        f"{'macro / overall':<24}{macro['accuracy']:>9.4f}{macro['precision']:>9.4f}"
        f"{macro['recall']:>9.4f}{macro['f1']:>9.4f}{macro['specificity']:>9.4f}"
        f"{macro['auc']:>9.4f}{metrics['num_samples']:>7d}"
    )
    lines.append(f"\noverall accuracy = {metrics['overall_accuracy']:.4f}")
    return "\n".join(lines)
