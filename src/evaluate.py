"""Functions for evaluating model performance and saving metrics."""

import pandas as pd
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix
)


def evaluate_classification(y_true, y_pred, y_prob=None) -> pd.DataFrame:
    """
    Evaluate classification model performance.
    Returns a DataFrame containing all important metrics.
    """

    # Core metrics
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)

    # Specificity (True Negative Rate)
    cm = confusion_matrix(y_true, y_pred)

    if cm.shape == (2, 2):
        TN = cm[0, 0]
        FP = cm[0, 1]
        specificity = TN / (TN + FP) if (TN + FP) != 0 else 0
    else:
        specificity = None  # For multiclass problems

    data = {
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1-Score": f1,
        "Specificity": specificity
    }

    # ROC-AUC (if probabilities provided)
    if y_prob is not None:
        try:
            data["ROC-AUC"] = roc_auc_score(y_true, y_prob)
        except Exception:
            data["ROC-AUC"] = None

    return pd.DataFrame([data])


def save_metrics(df: pd.DataFrame, path: str):
    """Save metrics DataFrame to CSV."""
    df.to_csv(path, index=False)


def plot_confusion_matrix(cm, classes=None):
    """Return a matplotlib figure showing the confusion matrix."""
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)

    if classes is not None:
        ax.set(
            xticks=np.arange(len(classes)),
            yticks=np.arange(len(classes)),
            xticklabels=classes,
            yticklabels=classes,
            ylabel='True label',
            xlabel='Predicted label'
        )

    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")

    fmt = 'd'
    thresh = cm.max() / 2.

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                j, i, format(cm[i, j], fmt),
                ha="center",
                va="center",
                color="white" if cm[i, j] > thresh else "black"
            )

    fig.tight_layout()
    return fig


def plot_roc(y_true, y_score):
    """Plot ROC curve."""
    from sklearn.metrics import roc_curve, auc
    import matplotlib.pyplot as plt

    fpr, tpr, _ = roc_curve(y_true, y_score)
    roc_auc = auc(fpr, tpr)

    fig, ax = plt.subplots()
    ax.plot(fpr, tpr, lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
    ax.plot([0, 1], [0, 1], lw=2, linestyle='--')

    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('Receiver Operating Characteristic')
    ax.legend(loc="lower right")

    return fig
