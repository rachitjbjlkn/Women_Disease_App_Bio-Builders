"""Comprehensive visualization and reporting utilities."""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px

from sklearn.metrics import (
    roc_curve,
    auc,
    precision_recall_curve,
    confusion_matrix,
    roc_auc_score,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)

from sklearn.calibration import calibration_curve


# -------------------- ROC --------------------

def plot_roc_auc_plotly(y_true, y_prob):
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    roc_auc = auc(fpr, tpr)

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=fpr, y=tpr, mode='lines',
                             name=f'ROC (AUC={roc_auc:.3f})'))
    fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode='lines',
                             name='Random', line=dict(dash='dash')))

    fig.update_layout(
        title='ROC Curve',
        xaxis_title='False Positive Rate',
        yaxis_title='True Positive Rate',
        template='plotly_dark'
    )

    return fig


# -------------------- Precision Recall --------------------

def plot_precision_recall_plotly(y_true, y_prob):
    precision, recall, _ = precision_recall_curve(y_true, y_prob)

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=recall, y=precision,
                             mode='lines', fill='tozeroy',
                             name='PR Curve'))

    fig.update_layout(
        title='Precision-Recall Curve',
        xaxis_title='Recall',
        yaxis_title='Precision',
        template='plotly_dark'
    )

    return fig


# -------------------- Confusion Matrix --------------------

def plot_confusion_matrix_plotly(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)

    fig = go.Figure(
        data=go.Heatmap(
            z=cm,
            colorscale='Blues',
            text=cm,
            texttemplate='%{text}'
        )
    )

    fig.update_layout(
        title='Confusion Matrix',
        xaxis_title='Predicted',
        yaxis_title='True',
        template='plotly_dark'
    )

    return fig


# -------------------- Calibration --------------------

def plot_calibration_plotly(y_true, y_prob):
    try:
        prob_true, prob_pred = calibration_curve(y_true, y_prob, n_bins=10)

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=prob_pred,
            y=prob_true,
            mode='lines+markers',
            name='Calibration'
        ))

        fig.add_trace(go.Scatter(
            x=[0, 1],
            y=[0, 1],
            mode='lines',
            name='Perfect',
            line=dict(dash='dash')
        ))

        fig.update_layout(
            title='Calibration Curve',
            xaxis_title='Mean Predicted Probability',
            yaxis_title='Fraction of Positives',
            template='plotly_dark'
        )

        return fig
    except:
        return None


# -------------------- Feature Importance --------------------

def plot_feature_importance_plotly(model, feature_names):
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_

        df = pd.DataFrame({
            'feature': feature_names,
            'importance': importances
        }).sort_values('importance', ascending=True).tail(10)

        fig = px.bar(
            df,
            x='importance',
            y='feature',
            orientation='h',
            title='Top 10 Feature Importances',
            template='plotly_dark'
        )

        return fig

    return None


# -------------------- Learning Curve --------------------

def plot_learning_curve_plotly(train_loss, val_loss):
    epochs = list(range(len(train_loss)))

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=epochs, y=train_loss,
                             mode='lines', name='Training Loss'))
    fig.add_trace(go.Scatter(x=epochs, y=val_loss,
                             mode='lines', name='Validation Loss'))

    fig.update_layout(
        title='Learning Curve',
        xaxis_title='Epoch',
        yaxis_title='Loss',
        template='plotly_dark'
    )

    return fig


# -------------------- Class Distribution --------------------

def plot_class_distribution_plotly(y):
    unique, counts = np.unique(y, return_counts=True)

    fig = px.bar(
        x=unique,
        y=counts,
        labels={'x': 'Class', 'y': 'Count'},
        title='Class Distribution',
        template='plotly_dark'
    )

    return fig


# -------------------- Prediction Distribution --------------------

def plot_prediction_distribution_plotly(y_prob):
    fig = px.histogram(
        x=y_prob,
        nbins=30,
        title='Prediction Probability Distribution',
        template='plotly_dark'
    )

    return fig


# -------------------- ROC Comparison --------------------

def plot_roc_comparison_plotly(results_dict):
    fig = go.Figure()

    for model_name, (y_true, y_pred, y_prob) in results_dict.items():
        if y_prob is not None:
            fpr, tpr, _ = roc_curve(y_true, y_prob)
            roc_auc = auc(fpr, tpr)

            fig.add_trace(go.Scatter(
                x=fpr,
                y=tpr,
                mode='lines',
                name=f'{model_name} (AUC={roc_auc:.3f})'
            ))

    fig.add_trace(go.Scatter(
        x=[0, 1],
        y=[0, 1],
        mode='lines',
        name='Random',
        line=dict(dash='dash')
    ))

    fig.update_layout(
        title='ROC Curves Comparison',
        xaxis_title='FPR',
        yaxis_title='TPR',
        template='plotly_dark'
    )

    return fig


# -------------------- Metrics Table --------------------

def create_metrics_table(y_true, y_pred, y_prob=None):
    """Create comprehensive metrics dictionary."""

    try:
        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred, zero_division=0)
        recall = recall_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)

        cm = confusion_matrix(y_true, y_pred)

        if cm.shape == (2, 2):
            TN = cm[0, 0]
            FP = cm[0, 1]
            specificity = TN / (TN + FP) if (TN + FP) != 0 else 0
        else:
            specificity = None

        metrics = {
            "Accuracy": accuracy,
            "Precision": precision,
            "Recall": recall,
            "F1-Score": f1,
            "Specificity": specificity
        }

        if y_prob is not None and cm.shape == (2, 2):
            metrics["AUC-ROC"] = roc_auc_score(y_true, y_prob)

        return metrics

    except Exception as e:
        print("Metric calculation error:", e)
        return {}
