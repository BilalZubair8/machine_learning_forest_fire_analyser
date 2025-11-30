import json
import logging

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.calibration import calibration_curve
from sklearn.metrics import (confusion_matrix,
                             accuracy_score, precision_score, recall_score, f1_score,
                             roc_auc_score, roc_curve, precision_recall_curve,
                             average_precision_score)

logger = logging.getLogger(__name__)


class ModelEvaluator:
    def __init__(self):
        self.metrics_history = {}
        self.best_model = None

    def comprehensiveEvalution(self, model, X, y, model_name,
                                 scaled=False, scaler=None,
                                 save_plots=True):
        """Comprehensive model evaluation with enhanced metrics"""
        if scaled and scaler is not None:
            X_eval = scaler.transform(X)
        else:
            X_eval = X

        # Predictions
        y_pred = model.predict(X_eval)
        y_pred_proba = model.predict_proba(X_eval)[:, 1] if hasattr(model, 'predict_proba') else None

        # Calculate comprehensive metrics
        metrics = self._calculateMetrics(y, y_pred, y_pred_proba, model_name)

        # Generate plots
        if save_plots:
            self._genarateAllPlots(y, y_pred, y_pred_proba, model_name, metrics)

        # Save metrics
        self.metrics_history[model_name] = metrics

        return metrics

    def _calculateMetrics(self, y_true, y_pred, y_pred_proba, model_name):
        """Calculate comprehensive performance metrics"""
        # Calculate basic metrics first
        metrics = {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred, zero_division=0),
            'recall': recall_score(y_true, y_pred, zero_division=0),
            'f1': f1_score(y_true, y_pred, zero_division=0),
            'roc_auc': roc_auc_score(y_true, y_pred_proba) if y_pred_proba is not None else None,
            'average_precision': average_precision_score(y_true, y_pred_proba) if y_pred_proba is not None else None
        }

        # Calculate confusion matrix components
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

        # Calculate specificity and other metrics
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        false_positive_rate = fp / (fp + tn) if (fp + tn) > 0 else 0
        false_negative_rate = fn / (fn + tp) if (fn + tp) > 0 else 0
        balanced_accuracy = (metrics['recall'] + specificity) / 2

        # Now update metrics with the new calculated values
        metrics.update({
            'specificity': specificity,
            'false_positive_rate': false_positive_rate,
            'false_negative_rate': false_negative_rate,
            'balanced_accuracy': balanced_accuracy
        })

        logger.info(f"\n{model_name.upper()} PERFORMANCE")
        logger.info(f"Accuracy: {metrics['accuracy']:.4f}")
        logger.info(f"Precision: {metrics['precision']:.4f}")
        logger.info(f"Recall: {metrics['recall']:.4f}")
        logger.info(f"F1-Score: {metrics['f1']:.4f}")
        logger.info(f"AUC-ROC: {metrics['roc_auc']:.4f}")
        logger.info(f"Average Precision: {metrics['average_precision']:.4f}")
        logger.info(f"Specificity: {metrics['specificity']:.4f}")
        logger.info(f"Balanced Accuracy: {metrics['balanced_accuracy']:.4f}")

        return metrics

    def _genarateAllPlots(self, y_true, y_pred, y_pred_proba, model_name, metrics):
        """Generate all evaluation plots"""
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        axes = axes.ravel()

        # 1. Confusion Matrix
        self._plotConfusionMatrix(y_true, y_pred, model_name, ax=axes[0])

        # 2. ROC Curve
        if y_pred_proba is not None:
            self._plotRocCurve(y_true, y_pred_proba, model_name, metrics['roc_auc'], ax=axes[1])

        # 3. Precision-Recall Curve
        if y_pred_proba is not None:
            self._plotPrecissionRecallCurve(y_true, y_pred_proba, model_name,
                                              metrics['average_precision'], ax=axes[2])


        # 4. Calibration Curve
        if y_pred_proba is not None:
            self._plotCallibrationCurve(y_true, y_pred_proba, model_name, ax=axes[3])

        # 5. Metrics Summary
        self._plotMetricsSummary(metrics, model_name, ax=axes[4])

        # 6. Prediction Distribution
        if y_pred_proba is not None:
            self._plotPredictionDistribution(y_pred_proba, model_name, ax=axes[5])

        plt.tight_layout()
        plt.savefig(f'plots/{model_name}_evaluation.png', dpi=300, bbox_inches='tight')
        plt.show()

    def _plotConfusionMatrix(self, y_true, y_pred, model_name, ax=None):
        """confusion matrix plot"""
        cm = confusion_matrix(y_true, y_pred)
        if ax is None:
            plt.figure(figsize=(8, 6))
            ax = plt.gca()

        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                    xticklabels=['Area <= 4%', 'Area > 4%'],
                    yticklabels=['Area <= 4%', 'Area > 4%'])
        ax.set_title(f'Confusion Matrix - {model_name}')
        ax.set_ylabel('Actual')
        ax.set_xlabel('Predicted')

    def _plotRocCurve(self, y_true, y_pred_proba, model_name, auc_score, ax=None):
        """Enhanced ROC curve"""
        fpr, tpr, _ = roc_curve(y_true, y_pred_proba)

        if ax is None:
            plt.figure(figsize=(8, 6))
            ax = plt.gca()

        ax.plot(fpr, tpr, linewidth=2, label=f'{model_name} (AUC = {auc_score:.3f})')
        ax.plot([0, 1], [0, 1], 'k--', label='Random Classifier')
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        ax.set_title(f'ROC Curve - {model_name}')
        ax.legend(loc='lower right')
        ax.grid(True, alpha=0.3)

    def _plotPrecissionRecallCurve(self, y_true, y_pred_proba, model_name, ap_score, ax=None):
        """Enhanced Precision-Recall curve"""
        precision, recall, _ = precision_recall_curve(y_true, y_pred_proba)

        if ax is None:
            plt.figure(figsize=(8, 6))
            ax = plt.gca()

        ax.plot(recall, precision, linewidth=2, label=f'AP = {ap_score:.3f}')
        ax.set_xlabel('Recall')
        ax.set_ylabel('Precision')
        ax.set_title(f'Precision-Recall Curve - {model_name}')
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.legend(loc='upper right')
        ax.grid(True, alpha=0.3)

    def _plotCallibrationCurve(self, y_true, y_pred_proba, model_name, ax=None):
        """Plot calibration curve"""
        prob_true, prob_pred = calibration_curve(y_true, y_pred_proba, n_bins=10)

        if ax is None:
            plt.figure(figsize=(8, 6))
            ax = plt.gca()

        ax.plot(prob_pred, prob_true, 's-', label=model_name)
        ax.plot([0, 1], [0, 1], 'k--', label='Perfectly calibrated')
        ax.set_xlabel('Mean predicted probability')
        ax.set_ylabel('Fraction of positives')
        ax.set_title(f'Calibration Curve - {model_name}')
        ax.legend()
        ax.grid(True, alpha=0.3)

    def _plotMetricsSummary(self, metrics, model_name, ax=None):
        """Plot metrics summary bar chart"""
        if ax is None:
            plt.figure(figsize=(8, 6))
            ax = plt.gca()

        metric_names = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'Specificity']
        metric_values = [
            metrics['accuracy'], metrics['precision'], metrics['recall'],
            metrics['f1'], metrics['specificity']
        ]

        bars = ax.bar(metric_names, metric_values, color=['#2E86AB', '#A23B72', '#F18F01', '#C73E1D', '#3B1F2B'])
        ax.set_ylim(0, 1)
        ax.set_title(f'Metrics Summary - {model_name}')
        ax.set_ylabel('Score')

        # Add value labels on bars
        for bar, value in zip(bars, metric_values):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                    f'{value:.3f}', ha='center', va='bottom')

    def _plotPredictionDistribution(self, y_pred_proba, model_name, ax=None):
        """Plot prediction distribution"""
        if ax is None:
            plt.figure(figsize=(8, 6))
            ax = plt.gca()

        ax.hist(y_pred_proba, bins=50, alpha=0.7, edgecolor='black')
        ax.set_xlabel('Predicted Probability')
        ax.set_ylabel('Frequency')
        ax.set_title(f'Prediction Distribution - {model_name}')
        ax.grid(True, alpha=0.3)


    def compareModels(self, models_metrics):
        """Compare multiple models """
        comparison_df = pd.DataFrame(models_metrics).T

        # Select best model based on multiple criteria
        best_f1 = comparison_df['f1'].idxmax()
        best_auc = comparison_df['roc_auc'].idxmax()
        best_balanced = comparison_df['balanced_accuracy'].idxmax()

        # Comprehensive scoring (weighted average)
        comparison_df['composite_score'] = (
                comparison_df['f1'] * 0.4 +
                comparison_df['roc_auc'] * 0.3 +
                comparison_df['balanced_accuracy'] * 0.3
        )

        best_model = comparison_df['composite_score'].idxmax()
        self.best_model = best_model

        logger.info(f"\n{'=' * 60}")
        logger.info("MODEL COMPARISON SUMMARY")
        logger.info(f"{'=' * 60}")
        for model_name, metrics in models_metrics.items():
            markers = []
            if model_name == best_f1: markers.append("F1")
            if model_name == best_auc: markers.append("AUC")
            if model_name == best_balanced: markers.append("BACC")
            if model_name == best_model: markers.append("BEST")

            # Use asterisk (*) instead of Unicode star (★) for better compatibility
            marker_str = " *" + "(" + ",".join(markers) + ")" if markers else ""

            logger.info(f"{model_name}{marker_str}: "
                        f"F1={metrics['f1']:.4f}, "
                        f"AUC={metrics['roc_auc']:.4f}, "
                        f"BAcc={metrics['balanced_accuracy']:.4f}")

        logger.info(f"\nOVERALL BEST MODEL: {best_model}")

        return best_model, comparison_df

    def saveEvaluationReport(self, filename='reports/evaluation_report.json'):
        """Save evaluation report"""
        report = {
            'metrics_history': self.metrics_history,
            'best_model': self.best_model,
            'timestamp': pd.Timestamp.now().isoformat()
        }

        with open(filename, 'w') as f:
            json.dump(report, f, indent=2)

        logger.info(f"Evaluation report saved to {filename}")



def evaluateModel(model, X, y, model_name, scaled=False, scaler=None):
    evaluator = ModelEvaluator()
    return evaluator.comprehensiveEvalution(model, X, y, model_name, scaled, scaler)



def plotFeatureImportance(model, feature_names, model_name, top_n=10):
    """feature importance plotting"""
    if hasattr(model, 'feature_importances_'):
        importance = model.feature_importances_
        feature_importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': importance
        }).sort_values('importance', ascending=False)

        plt.figure(figsize=(12, 8))
        sns.barplot(data=feature_importance_df.head(top_n),
                    x='importance', y='feature',
                    hue='feature',  # Add this line
                    palette='viridis',
                    legend=False)   # Add this line
        plt.title(f'Top {top_n} Feature Importance - {model_name}')
        plt.xlabel('Importance')
        plt.tight_layout()
        plt.savefig(f'plots/{model_name}_feature_importance.png', dpi=300, bbox_inches='tight')
        plt.show()

        return feature_importance_df
    else:
        logger.warning(f"Feature importance not available for {model_name}")
        return None


# Standalone plotting functions for backward compatibility

def plotConfusionMatrix(y_true, y_pred, model_name):
    """
    Standalone confusion matrix plot function
    """
    cm = confusion_matrix(y_true, y_pred)

    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Area <= 4%', 'Area > 4%'],
                yticklabels=['Area <= 4%', 'Area > 4%'])
    plt.title(f'Confusion Matrix - {model_name}')
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.tight_layout()
    plt.show()


def plotRocCurve(y_true, y_pred_proba, model_name):
    """
     ROC curve plot function
    """
    fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
    auc_score = roc_auc_score(y_true, y_pred_proba)

    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, linewidth=2, label=f'{model_name} (AUC = {auc_score:.3f})')
    plt.plot([0, 1], [0, 1], 'k--', label='Random Classifier')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC Curve - {model_name}')
    plt.legend(loc='lower right')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


def plotPrecisionRecallCurve(y_true, y_pred_proba, model_name):
    """
     Precision-Recall curve plot function
    """
    precision, recall, _ = precision_recall_curve(y_true, y_pred_proba)
    ap_score = average_precision_score(y_true, y_pred_proba)

    plt.figure(figsize=(8, 6))
    plt.plot(recall, precision, linewidth=2, label=f'AP = {ap_score:.3f}')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title(f'Precision-Recall Curve - {model_name}')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.legend(loc='upper right')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()