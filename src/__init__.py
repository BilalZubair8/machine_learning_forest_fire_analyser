
from .data_preprocessing import (
    loadAndPreProcessData,
    prepareFeaturesTarget,
    splitAndScaleData,
    DataPreProcessor
)

from .feature_engineering import (
    FeatureEngineer
)

from .model_training import (
    trainRandomForest,
    trainSVM,
    trainNeuralNetwork,
    trainXGboast,
    ModelTrainer
)

from .hyperparameter_tuning import (
    HyperparameterOptimizer
)

from .evaluation import (
    evaluateModel,
    plotFeatureImportance,
    plotConfusionMatrix,
    plotRocCurve,
    plotPrecisionRecallCurve,
    ModelEvaluator
)

from .utils import (
    saveModel,
    loadModel,
    predictNewData,
    createSamplePredictionData
)

# Package metadata
__version__ = "1.0.0"
__author__ = "Forest Fire Classification Team"
__description__ = "Machine Learning pipeline for forest fire severity classification"

# Export all public classes and functions
__all__ = [
    # Data Preprocessing
    'loadAndPreProcessData',
    'prepareFeaturesTarget',
    'splitAndScaleData',
    'DataPreProcessor',

    # Feature Engineering
    'FeatureEngineer',

    # Model Training
    'trainRandomForest',
    'trainSVM',
    'trainNeuralNetwork',
    'trainXGboast',
    # 'train_ensemble',
    'ModelTrainer',

    # Hyperparameter Tuning
    'HyperparameterOptimizer',

    # Evaluation
    'evaluateModel',
    'plotFeatureImportance',
    'plotConfusionMatrix',
    'plotRocCurve',
    'plotPrecisionRecallCurve',
    'ModelEvaluator',

    # Utilities
    'saveModel',
    'loadModel',
    'predictNewData',
    'createSamplePredictionData'
]


# Package initialization
def get_version():
    """Return package version"""
    return __version__


def get_available_modules():
    """Return list of available modules in the package"""
    return {
        'data_preprocessing': 'Data loading, preprocessing, and splitting',
        'feature_engineering': 'Advanced feature creation and transformation',
        'model_training': 'Model training with hyperparameter optimization',
        'hyperparameter_tuning': 'Advanced hyperparameter search strategies',
        'evaluation': 'Comprehensive model evaluation and visualization',
        'utils': 'Utility functions for model persistence and prediction'
    }

# Package configuration
class PackageConfig:
    """Package-level configuration"""
    SUPPORTED_MODELS = [
        'random_forest',
        'svm',
        'neural_network',
        'xgboost',
    ]

    SUPPORTED_EVALUATION_METRICS = [
        'accuracy',
        'precision',
        'recall',
        'f1',
        'roc_auc',
        'average_precision',
        'specificity',
        'balanced_accuracy'
    ]

    @classmethod
    def get_supported_models(cls):
        return cls.SUPPORTED_MODELS

    @classmethod
    def get_supported_metrics(cls):
        return cls.SUPPORTED_EVALUATION_METRICS


print(f"Forest Fire Classification Package v{__version__}")
print("Enhanced with feature engineering and model optimization")
print("Supports: Random Forest, SVM, Neural Network, XGBoost")