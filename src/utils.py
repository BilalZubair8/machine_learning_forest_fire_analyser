import logging
from datetime import datetime
from typing import Dict, Any
import joblib
import numpy as np
import pandas as pd
import yaml

logger = logging.getLogger(__name__)


def saveModel(model_artifacts: Dict[str, Any], filepath: str):
    """
    Save model artifacts to file with metadata
    """
    # Add metadata
    model_artifacts['metadata'] = {
        'saved_at': datetime.now().isoformat(),
        'version': '1.0.0',
        'model_type': model_artifacts.get('model_name', 'unknown'),
        'feature_count': len(model_artifacts.get('feature_columns', [])),
        'performance': model_artifacts.get('performance_metrics', {}),
        'encoding_mappings': model_artifacts.get('encoding_mappings', {})
    }

    joblib.dump(model_artifacts, filepath)
    logger.info(f"Model saved to {filepath} with metadata")

    # Also save a human-readable summary
    summary = generateModelSummary(model_artifacts)

    summary_path = f"reports/best_forest_fire_model_summary.txt"
    with open(summary_path, 'w') as f:
        f.write(summary)
    logger.info(f"Model summary saved to {summary_path}")



def loadModel(filepath: str) -> Dict[str, Any]:
    """
    Load model artifacts from file with validation
    """
    try:
        model_artifacts = joblib.load(filepath)

        # Validate loaded artifacts
        required_keys = ['model', 'feature_columns']
        for key in required_keys:
            if key not in model_artifacts:
                raise ValueError(f"Missing required key in model artifacts: {key}")

        logger.info(f"Model loaded from {filepath}")
        logger.info(f"Model type: {model_artifacts.get('model_name', 'Unknown')}")
        logger.info(f"Features: {len(model_artifacts['feature_columns'])}")

        return model_artifacts

    except Exception as e:
        logger.error(f"Error loading model from {filepath}: {str(e)}")
        raise


def predictNewData(new_data, model_artifacts):
    """
    Make predictions on new data using the saved model artifacts with consistent encoding
    """
    try:
        model = model_artifacts['model']
        scaler = model_artifacts['scaler']
        feature_columns = model_artifacts['feature_columns']
        encoding_mappings = model_artifacts.get('encoding_mappings', {})

        # Create a copy to avoid modifying original data
        processed_data = new_data.copy()

        # Apply consistent encoding using saved mappings
        if 'month' in processed_data.columns and 'month_encoded' not in processed_data.columns:
            month_mapping = encoding_mappings.get('month_mapping', {
                'jan': 1, 'feb': 2, 'mar': 3, 'apr': 4, 'may': 5, 'jun': 6,
                'jul': 7, 'aug': 8, 'sep': 9, 'oct': 10, 'nov': 11, 'dec': 12
            })
            processed_data['month_encoded'] = processed_data['month'].map(month_mapping)

        if 'day' in processed_data.columns and 'day_encoded' not in processed_data.columns:
            day_mapping = encoding_mappings.get('day_mapping', {
                'mon': 1, 'tue': 2, 'wed': 3, 'thu': 4,
                'fri': 5, 'sat': 6, 'sun': 7
            })
            processed_data['day_encoded'] = processed_data['day'].map(day_mapping)

        # Select only the features used during training
        if feature_columns:
            missing_cols = set(feature_columns) - set(processed_data.columns)
            if missing_cols:
                raise ValueError(f"Missing columns in new data: {missing_cols}")
            processed_data = processed_data[feature_columns].copy()

        # Scale the features
        if scaler is not None:
            processed_data = scaler.transform(processed_data)

        # Make predictions
        predictions = model.predict(processed_data)

        # Get full probability arrays, not just class 1 probabilities
        if hasattr(model, 'predict_proba'):
            probabilities = model.predict_proba(processed_data)
        else:
            probabilities = None

        return predictions, probabilities

    except Exception as e:
        logger.error(f"Error making predictions: {str(e)}")
        raise



def createSamplePredictionData() -> pd.DataFrame:
    """
    Create sample data for prediction demonstration
    """
    sample_data = {
        'X': [7, 5, 8, 3, 6],
        'Y': [5, 4, 6, 2, 3],
        'FFMC': [86.2, 90.6, 91.7, 85.0, 92.3],
        'DMC': [26.2, 35.4, 33.3, 120.5, 15.8],
        'DC': [94.3, 669.1, 77.5, 700.2, 300.4],
        'ISI': [5.1, 6.7, 9.0, 15.2, 3.4],
        'temp': [8.2, 18.0, 8.3, 25.6, 12.3],
        'RH': [51, 33, 97, 25, 75],
        'wind': [6.7, 0.9, 4.0, 8.2, 2.5],
        'rain': [0, 0, 0.2, 0, 1.2],
        'month': ['mar', 'oct', 'mar', 'aug', 'feb'],
        'day': ['fri', 'tue', 'fri', 'sun', 'mon']
    }

    df = pd.DataFrame(sample_data)
    logger.info("Created sample prediction data with 5 diverse examples")
    return df


def generateModelSummary(model_artifacts: Dict[str, Any]) -> str:
    """
    Generate model summary
    """
    model = model_artifacts['model']
    metrics = model_artifacts.get('performance_metrics', {})
    feature_columns = model_artifacts.get('feature_columns', [])
    encoding_mappings = model_artifacts.get('encoding_mappings', {})

    summary = f"""
FOREST FIRE ANALYSIS MODEL SUMMARY
=======================================

Model Information:
-----------------
Model Type: {model_artifacts.get('model_name', 'Unknown')}
Model Class: {model.__class__.__name__}
Features Used: {len(feature_columns)}
Scaler Used: {'Yes' if model_artifacts.get('scaler') else 'No'}
Feature Engineering: {'Yes' if model_artifacts.get('feature_engineer') else 'No'}

Performance Metrics:
------------------
{chr(10).join([f'{k}: {v:.4f}' for k, v in metrics.items() if isinstance(v, (int, float))])}

Feature Columns:
---------------
{chr(10).join([f'{i + 1}. {col}' for i, col in enumerate(feature_columns)])}

Encoding Mappings:
-----------------
Month: {encoding_mappings.get('month_mapping', 'Not available')}
Day: {encoding_mappings.get('day_mapping', 'Not available')}

Model Parameters:
----------------
{str(model.get_params()) if hasattr(model, 'get_params') else 'Not available'}

Metadata:
---------
Saved At: {model_artifacts.get('metadata', {}).get('saved_at', 'Unknown')}
Version: {model_artifacts.get('metadata', {}).get('version', 'Unknown')}
"""
    return summary


def saveConfig(config_dict: Dict[str, Any], filepath: str):
    """
    Save configuration to YAML file
    """
    with open(filepath, 'w') as f:
        yaml.dump(config_dict, f, default_flow_style=False)
    logger.info(f"Configuration saved to {filepath}")


def loadConfig(filepath: str) -> Dict[str, Any]:
    """
    Load configuration from YAML file
    """
    with open(filepath, 'r') as f:
        config_dict = yaml.safe_load(f)
    logger.info(f"Configuration loaded from {filepath}")
    return config_dict


def createPredictionReport(predictions: np.ndarray,
                             probabilities: np.ndarray,
                             input_data: pd.DataFrame) -> pd.DataFrame:
    """
    Create a prediction report
    """
    report_data = input_data.copy()
    report_data['prediction'] = predictions
    report_data['probability_class_0'] = probabilities[:, 0]
    report_data['probability_class_1'] = probabilities[:, 1]
    report_data['confidence'] = np.max(probabilities, axis=1)
    report_data['predicted_class'] = report_data['prediction'].map(
        {0: 'Area <= 4%', 1: 'Area > 4%'}
    )
    report_data['risk_level'] = report_data['probability_class_1'].apply(
        lambda x: 'HIGH' if x > 0.7 else 'MEDIUM' if x > 0.3 else 'LOW'
    )

    return report_data



def saveModelArtifacts(model_artifacts: Dict[str, Any], filepath: str):
    """for saveModel for backward compatibility"""
    return saveModel(model_artifacts, filepath)


def loadModelArtifacts(filepath: str) -> Dict[str, Any]:
    """for loadModel for backward compatibility"""
    return loadModel(filepath)