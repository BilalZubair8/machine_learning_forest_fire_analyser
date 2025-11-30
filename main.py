import logging
import os
import sys
from datetime import datetime

import numpy as np

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f'forest_fire_analysis_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log',
                            encoding='utf-8'),  # Add encoding here
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# Add src to path
sys.path.append('src')

from src.data_preprocessing import DataPreProcessor
from src.feature_engineering import FeatureEngineer
from src.model_training import ModelTrainer
from src.evaluation import ModelEvaluator, plotFeatureImportance
from src.utils import saveModel, predictNewData, createSamplePredictionData
from config import config


class ForestFirePipeline:
    def __init__(self):
        self.data_processor = DataPreProcessor()
        self.feature_engineer = FeatureEngineer()
        self.model_trainer = ModelTrainer()
        self.evaluator = ModelEvaluator()
        self.best_model = None
        self.best_metrics = None

    def ensure_directories(self):
        """Create necessary directories"""
        directories = ['data', 'models', 'config', 'reports', 'plots']
        for directory in directories:
            os.makedirs(directory, exist_ok=True)
            logger.info(f"Created directory: {directory}")

    def run_pipeline(self):
        """Run the complete forest fire data analysis pipeline"""
        logger.info("_._." * 40)
        logger.info("FOREST FIRE ANALYSIS PROJECT")
        logger.info("CRISP-DM Methodology Implementation")
        logger.info("_._." * 40)

        self.ensure_directories()

        try:
            # 1. DATA PREPROCESSING & FEATURE ENGINEERING
            logger.info("\n1. DATA PREPROCESSING & FEATURE ENGINEERING PHASE")
            logger.info("_._." * 40)

            # Load and preprocess data
            df = self.data_processor.loadAndPreProcessData(config.data_config.file_path)

            # feature engineering
            df_engineered = self.feature_engineer.engineerAllFeatures(df)

            # Prepare features and target
            X, y, feature_columns = self.data_processor.prepareFeaturesTarget(df_engineered)

            # Split and scale data
            (X_train, X_val, X_test, X_train_scaled, X_val_scaled, X_test_scaled,
             y_train, y_val, y_test, scaler) = self.data_processor.splitAndScaleData(X, y)

            # 2. MODEL TRAINING
            logger.info("\n2. MODEL TRAINING PHASE")
            logger.info("_._." * 40)

            # Train multiple models
            rf_model = self.model_trainer.trainRandomForest(X_train, y_train)
            svm_model = self.model_trainer.trainSVM(X_train_scaled, y_train)
            nn_model = self.model_trainer.trainNeuralNetwork(X_train_scaled, y_train)
            xgb_model = self.model_trainer.trainXGboast(X_train, y_train)


            # Save all models
            self.model_trainer.save_models()

            # 3. MODEL EVALUATION
            logger.info("\n3. MODEL EVALUATION PHASE")
            logger.info("_._." * 40)

            models_to_evaluate = {
                'Random Forest': (rf_model, X_val, y_val, False, None),
                'SVM': (svm_model, X_val_scaled, y_val, True, scaler),
                'Neural Network': (nn_model, X_val_scaled, y_val, True, scaler),
                'XGBoost': (xgb_model, X_val, y_val, False, None),

            }

            validation_metrics = {}
            for name, (model, X_eval, y_eval, needs_scaling, sc) in models_to_evaluate.items():
                metrics = self.evaluator.comprehensiveEvalution(
                    model, X_eval, y_eval, name, needs_scaling, sc
                )
                validation_metrics[name] = metrics

            # Compare and select best model
            best_model_name, comparison_df = self.evaluator.compareModels(validation_metrics)
            self.best_model = best_model_name
            self.best_metrics = validation_metrics[best_model_name]

            # 4. FINAL TESTING WITH BEST MODEL
            logger.info("\n4. FINAL TESTING PHASE")
            logger.info("_._." * 40)

            # Determine best model configuration
            if best_model_name in ['SVM', 'Neural Network']:
                best_model_instance = self.model_trainer.trained_models[best_model_name.lower().replace(' ', '_')]
                X_test_processed = X_test_scaled
                scaling_required = True
            else:
                best_model_instance = self.model_trainer.trained_models[best_model_name.lower().replace(' ', '_')]
                X_test_processed = X_test
                scaling_required = False

            logger.info(f"Testing {best_model_name} on unseen test data...")
            test_metrics = self.evaluator.comprehensiveEvalution(
                best_model_instance, X_test_processed, y_test,
                f"Final {best_model_name} (Test Set)",
                scaling_required, scaler if scaling_required else None
            )

            # 5. FEATURE ANALYSIS & INTERPRETABILITY
            logger.info("\n5. FEATURE ANALYSIS & INTERPRETABILITY")
            logger.info("_._." * 40)

            if hasattr(best_model_instance, 'feature_importances_'):
                feature_importance_df = plotFeatureImportance(
                    best_model_instance, feature_columns, best_model_name
                )
                if feature_importance_df is not None:
                    logger.info("\nTop 10 Most Important Features:")
                    logger.info(feature_importance_df.head(10).to_string())

            # 6. MODEL DEPLOYMENT PREPARATION
            logger.info("\n6. MODEL DEPLOYMENT PREPARATION")
            logger.info("_._." * 40)

            model_artifacts = {
                'model': best_model_instance,
                'scaler': scaler if scaling_required else None,
                'feature_columns': feature_columns,
                'label_encoders': self.data_processor.label_encoders,
                'feature_engineer': self.feature_engineer,
                'model_name': best_model_name,
                'performance_metrics': test_metrics,
                'config': config.data,
                'encoding_mappings': self.feature_engineer.getEncodingMappings()
            }

            saveModel(model_artifacts, 'models/best_forest_fire_model.pkl')

            # 7. PREDICTION DEMONSTRATION & BUSINESS INSIGHTS
            logger.info("\n7. PREDICTION DEMONSTRATION & BUSINESS INSIGHTS")
            logger.info('_._.' * 40)

            self._demonstrate_predictions(model_artifacts)
            self._generate_business_insights(test_metrics)

            # 8. SAVE COMPREHENSIVE REPORT
            logger.info("\n8. COMPREHENSIVE REPORT GENERATION")
            logger.info("_._." * 40)

            self.evaluator.saveEvaluationReport()
            self._generate_executive_summary(test_metrics)

            logger.info(f"\n{'_._.' * 40}")
            logger.info("PROJECT COMPLETED SUCCESSFULLY!")
            logger.info(f"{'_._.' * 40}")
            logger.info(f"Best Model: {best_model_name}")
            logger.info(f"Test Accuracy: {test_metrics['accuracy']:.4f}")
            logger.info(f"Test F1-Score: {test_metrics['f1']:.4f}")
            logger.info(f"Test AUC-ROC: {test_metrics['roc_auc']:.4f}")
            logger.info(f"Model saved: models/best_forest_fire_model.pkl")
            logger.info(f"Evaluation report: models/evaluation_report.json")
            logger.info(f"{'_._.' * 40}")

        except Exception as e:
            logger.error(f"Error occurred in pipeline: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
            raise

    def _demonstrate_predictions(self, model_artifacts):
        """Demonstrate predictions on sample data"""
        sample_data = createSamplePredictionData()
        logger.info("Sample data for prediction:")
        logger.info(sample_data)

        # Load model and make predictions
        predictions, probabilities = predictNewData(sample_data, model_artifacts)

        logger.info("\nPrediction Results:")

        # Debug: Check what probabilities look like
        logger.info(
            f"Predictions type: {type(predictions)}, shape: {predictions.shape if hasattr(predictions, 'shape') else 'No shape'}")
        logger.info(f"Probabilities type: {type(probabilities)}")

        for i, (pred, prob) in enumerate(zip(predictions, probabilities)):
            # Handle different probability formats safely
            if isinstance(prob, (np.ndarray, list)) and len(prob) > 1:
                # If prob is [prob_class0, prob_class1]
                confidence = prob[1] if pred == 1 else prob[0]
            elif isinstance(prob, (np.ndarray, list)) and len(prob) == 1:
                # If prob is [prob_class1] only
                confidence = prob[0] if pred == 1 else (1 - prob[0])
            else:
                # If prob is a single scalar value (probability of class 1)
                confidence = float(prob) if pred == 1 else (1 - float(prob))

            risk_level = "HIGH RISK" if pred == 1 else "LOW RISK"
            logger.info(f"Sample {i + 1}: {risk_level} (Confidence: {confidence:.2%})")

    def _generate_business_insights(self, test_metrics):
        """Generate business-focused insights"""
        logger.info("\nBUSINESS INSIGHTS:")
        logger.info("-" * 30)

        if test_metrics['recall'] > 0.8:
            logger.info("[PASS] Excellent at detecting severe fires (high recall)")
        else:
            logger.info("[WARNING] May miss some severe fires - consider improving recall")

        if test_metrics['precision'] > 0.7:
            logger.info("[PASS] Good at avoiding false alarms (high precision)")
        else:
            logger.info("[WARNING] Some false alarms expected - consider resource allocation")

        if test_metrics['f1'] > 0.75:
            logger.info("[PASS] Well-balanced model for practical deployment")
        else:
            logger.info("[NEEDS WORK] Model may need further optimization")

    def _generate_executive_summary(self, test_metrics):
        """Generate executive summary"""
        summary = f"""
EXECUTIVE SUMMARY
================

Project: Forest Fire Analysis 
Objective: Predict if burned area exceeds 4%
Best Model: {self.best_model}

PERFORMANCE HIGHLIGHTS:
• Accuracy: {test_metrics['accuracy']:.1%}
• Severe Fire Detection Rate: {test_metrics['recall']:.1%}
• Prediction Reliability: {test_metrics['precision']:.1%}
• Overall Balance (F1-Score): {test_metrics['f1']:.1%}
"""
        logger.info(summary)

        # Save executive summary
        with open('reports/executive_summary.txt', 'w') as f:
            f.write(summary)


def main():
    """Main execution function"""
    pipeline = ForestFirePipeline()
    pipeline.run_pipeline()


if __name__ == "__main__":
    main()
