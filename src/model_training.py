import logging

import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.neural_network import MLPClassifier

from config import config

logger = logging.getLogger(__name__)


class ModelTrainer:
    def __init__(self):
        self.trained_models = {}

    def trainRandomForest(self, X_train, y_train):
        """Train Random Forest with configuration"""
        logger.info("Training Random Forest...")

        rf_model = RandomForestClassifier(
            random_state=config.data_config.random_state,
            n_jobs=-1,
            class_weight='balanced'
        )

        param_grid = {
            'n_estimators': [100, 200, 300],
            'max_depth': [10, 20, None],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4],
            'class_weight': ['balanced']
        }

        search = GridSearchCV(
            rf_model, param_grid, cv=5, scoring='f1', n_jobs=-1, verbose=1
        )

        search.fit(X_train, y_train)
        best_model = search.best_estimator_
        self.trained_models['random_forest'] = best_model

        logger.info(f"Random Forest best params: {search.best_params_}")
        logger.info(f"Random Forest best score: {search.best_score_:.4f}")

        return best_model


    def trainSVM(self, X_train, y_train):
        """Train SVM with proper handling for imbalanced data"""
        try:
            from sklearn.svm import SVC

            logger.info("Training SVM with balanced class weights...")

            # Create SVM with balanced class weights and probability
            svm_model = SVC(
                class_weight='balanced',  # Handle imbalanced data
                probability=True,  # Enable probability predictions
                random_state=42,  # Fixed random state
                kernel='rbf',  # Good default kernel
                C=1.0,  # Single float value, NOT a list
                gamma='scale'  # Kernel coefficient
            )

            svm_model.fit(X_train, y_train)

            self.trained_models['svm'] = svm_model
            logger.info("SVM training completed with balanced class weights")
            return svm_model

        except Exception as e:
            logger.error(f"Error training SVM: {str(e)}")
            raise

    def trainNeuralNetwork(self, X_train, y_train):
        """Train Neural Network with configuration"""
        logger.info("Training Neural Network...")

        nn_model = MLPClassifier(
            random_state=config.data_config.random_state,
            early_stopping=True,
            validation_fraction=0.1,
            n_iter_no_change=10,
            max_iter=2000
        )

        param_grid = {
            'hidden_layer_sizes': [(50,), (100,), (50, 50), (100, 50)],
            'activation': ['relu', 'tanh'],
            'alpha': [0.0001, 0.001, 0.01],
            'learning_rate': ['constant', 'adaptive']
        }

        search = GridSearchCV(
            nn_model, param_grid, cv=5, scoring='f1', n_jobs=-1, verbose=1
        )

        search.fit(X_train, y_train)
        best_model = search.best_estimator_
        self.trained_models['neural_network'] = best_model

        logger.info(f"Neural Network best params: {search.best_params_}")
        logger.info(f"Neural Network best score: {search.best_score_:.4f}")

        return best_model

    def trainXGboast(self, X_train, y_train):
        """Train XGBoost classifier"""
        logger.info("Training XGBoost...")

        try:
            from xgboost import XGBClassifier

            xgb_model = XGBClassifier(
                random_state=config.data_config.random_state,
                n_jobs=-1,
                eval_metric='logloss',
                # use_label_encoder=False
            )

            param_grid = {
                'n_estimators': [100, 200, 300],
                'max_depth': [3, 5, 7],
                'learning_rate': [0.01, 0.1, 0.2],
                'subsample': [0.8, 0.9, 1.0],
                'scale_pos_weight': [1, 2, 5]  # Handle imbalance
            }

            search = GridSearchCV(
                xgb_model, param_grid, cv=5, scoring='f1', n_jobs=-1, verbose=1
            )

            search.fit(X_train, y_train)
            best_model = search.best_estimator_
            self.trained_models['xgboost'] = best_model

            logger.info(f"XGBoost best params: {search.best_params_}")
            logger.info(f"XGBoost best score: {search.best_score_:.4f}")

            return best_model

        except ImportError:
            logger.warning("XGBoost not available. Please install with: pip install xgboost")
            return None

    def save_models(self, directory='models'):
        """Save all trained models"""
        import os
        os.makedirs(directory, exist_ok=True)

        for name, model in self.trained_models.items():
            if model is not None:  # Only save if model exists
                joblib.dump(model, f'{directory}/{name}_model.pkl')
                logger.info(f"Saved {name} model")

    def load_models(self, directory='models'):
        """Load trained models"""
        for name in ['random_forest', 'svm', 'neural_network', 'xgboost', 'ensemble']:
            try:
                model = joblib.load(f'{directory}/{name}_model.pkl')
                self.trained_models[name] = model
                logger.info(f"Loaded {name} model")
            except FileNotFoundError:
                logger.warning(f"Model {name} not found")



def trainRandomForest(X_train, y_train):
    trainer = ModelTrainer()
    return trainer.trainRandomForest(X_train, y_train)


def trainSVM(X_train, y_train):
    trainer = ModelTrainer()
    return trainer.trainSVM(X_train, y_train)


def trainNeuralNetwork(X_train, y_train):
    trainer = ModelTrainer()
    return trainer.trainNeuralNetwork(X_train, y_train)


def trainXGboast(X_train, y_train):
    trainer = ModelTrainer()
    return trainer.trainXGboast(X_train, y_train)


