import logging

from sklearn.metrics import make_scorer, f1_score
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV

from config import config

logger = logging.getLogger(__name__)


class HyperparameterOptimizer:
    def __init__(self):
        self.scoring = make_scorer(f1_score)
        self.cv = config.eval_config.cv_folds

    def optRandomForest(self, model, X_train, y_train):
        """Optimize Random Forest with search"""
        param_dist = {
            'n_estimators': [100, 200, 300, 500],
            'max_depth': [10, 20, 30, None],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4],
            'max_features': ['sqrt', 'log2', None],
            'bootstrap': [True, False],
            'class_weight': ['balanced', 'balanced_subsample', None]
        }

        search = RandomizedSearchCV(
            model, param_dist, n_iter=50,  # More iterations for better search
            cv=self.cv, scoring=self.scoring,
            n_jobs=-1, random_state=config.data_config.random_state,
            verbose=1
        )

        search.fit(X_train, y_train)
        logger.info(f"Random Forest best params: {search.best_params_}")
        logger.info(f"Random Forest best score: {search.best_score_:.4f}")

        return search.best_estimator_

    def optSVM(self, model, X_train, y_train):
        """Optimize SVM with Bayesian optimization alternative"""
        from skopt import BayesSearchCV
        from skopt.space import Real, Categorical

        search_space = {
            'C': Real(1e-3, 1e3, prior='log-uniform'),
            'gamma': Real(1e-4, 1e1, prior='log-uniform'),
            'kernel': Categorical(['rbf', 'linear', 'poly']),
            'class_weight': Categorical(['balanced', None])
        }

        search = BayesSearchCV(
            model, search_space, n_iter=50,
            cv=self.cv, scoring=self.scoring,
            n_jobs=-1, random_state=config.data_config.random_state,
            verbose=1
        )

        search.fit(X_train, y_train)
        logger.info(f"SVM best params: {search.best_params_}")
        logger.info(f"SVM best score: {search.best_score_:.4f}")

        return search.best_estimator_

    def optNeuralNetwork(self, model, X_train, y_train):
        """Optimize Neural Network with search"""
        param_grid = {
            'hidden_layer_sizes': [(50,), (100,), (50, 50), (100, 50), (100, 100, 50)],
            'activation': ['relu', 'tanh', 'logistic'],
            'alpha': [0.0001, 0.001, 0.01, 0.1],
            'learning_rate': ['constant', 'adaptive'],
            'learning_rate_init': [0.001, 0.01, 0.1],
            'early_stopping': [True, False],
            'solver': ['adam', 'sgd']
        }

        search = GridSearchCV(
            model, param_grid, cv=self.cv, scoring=self.scoring,
            n_jobs=-1, verbose=1
        )

        search.fit(X_train, y_train)
        logger.info(f"Neural Network best params: {search.best_params_}")
        logger.info(f"Neural Network best score: {search.best_score_:.4f}")

        return search.best_estimator_