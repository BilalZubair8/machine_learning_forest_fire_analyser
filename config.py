from dataclasses import dataclass
from typing import List, Dict, Any

import yaml


@dataclass
class DataConfig:
    file_path: str
    test_size: float
    val_size: float
    random_state: int


@dataclass
class ModelConfig:
    random_forest: Dict[str, Any]
    svm: Dict[str, Any]
    neural_network: Dict[str, Any]


@dataclass
class EvaluationConfig:
    cv_folds: int
    scoring_metric: str
    metrics: List[str]


class Config:
    def __init__(self, config_path: str = "config/parameters.yaml"):
        self.config_path = config_path
        self._load_config()

    def _load_config(self):
        with open(self.config_path, 'r') as file:
            self.data = yaml.safe_load(file)

        self.data_config = DataConfig(**self.data['data'])
        self.model_config = ModelConfig(**self.data['models'])
        self.eval_config = EvaluationConfig(**self.data['evaluation'])

    def get(self, key: str, default=None):
        return self.data.get(key, default)


# Global config instance
config = Config()