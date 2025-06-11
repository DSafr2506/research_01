from dataclasses import dataclass
from typing import List, Optional
import yaml
import os

@dataclass
class ModelConfig:
    input_size: int
    hidden_sizes: List[int]
    output_size: int
    activation: str

@dataclass
class OptimizerConfig:
    type: str
    learning_rate: float
    weight_decay: float
    betas: tuple
    eps: float
    amsgrad: bool
    momentum: float

@dataclass
class TrainingConfig:
    batch_size: int
    num_epochs: int
    validation_split: float

class Config:
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        if not hasattr(self, 'initialized'):
            self.model_config = None
            self.optimizer_config = None
            self.training_config = None
            self.initialized = True
    #загружаем конфигурацию из YAML
    def load_from_yaml(self, path: str):
        if not os.path.exists(path):
            raise FileNotFoundError(f"Конфигурационный файл не найден: {path}")
            
        with open(path, 'r') as f:
            config = yaml.safe_load(f)
            
        self.model_config = ModelConfig(**config['model'])
        self.optimizer_config = OptimizerConfig(**config['optimizer'])
        self.training_config = TrainingConfig(**config['training'])
    
    def get_model_config(self) -> ModelConfig:
        return self.model_config
    
    def get_optimizer_config(self) -> OptimizerConfig:
        return self.optimizer_config
    
    def get_training_config(self) -> TrainingConfig:
        return self.training_config 