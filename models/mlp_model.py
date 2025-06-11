import torch.nn as nn
from .base_model import BaseModel

class MLPModel(BaseModel):
    """
    Многослойный перцептрон с использованием паттерна Template Method
    """
    def __init__(self, input_size, hidden_sizes, output_size, activation):
        self.input_size = input_size
        self.hidden_sizes = hidden_sizes
        self.output_size = output_size
        self.activation = activation
        super().__init__()
    
    def _build_layers(self):
        """Построение слоев MLP"""
        layers = []
        prev_size = self.input_size
        
        for hidden_size in self.hidden_sizes:
            layers.extend([
                nn.Linear(prev_size, hidden_size),
                self.activation.get()
            ])
            prev_size = hidden_size
            
        layers.append(nn.Linear(prev_size, self.output_size))
        self.layers = nn.Sequential(*layers)
    
    def forward(self, x):
        """Прямой проход через сеть"""
        return self.layers(x) 