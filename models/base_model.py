import torch.nn as nn
from abc import ABC, abstractmethod

class BaseModel(nn.Module, ABC):
    def __init__(self):
        super().__init__()
        self._build_layers()
        
    @abstractmethod
    def _build_layers(self):   
        pass
    
    @abstractmethod
    def forward(self, x):
        pass
    
    def save(self, path):
        torch.save(self.state_dict(), path)

    def load(self, path):
        self.load_state_dict(torch.load(path)) 