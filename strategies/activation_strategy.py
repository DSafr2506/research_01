import torch
import torch.nn as nn

class ActivationStrategy:
    def get(self):
        raise NotImplementedError()

class ReLUActivation(ActivationStrategy):
    def get(self):
        return nn.ReLU()

class GELUActivation(ActivationStrategy):
    def get(self):
        return nn.GELU()

class SwishActivation(ActivationStrategy):
    def get(self):
        return lambda x: x * torch.sigmoid(x)
