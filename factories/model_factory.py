import torch.nn as nn

class ModelFactory:
    def create(self, activation_strategy):
        return nn.Sequential(
            nn.Linear(64*64*3, 512),
            activation_strategy.get(),
            nn.Linear(512, 200)
        )
