import torch.optim as optim
from optimizers.sophie import Sophie

class OptimizerStrategy:
    def get(self, model):
        raise NotImplementedError()

class AdamOptimizer(OptimizerStrategy):
    def get(self, model):
        return optim.Adam(model.parameters(), lr=0.001)

class SGDOptimizer(OptimizerStrategy):
    def get(self, model):
        return optim.SGD(model.parameters(), lr=0.01)

class AdamWOptimizer(OptimizerStrategy):
    def get(self, model):
        return optim.AdamW(model.parameters(), lr=0.001)

class SophieOptimizer(OptimizerStrategy):
    def get(self, model):
        return Sophie(
            model.parameters(),
            lr=0.001,
            betas=(0.9, 0.999),
            eps=1e-8,
            weight_decay=0.01,
            amsgrad=True,
            momentum=0.9
        )
