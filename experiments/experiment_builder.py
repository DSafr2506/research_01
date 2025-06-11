from experiments.experiment import Experiment

class ExperimentBuilder:
    def __init__(self):
        self.model = None
        self.optimizer = None
        self.criterion = None

    def with_model(self, model):
        self.model = model
        return self

    def with_optimizer(self, optimizer):
        self.optimizer = optimizer
        return self

    def with_loss(self, criterion):
        self.criterion = criterion
        return self

    def build(self):
        return Experiment(self.model, self.optimizer, self.criterion)
