import torch
from utils.metrics import Metrics
from typing import Dict, Tuple, Optional

class Experiment:
    def __init__(self, model, optimizer, criterion):
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.metrics = Metrics()
        self.best_val_accuracy = 0.0
        self.best_model_state = None

    def run_epoch(self, dataloader, train=True):
        self.model.train() if train else self.model.eval()
        total_loss = 0
        for x, y in dataloader:
            x = x.view(x.size(0), -1)
            preds = self.model(x)
            loss = self.criterion(preds, y)
            if train:
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
            total_loss += loss.item()
        return total_loss / len(dataloader)

    def validate(self, val_dataloader) -> Dict[str, float]:
        """
        Валидация модели
        
        Args:
            val_dataloader: DataLoader с валидационными данными
            
        Returns:
            Dict[str, float]: Словарь с метриками валидации
        """
        metrics = self.run_epoch(val_dataloader, train=False)
        
        # Сохранение лучшей модели
        if metrics['accuracy'] > self.best_val_accuracy:
            self.best_val_accuracy = metrics['accuracy']
            self.best_model_state = self.model.state_dict().copy()
            
        return metrics
    # функция ддя получения лучшего результата модели 
    def get_best_model(self) -> Optional[torch.nn.Module]:
        
        if self.best_model_state is not None:
            self.model.load_state_dict(self.best_model_state)
        return self.model
