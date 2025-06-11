from abc import ABC, abstractmethod
import torch
from typing import Dict, Any

class TrainingStrategy(ABC):

    
    @abstractmethod
    def train_epoch(self, model, dataloader, criterion, optimizer) -> Dict[str, float]:
        pass

class StandardTrainingStrategy(TrainingStrategy):
    
    
    def train_epoch(self, model, dataloader, criterion, optimizer) -> Dict[str, float]:
        model.train()
        total_loss = 0
        total_correct = 0
        total_samples = 0
        
        for x, y in dataloader:
            x = x.view(x.size(0), -1)
            preds = model(x)
            loss = criterion(preds, y)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            _, predicted = torch.max(preds, 1)
            total_correct += (predicted == y).sum().item()
            total_samples += y.size(0)
        
        return {
            'loss': total_loss / len(dataloader),
            'accuracy': total_correct / total_samples
        }

class Trainer:
    
    def __init__(self, model, criterion, optimizer, training_strategy: TrainingStrategy):
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.training_strategy = training_strategy
    
    def train(self, dataloader, num_epochs: int) -> Dict[str, Any]:
        history = {
            'loss': [],
            'accuracy': []
        }
        
        for epoch in range(num_epochs):
            metrics = self.training_strategy.train_epoch(
                self.model,
                dataloader,
                self.criterion,
                self.optimizer
            )
            
            history['loss'].append(metrics['loss'])
            history['accuracy'].append(metrics['accuracy'])
            
            print(f"Epoch {epoch+1}/{num_epochs}")
            print(f"Loss: {metrics['loss']:.4f}, Accuracy: {metrics['accuracy']:.4f}")
        
        return history 