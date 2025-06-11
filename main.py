from models.cnn_model import CNNModel
from optimizers.sophie import Sophie
from data.tiny_imagenet_dataset import get_dataloaders
from training.trainer import Trainer, StandardTrainingStrategy
import torch.nn as nn
import argparse
import torch

def parse_args():
    parser = argparse.ArgumentParser(description='Tiny ImageNet Training')
    parser.add_argument('--data_dir', type=str, required=True,
                      help='путь к директории с датасетом Tiny ImageNet')
    parser.add_argument('--epochs', type=int, default=100,
                      help='количество эпох для обучения')
    parser.add_argument('--batch_size', type=int, default=32,
                      help='размер батча')
    parser.add_argument('--lr', type=float, default=0.001,
                      help='скорость обучения')
    parser.add_argument('--num_workers', type=int, default=4,
                      help='количество воркеров для загрузки данных')
    return parser.parse_args()

def main():
    # Парсинг аргументов командной строки
    args = parse_args()
    
    # Проверка доступности GPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Используется устройство: {device}')
    
    # Создание модели
    model = CNNModel(num_classes=200)
    model = model.to(device)
    
    # Создание оптимизатора
    optimizer = Sophie(
        model.parameters(),
        lr=args.lr,
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=0.01,
        amsgrad=True,
        momentum=0.9
    )
    
    # Создание датасетов
    train_loader, val_loader = get_dataloaders(
        args.data_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers
    )
    
    # Создание критерия и стратегии обучения
    criterion = nn.CrossEntropyLoss()
    training_strategy = StandardTrainingStrategy()
    
    # Создание тренера и запуск обучения
    trainer = Trainer(model, criterion, optimizer, training_strategy)
    history = trainer.train(
        train_loader,
        num_epochs=args.epochs
    )

if __name__ == "__main__":
    main()
