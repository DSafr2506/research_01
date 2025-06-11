import os
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import numpy as np
from typing import Dict, List, Tuple

class TinyImageNetDataset(Dataset):
    
    
    def __init__(self, root_dir: str, split: str = 'train', transform=None):
       
        self.root_dir = root_dir
        self.split = split
        self.transform = transform or transforms.Compose([
            transforms.Resize((64, 64)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                              std=[0.229, 0.224, 0.225])
        ])
        
        self.classes = self._get_classes()
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}
        self.images, self.labels = self._load_data()
    
    def _get_classes(self) -> List[str]:
        
        if self.split == 'train':
            classes_file = os.path.join(self.root_dir, 'wnids.txt')
            with open(classes_file, 'r') as f:
                return [line.strip() for line in f.readlines()]
        else:
            val_annotations = os.path.join(self.root_dir, 'val', 'val_annotations.txt')
            with open(val_annotations, 'r') as f:
                return sorted(list(set([line.split('\t')[1] for line in f.readlines()])))
    
    def _load_data(self) -> Tuple[List[str], List[int]]:
        
        images = []
        labels = []
        
        if self.split == 'train':
            for class_name in self.classes:
                class_dir = os.path.join(self.root_dir, 'train', class_name, 'images')
                for img_name in os.listdir(class_dir):
                    if img_name.endswith('.JPEG'):
                        images.append(os.path.join(class_dir, img_name))
                        labels.append(self.class_to_idx[class_name])
        else:
            val_dir = os.path.join(self.root_dir, 'val', 'images')
            annotations_file = os.path.join(self.root_dir, 'val', 'val_annotations.txt')
            
            with open(annotations_file, 'r') as f:
                for line in f:
                    img_name, class_name = line.strip().split('\t')[:2]
                    images.append(os.path.join(val_dir, img_name))
                    labels.append(self.class_to_idx[class_name])
        
        return images, labels
    
    def __len__(self) -> int:
        return len(self.images)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        img_path = self.images[idx]
        label = self.labels[idx]
        
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
            
        return image, label

class TinyImageNetFactory:
    
    
    def __init__(self, root_dir: str):
        self.root_dir = root_dir
        self.transform = transforms.Compose([
            transforms.Resize((64, 64)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                              std=[0.229, 0.224, 0.225])
        ])
    
    def create_dataset(self, split: str = 'train') -> TinyImageNetDataset:
       
        return TinyImageNetDataset(
            root_dir=self.root_dir,
            split=split,
            transform=self.transform
        )
    
    def create_dataloader(self, dataset: TinyImageNetDataset, 
                         batch_size: int, 
                         shuffle: bool = True,
                         num_workers: int = 4) -> DataLoader:
        
        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=True
        ) 