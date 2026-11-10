import os
from typing import Tuple, Dict, List
import torch
from torch.utils.data import DataLoader, Dataset, random_split, Subset
from torchvision import datasets, transforms
from sklearn.model_selection import train_test_split
from PIL import Image

# ImageNet normalization stats
NORM_MEAN = [0.485, 0.456, 0.406]
NORM_STD = [0.229, 0.224, 0.225]

def get_transforms(img_size: int = 224) -> Dict[str, transforms.Compose]:
    """
    Returns data transforms for training and validation/testing.
    Applies medical-appropriate data augmentation for MRI scans.
    """
    train_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=15),
        transforms.ColorJitter(brightness=0.15, contrast=0.15),
        transforms.ToTensor(),
        transforms.Normalize(mean=NORM_MEAN, std=NORM_STD)
    ])

    val_test_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=NORM_MEAN, std=NORM_STD)
    ])

    return {
        'train': train_transform,
        'val': val_test_transform,
        'test': val_test_transform
    }

class TransformedSubset(Dataset):
    """
    Subset wrapper that applies a specific transform.
    """
    def __init__(self, subset: Subset, transform=None):
        self.subset = subset
        self.transform = transform

    def __getitem__(self, idx):
        x, y = self.subset[idx]
        if self.transform is not None:
            # If x is a PIL Image
            if isinstance(x, Image.Image):
                x = self.transform(x)
        return x, y

    def __len__(self):
        return len(self.subset)

def build_dataloaders(
    data_dir: str,
    batch_size: int = 32,
    img_size: int = 224,
    num_workers: int = 2,
    val_split: float = 0.15,
    test_split: float = 0.15,
    seed: int = 42
) -> Tuple[DataLoader, DataLoader, DataLoader, List[str]]:
    """
    Builds Train, Validation, and Test DataLoaders.
    Handles both pre-split ('Training'/'Testing') and unified directory formats.
    """
    tfs = get_transforms(img_size)

    # Check if dataset has predefined Training and Testing folders
    train_dir = os.path.join(data_dir, 'Training')
    test_dir = os.path.join(data_dir, 'Testing')

    if os.path.exists(train_dir) and os.path.exists(test_dir):
        # Predefined train/test directories
        full_train_dataset = datasets.ImageFolder(train_dir)
        classes = full_train_dataset.classes

        # Split full_train into train and validation sets
        train_indices, val_indices = train_test_split(
            list(range(len(full_train_dataset))),
            test_size=val_split,
            stratify=full_train_dataset.targets,
            random_state=seed
        )

        train_subset = Subset(full_train_dataset, train_indices)
        val_subset = Subset(full_train_dataset, val_indices)

        train_ds = TransformedSubset(train_subset, transform=tfs['train'])
        val_ds = TransformedSubset(val_subset, transform=tfs['val'])

        test_dataset_raw = datasets.ImageFolder(test_dir)
        test_ds = TransformedSubset(test_dataset_raw, transform=tfs['test'])

    else:
        # Single directory: split into Train, Val, and Test
        raw_dataset = datasets.ImageFolder(data_dir)
        classes = raw_dataset.classes
        targets = raw_dataset.targets

        train_idx, temp_idx = train_test_split(
            list(range(len(raw_dataset))),
            test_size=(val_split + test_split),
            stratify=targets,
            random_state=seed
        )
        temp_targets = [targets[i] for i in temp_idx]
        val_ratio = val_split / (val_split + test_split)
        val_idx, test_idx = train_test_split(
            temp_idx,
            test_size=(1.0 - val_ratio),
            stratify=temp_targets,
            random_state=seed
        )

        train_ds = TransformedSubset(Subset(raw_dataset, train_idx), transform=tfs['train'])
        val_ds = TransformedSubset(Subset(raw_dataset, val_idx), transform=tfs['val'])
        test_ds = TransformedSubset(Subset(raw_dataset, test_idx), transform=tfs['test'])

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True
    )
    test_loader = DataLoader(
        test_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True
    )

    return train_loader, val_loader, test_loader, classes
