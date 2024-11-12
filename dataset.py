import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

def create_dataloader(data_dir, batch_size=16, is_training=True):
    if is_training:
        # Transformare pentru augmentarea datelor
        transform = transforms.Compose([
            transforms.Grayscale(num_output_channels=1),
            transforms.Resize((64, 64)),
            transforms.RandomApply([transforms.ColorJitter(brightness=0.5, contrast=0.5)], p=0.3),
            transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),  # Mică translație
            transforms.ToTensor(),
        ])
    else:
        # Transformare fără augmentare pentru datele de testare
        transform = transforms.Compose([
            transforms.Grayscale(num_output_channels=1),
            transforms.Resize((64, 64)),
            transforms.ToTensor(),
        ])

    dataset = datasets.ImageFolder(root=data_dir, transform=transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    return dataloader
