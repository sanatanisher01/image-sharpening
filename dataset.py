import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torchvision.transforms as transforms
import os
import cv2
import numpy as np

class ImageSharpeningDataset(Dataset):
    def __init__(self, data_dir, transform=None, crop_size=256):
        self.data_dir = data_dir
        self.transform = transform
        self.crop_size = crop_size
        self.image_files = [f for f in os.listdir(data_dir) if f.endswith('.png')]
        
    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        img_path = os.path.join(self.data_dir, self.image_files[idx])
        
        # Load high-resolution image
        hr_image = Image.open(img_path).convert('RGB')
        
        # Random crop for training efficiency
        if self.crop_size:
            hr_image = self._random_crop(hr_image, self.crop_size)
        
        # Create degraded version (simulating video conferencing conditions)
        lr_image = self._create_degraded_image(hr_image)
        
        if self.transform:
            hr_image = self.transform(hr_image)
            lr_image = self.transform(lr_image)
            
        return lr_image, hr_image
    
    def _random_crop(self, image, size):
        w, h = image.size
        if w < size or h < size:
            image = image.resize((max(size, w), max(size, h)), Image.BICUBIC)
            w, h = image.size
        
        x = np.random.randint(0, w - size + 1)
        y = np.random.randint(0, h - size + 1)
        return image.crop((x, y, x + size, y + size))
    
    def _create_degraded_image(self, hr_image):
        """Create blurred version for sharpening"""
        img_np = np.array(hr_image)
        
        # Apply Gaussian blur for actual blur degradation
        kernel_size = np.random.choice([5, 7, 9])
        sigma = np.random.uniform(1.0, 2.5)
        blurred = cv2.GaussianBlur(img_np, (kernel_size, kernel_size), sigma)
        
        return Image.fromarray(blurred)

def get_dataloaders(train_dir, val_dir, batch_size=8, crop_size=256):
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])
    
    train_dataset = ImageSharpeningDataset(train_dir, transform, crop_size)
    val_dataset = ImageSharpeningDataset(val_dir, transform, None)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=2)
    
    return train_loader, val_loader