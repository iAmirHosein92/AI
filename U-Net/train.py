# train.py

import torch
from model import UNet
from dataset import LungSegmentationDataset
from torch.utils.data import DataLoader
from loss import dice_bce_loss
from utils import evaluate_sample
import albumentations as A
from albumentations.pytorch import ToTensorV2
import os
from PIL import Image
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
image_dir = './images'
mask_dir = './masks'
batch_size = 4
num_epochs = 25
lr = 1e-4

# Transformations
train_transforms = A.Compose([
    A.Resize(256, 256),
    A.HorizontalFlip(p=0.5),
    A.RandomRotate90(p=0.5),
    A.Normalize(mean=(0.5,0.5, 0.5), std=(0.5,0.5, 0.5)),
    ToTensorV2()
])

# Dataset & DataLoader
dataset = LungSegmentationDataset(image_dir, mask_dir, transform=train_transforms)
loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# Model & Optimizer & Loss
model = UNet().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=lr)
criterion = dice_bce_loss

# Training Loop
for epoch in range(num_epochs):
    model.train()
    epoch_loss = 0
    for images, masks in loader:
        masks = masks.unsqueeze(1)
        images, masks = images.to(device), masks.to(device)

        preds = model(images)
        loss = criterion(preds, masks)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()

    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss / len(loader):.4f}")

# Save model
torch.save(model.state_dict(), "unet_model.pth")