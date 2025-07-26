import torch
from unet import UNet  # فرض بر اینکه UNet رو جدا تو فایل unet.py داری

def load_model(path='mask_detector_unet.pt', device='cpu'):
    model = UNet()
    model.load_state_dict(torch.load(path, map_location=device))
    model.to(device)
    model.eval()
    return model