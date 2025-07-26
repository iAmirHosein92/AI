import numpy as np
from PIL import Image
import io
import torch
import cv2
import torchvision.transforms as T

def preprocess_image(uploaded_file):
    image = Image.open(uploaded_file).convert("RGB")
    image = image.resize((256, 256))
    transform = T.Compose([
        T.ToTensor(),
        T.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
    ])
    return transform(image).unsqueeze(0)

def postprocess_mask(mask_tensor):
    mask = mask_tensor.squeeze().detach().cpu().numpy()
    mask = (mask > 0.5).astype(np.uint8) * 255
    return Image.fromarray(mask)