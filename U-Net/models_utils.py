# model_utils.py

import torch

def save_model(model, path):
    """
    ذخیره فقط state_dict مدل برای استفاده ایمن و مستقل
    """
    torch.save(model.state_dict(), path)
    print(f"✅ Model saved to: {path}")


def load_model(model_class, path, device):
    """
    بارگذاری مدل ذخیره‌شده به‌صورت state_dict
    
    - model_class: کلاسی که ساختار مدل رو تعریف می‌کنه
    - path: مسیر فایل .pth
    - device: 'cpu' یا 'cuda'
    """
    model = model_class().to(device)
    model.load_state_dict(torch.load(path, map_location=device))
    model.eval()
    print(f"✅ Model loaded from: {path}")
    return model