import albumentations as A
from albumentations.pytorch import ToTensorV2

def preprocess_image(image):
    transform = A.Compose([
        A.Resize(256, 256),
        A.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
        ToTensorV2()
    ])
    image = transform(image=np.array(image))["image"]
    return image