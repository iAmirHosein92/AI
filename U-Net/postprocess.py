import numpy as np
import cv2
from skimage.measure import label, regionprops

def postprocess_mask(mask, min_area=300):
    labeled = label(mask)
    cleaned = np.zeros_like(mask)

    for region in regionprops(labeled):
        if region.area >= min_area:
            for coord in region.coords:
                cleaned[coord[0], coord[1]] = 1

    cleaned = cleaned.astype(np.uint8)
    kernel = np.ones((5, 5), np.uint8)
    cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, kernel)

    return cleaned