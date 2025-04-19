from PIL import Image
import numpy as np
from typing import Dict
from hist_utils import calculate_hist_of_img

def greedy(img_array: np.ndarray, Lg: int) -> np.ndarray:
    total_pixels = img_array.size #Calculate N
    hist = calculate_hist_of_img(img_array, return_normalized=False) #Take histogram values
    print(hist)
    pixels_for_g = total_pixels / Lg
    gmin, gmax = img_array.min(), img_array.max()


    return img_array