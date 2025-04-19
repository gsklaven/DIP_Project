import numpy as np
from typing import Dict
from hist_utils import calculate_hist_of_img

def greedy(img_array: np.ndarray, hist_ref: Dict) -> np.ndarray:
    total_pixels = img_array.size
    hist = calculate_hist_of_img(img_array, return_normalized=True)

    pixels_for_g = total_pixels / len(hist_ref)

    #for key, value in hist_ref.items()


    return img_array