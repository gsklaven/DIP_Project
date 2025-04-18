import numpy as np
from typing import Dict
from hist_utils import calculate_hist_of_img
from hist_utils import apply_hist_modification_transform


def perform_hist_modification(img_array: np.ndarray, hist_ref: Dict, mode: str) -> np.ndarray:

    hist = calculate_hist_of_img(img_array, return_normalized=True)
    hist_ref[] = apply_hist_modification_transform(img_array, hist)

    if mode == "greedy":

    if mode == "non-greedy":

    if mode == "post-disturbance":



    return modified_img