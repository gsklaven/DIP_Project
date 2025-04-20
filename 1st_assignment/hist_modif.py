from typing import Dict
import numpy as np
from hist_utils import calculate_hist_of_img, apply_hist_modification_transform
from equalization import greedy, nongreedy, post_disturbance


def perform_hist_modification(img_array: np.ndarray, hist_ref: Dict, mode: str) -> np.ndarray:
    """
    Υπολογίζει το histogram modification για μια εικόνα με βάση την επιθυμητή κατανομή.

    Args:
        img_array (np.ndarray): Δισδιάστατος πίνακας που αναπαριστά την εικόνα εισόδου [0, 1].
        hist_ref (Dict): Λεξικό με τα επίπεδα εξόδου ως κλειδιά και τις επιθυμητές συχνότητες ως τιμές.
        mode (str): Επιλογή αλγορίθμου ("greedy", "non-greedy", "post-disturbance").

    Returns:
        np.ndarray: Τροποποιημένη εικόνα με κατανομή ιστογράμματος κοντά στο hist_ref.
    """
    hist_input = calculate_hist_of_img(img_array, return_normalized=True)

    modified_img = apply_hist_modification_transform(img_array, hist_ref)

    # Επιλογή αλγορίθμου βάσει mode
    if mode == "greedy":
        modified_img = greedy(modified_img, len(hist_ref))
    elif mode == "non-greedy":
        modified_img = nongreedy(modified_img, len(hist_ref))
    elif mode == "post-disturbance":
        modified_img = post_disturbance(modified_img, len(hist_ref))
    else:
        raise ValueError("Μη έγκυρη τιμή για το mode.")

    return modified_img
