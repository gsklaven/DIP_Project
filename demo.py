from PIL import Image
import numpy as np
from hist_utils import calculate_hist_of_img
from hist_utils import apply_hist_modification_transform
from equalization import greedy
from equalization import nongreedy


filename = "images/input_img.jpg"
img = Image.open(fp=filename)
bw_img = img.convert("L")
# img_array = np.array(bw_img).astype(float) / 255.0

img_array = np.array([
    [0.0, 0.0, 0.0, 0.3],
    [0.3, 0.3, 0.6, 0.6],
    [0.8, 0.8, 1.0, 1.0],
    [1.0, 1.0, 1.0, 1.0],
    [1.0, 1.0, 1.0, 1.0],
    [1.0, 1.0, 1.0, 1.0],
    [0.0, 0.0, 0.0, 0.3],
])

hist = calculate_hist_of_img(img_array, return_normalized=False)
print(hist)

test = apply_hist_modification_transform(img_array, hist)
print(test)

best = greedy(img_array, 5)
print(best)

cist = nongreedy(img_array, 5)
print(cist)
