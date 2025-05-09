from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import os
from fir_conv import fir_conv

plt.ioff()

output_dir = "./output_plots"
os.makedirs(output_dir, exist_ok=True)

# Load images
filename = "./basketball_large.png"
img = Image.open(fp=filename)
bw_img = img.convert("L")
img_array = np.array(bw_img).astype(float) / 255.0

h = np.array([[1, 1],
              [1, 1]])
in_img_array = np.array([[1, 2, 3],
                         [4, 5, 6],
                         [7, 8, 9]])


result, origin = fir_conv(in_img_array, h, None, None)
print("Input Matrix:")
print(in_img_array)
print("Output Matrix:")
print(result)
print("Output Origin:", origin)
