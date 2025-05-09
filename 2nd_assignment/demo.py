from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import os
from fir_conv import fir_conv
from sobel_edge import sobel_edge
from log_edge import log_edge

plt.ioff()

output_dir = "./output_plots2"
os.makedirs(output_dir, exist_ok=True)

# Load images
filename = "images/europe_map.gif"
img = Image.open(fp=filename)
bw_img = img.convert("L")
img_array = np.array(bw_img).astype(float) / 255.0
plt.imshow(img_array, cmap='gray')
plt.title("Original Image")
plt.axis('off')
plt.savefig(os.path.join(output_dir, "1_Original Image.png"))
plt.show()

# h = np.array([[1, 1],
#               [1, 1]])
# in_img_array = np.array([[1, 2, 3],
#                          [4, 5, 6],
#                          [7, 8, 9]])
#
#
# result, origin = fir_conv(in_img_array, h, None, None)
# print("Input Matrix:")
# print(in_img_array)
# print("Output Matrix:")
# print(result)
# print("Output Origin:", origin)

# Sobel edge detection
thres = 0.5
out_img_array = sobel_edge(img_array, thres)
plt.imshow(out_img_array, cmap='gray')
plt.title("Sobel Edge Detection")
plt.axis('off')
plt.savefig(os.path.join(output_dir, "2_Sobel_Edge_Detection.png"))
plt.show()

# Log edge detection
out_img_array = log_edge(img_array)
plt.imshow(out_img_array, cmap='gray')
plt.title("Log Edge Detection")
plt.axis('off')
plt.savefig(os.path.join(output_dir, "3_Log_Edge_Detection.png"))
plt.show()
