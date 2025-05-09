from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import os
from sobel_edge import sobel_edge
from log_edge import log_edge

plt.ioff()
output_dir = "./output_plots"
os.makedirs(output_dir, exist_ok=True)

# Load images
filename = "images/basketball_large.png"
img = Image.open(fp=filename)
bw_img = img.convert("L")
img_array = np.array(bw_img).astype(float) / 255.0
plt.imshow(img_array, cmap='gray')
plt.title("Original Image")
plt.axis('off')
plt.savefig(os.path.join(output_dir, "1_Original Image.png"))

# Sobel edge detection
thres = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
detected_points = []

plt.figure(figsize=(12, 12))
for i, t in enumerate(thres):
    out_img_array = sobel_edge(img_array, t)
    num_edges = np.sum(out_img_array)  # Count the number of edge pixels
    detected_points.append(num_edges)
    plt.subplot(3, 3, i + 1)
    plt.imshow(out_img_array, cmap='gray')
    plt.title(f"Threshold: {t}")
    plt.axis('off')
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "2_Sobel_Edge_Detection_Grid_Compact.png"))

plt.figure()
plt.plot(thres, detected_points, marker='o')
plt.title("Number of Detected Edge Pixels vs. Threshold")
plt.xlabel("Threshold Value")
plt.ylabel("Number of Edge Pixels")
plt.grid(True)
plt.savefig(os.path.join(output_dir, "Sobel_Edge_Pixel_Count_vs_Threshold.png"))
plt.show()

# for i in thres:
#     out_img_array = sobel_edge(img_array, i)
# 
#     plt.imshow(out_img_array, cmap='gray')
#     plt.title(f"Sobel Edge Detection (Threshold: {i})")
#     plt.axis('off')
#     plt.savefig(os.path.join(output_dir, f"2_Sobel_Edge_Detection_Threshold_{i}.png"))
#     plt.close()

# Log edge detection
out_img_array = log_edge(img_array)
plt.imshow(out_img_array, cmap='gray')
plt.title("Log Edge Detection")
plt.axis('off')
plt.savefig(os.path.join(output_dir, "3_Log_Edge_Detection.png"))
plt.show()
