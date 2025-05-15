from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import os
from sobel_edge import sobel_edge
from log_edge import log_edge
from circ_hough import circ_hough


plt.ioff()
output_dir = "./output_plots"
os.makedirs(output_dir, exist_ok=True)

# Load images
filename = "images/basketball_large.png"
img = Image.open(fp=filename)
bw_img = img.convert("L")
img_array = np.array(bw_img).astype(float) / 255.0

# Display original image
plt.imshow(img_array, cmap='gray')
plt.title("Original Image")
plt.axis('off')
plt.savefig(os.path.join(output_dir, "1_Original_Image.png"))

# Sobel edge detection
thres = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
detected_points = []

plt.figure(figsize=(12, 12))
for i, t in enumerate(thres):
    sobel_out_img_array = sobel_edge(img_array, t)
    num_edges = np.sum(sobel_out_img_array)
    detected_points.append(num_edges)
    plt.subplot(3, 3, i + 1)
    plt.imshow(sobel_out_img_array, cmap='gray')
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

# Log edge detection
log_out_img_array = log_edge(img_array)
plt.imshow(log_out_img_array, cmap='gray')
plt.title("Log Edge Detection")
plt.axis('off')
plt.savefig(os.path.join(output_dir, "3_Log_Edge_Detection.png"))
plt.show()

# Circle Hough Transform initialization
R_max = 500
dim = np.array([400, 400, 200])
V_min = [1500, 1750, 1850, 1950, 2000, 2250, 2500, 2750]

for i in V_min:
    # Circle Hough Transform with Sobel edge detection
    edge_array = sobel_edge(img_array, thres=0.5)
    centers_sobel, radii_sobel = circ_hough(edge_array, R_max, dim, i)

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.imshow(img_array, cmap='gray')
    for (cx, cy), r in zip(centers_sobel, radii_sobel):
        circle = plt.Circle((cy, cx), r, fill=False, color='blue', linewidth=2)
        ax.add_patch(circle)
        ax.plot(cy, cx, marker='x', color='green', markersize=8)
    plt.title(f"Ανίχνευση κύκλων Hough (Sobel) - V_min={i}")
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"Circle_Hough_Sobel_Vmin_{i}.png"))
    plt.close()

    # Circle Hough Transform with Log edge detection
    centers_log, radii_log = circ_hough(log_out_img_array, R_max, dim, i)

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.imshow(img_array, cmap='gray')
    for (cx, cy), r in zip(centers_log, centers_log):
        circle = plt.Circle((cy, cx), r, fill=False, color='blue', linewidth=2)
        ax.add_patch(circle)
        ax.plot(cy, cx, marker='x', color='green', markersize=8)
    plt.title(f"Ανίχνευση κύκλων Hough (LoG) - V_min={i}")
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"Circle_Hough_Log_Vmin_{i}.png"))
    plt.close()
