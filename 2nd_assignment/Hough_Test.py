import numpy as np
import matplotlib.pyplot as plt
import os
from PIL import Image
from sobel_edge import sobel_edge
from circ_hough import circ_hough

plt.ioff()
output_dir = "output_plots/Plots for different dim"
os.makedirs(output_dir, exist_ok=True)

filename = "images/basketball_large.png"
img = Image.open(fp=filename)
width, height = img.size
bw_img = img.convert("L")
img_array = np.array(bw_img).astype(float) / 255.0

edge_array = sobel_edge(img_array, thres=0.5)

R_max = 500
dim = np.array([200, 200, 100])
V_min = 1750

fig, ax = plt.subplots(figsize=(6, 6))
ax.imshow(img, cmap='gray')
centers, radii = circ_hough(edge_array, R_max, dim, V_min)
for (cx, cy), r in zip(centers, radii):
    circle = plt.Circle((cy, cx), r, fill=False, color='blue', linewidth=2)
    ax.add_patch(circle)
    ax.plot(cy, cx, marker='x', color='green', markersize=8, label="Κέντρο")
plt.title(f"Ανίχνευση κύκλων Hough")
plt.axis("off")
plt.tight_layout()
plt.show()