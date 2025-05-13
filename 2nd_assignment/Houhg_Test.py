import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from sobel_edge import sobel_edge
from circ_hough_new import circ_hough

# Φόρτωση εικόνας
filename = "images/basketball_large.png"
img = Image.open(fp=filename)
# width, height = img.size
# img_size = (width/2, height/2)
# img = img.resize((int(img_size[0]), int(img_size[1])))
bw_img = img.convert("L")
img_array = np.array(bw_img).astype(float) / 255.0


edge_array = sobel_edge(img_array, thres=0.5)

R_max = 800
dim = np.array([200, 200, 30])
V_min = 600

# Ανίχνευση κύκλων με circ_hough
centers, radii = circ_hough(edge_array, R_max, dim, V_min)

# Σχεδίαση αποτελέσματος
fig, ax = plt.subplots(figsize=(6, 6))
ax.imshow(img_array, cmap='gray')  # Υπόβαθρο η grayscale εικόνα

for (cx, cy), r in zip(centers, radii):
    circle = plt.Circle((cy, cx), r, fill=False, color='blue', linewidth=2)
    ax.add_patch(circle)
    ax.plot(cy, cx, marker='x', color='green', markersize=8, label="Κέντρο")

plt.title("Ανίχνευση κύκλων (custom Hough)")
plt.axis("off")
plt.tight_layout()
plt.show()
