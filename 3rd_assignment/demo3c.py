import matplotlib.pyplot as plt
import os
from scipy.io import loadmat
from image_to_graph import image_to_graph
from n_cuts import n_cuts_recursive


plt.ioff()
output_dir = "./output_plots/ncuts_recursive"
os.makedirs(output_dir, exist_ok=True)

# Ορισμός παραμέτρων για τον αλγόριθμο normalized cuts
t1 = 5      # Μέγιστος αριθμός clusters
t2 = 0.20   # Κατώφλι για το n-cut value

# Φόρτωση δεδομένων εικόνας d2a από το αρχείο .mat
data2a = loadmat("dip_hw_3.mat")
d2a = data2a["d2a"]

m2, n2, c2 = d2a.shape
img_graph = image_to_graph(d2a)

# Εφαρμογή normalized cuts με αναδρομή
labels = n_cuts_recursive(img_graph, t1, t2)

# Αναδιάταξη labels σε μορφή εικόνας
label_img = labels.reshape(m2, n2)

plt.figure(figsize=(10, 4))

# Εμφάνιση αρχικής εικόνας
plt.subplot(1, 2, 1)
plt.title("Original Image")
plt.imshow(d2a)
plt.axis('off')

# Εμφάνιση αποτελέσματος normalized recursive cuts
plt.subplot(1, 2, 2)
plt.title(f"Normalized recursive cuts (k={2}, T1={t1}, T2={t2})")
plt.imshow(label_img, cmap='viridis')
plt.axis('off')

# Αποθήκευση plot σε αρχείο
plt.savefig(os.path.join(output_dir, f"d2a_ncuts_recursive_{2}.png"))
plt.tight_layout()

# Επαναλαμβάνεται η ίδια διαδικασία για τη δεύτερη εικόνα d2b
data2b = loadmat("dip_hw_3.mat")
d2b = data2b["d2b"]

m3, n3, c3 = d2b.shape
img_graph = image_to_graph(d2b)

labels = n_cuts_recursive(img_graph, t1, t2)
label_img = labels.reshape(m3, n3)

plt.figure(figsize=(10, 4))

plt.subplot(1, 2, 1)
plt.title("Original Image")
plt.imshow(d2b)
plt.axis('off')

plt.subplot(1, 2, 2)
plt.title(f"Normalized recursive cuts (k={2}, T1={t1}, T2={t2})")
plt.imshow(label_img, cmap='viridis')
plt.axis('off')

plt.tight_layout()
plt.savefig(os.path.join(output_dir, f"d2b_ncuts_recursive_{2}.png"))
