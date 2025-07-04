import matplotlib.pyplot as plt
import os
from scipy.io import loadmat
from image_to_graph import image_to_graph
from n_cuts import n_cuts, calculate_n_cut_value


plt.ioff()
output_dir = "./output_plots/ncuts_recursive_step1"
os.makedirs(output_dir, exist_ok=True)

# Λίστα με τιμές k (αριθμός clusters)
k_list = [2]

# Φόρτωση δεδομένων εικόνας d2a από το αρχείο .mat
data2a = loadmat("dip_hw_3.mat")
d2a = data2a["d2a"]

m2, n2, c2 = d2a.shape
img_graph = image_to_graph(d2a)

# Εφαρμογή normalized cuts για κάθε τιμή του k
for idx, k in enumerate(k_list):
    labels = n_cuts(img_graph, k)  # Υπολογισμός labels
    n_cut_value = calculate_n_cut_value(img_graph, labels)  # Υπολογισμός τιμής n-cut

    label_img = labels.reshape(m2, n2)  # Αναδιάταξη labels σε μορφή εικόνας

    plt.figure(figsize=(10, 4))

    # Εμφάνιση αρχικής εικόνας
    plt.subplot(1, 2, 1)
    plt.title("Original Image")
    plt.imshow(d2a)
    plt.axis('off')

    # Εμφάνιση αποτελέσματος normalized cuts με τιμή n-cut
    plt.subplot(1, 2, 2)
    plt.title(f"Normalized Cuts (k=2, n_cut_value={n_cut_value:.4f})")
    plt.imshow(label_img, cmap='viridis')
    plt.axis('off')

    # Αποθήκευση plot σε αρχείο
    plt.savefig(os.path.join(output_dir, f"d2a_ncuts_recursive_step1_{k}.png"))
    plt.tight_layout()

# Επαναλαμβάνεται η ίδια διαδικασία για τη δεύτερη εικόνα d2b
data2b = loadmat("dip_hw_3.mat")
d2b = data2b["d2b"]

m3, n3, c3 = d2b.shape
img_graph = image_to_graph(d2b)

for idx, k in enumerate(k_list):
    labels = n_cuts(img_graph, k)
    n_cut_value = calculate_n_cut_value(img_graph, labels)

    label_img = labels.reshape(m3, n3)

    plt.figure(figsize=(10, 4))

    plt.subplot(1, 2, 1)
    plt.title("Original Image")
    plt.imshow(d2b)
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.title(f"Normalized Cuts (k=2, n_cut_value={n_cut_value:.4f})")
    plt.imshow(label_img, cmap='viridis')
    plt.axis('off')

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"d2b_ncuts_recursive_step1_{k}.png"))
