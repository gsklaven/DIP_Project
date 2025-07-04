from scipy.io import loadmat
from spectral_clustering import spectral_clustering


# Φόρτωση δεδομένων από το αρχείο .mat
data1 = loadmat("dip_hw_3.mat")
d1a = data1["d1a"]  # Επιλογή του πίνακα d1a από τα δεδομένα

k_list = [2, 3, 4]  # Λίστα με διαφορετικές τιμές για τον αριθμό των clusters
for idx, k in enumerate(k_list):
    spectral_clustering(d1a, k)  # Εκτέλεση spectral clustering για κάθε k
    print(f"Spectral clustering with k={k} completed.")  # Εμφάνιση μηνύματος ολοκλήρωσης
    print(spectral_clustering(d1a, k))  # Εμφάνιση των αποτελεσμάτων του clustering
