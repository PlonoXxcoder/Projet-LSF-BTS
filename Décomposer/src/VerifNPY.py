import numpy as np
import os
data_dir = "data"
for filename in os.listdir(data_dir):
    if filename.endswith(".npy"):
        filepath = os.path.join(data_dir, filename)
        try:
            data = np.load(filepath)
            print(f"Forme de {filename}: {data.shape}")
        except Exception as e:
            print(f"Erreur lors du chargement de {filename}: {e}")