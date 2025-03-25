import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten, Input, Conv3D, LSTM, BatchNormalization
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import pickle

dossier_npy = "data/coordonnees"
longueur_sequence = 20

x = []
y = []

for fichier in os.listdir(dossier_npy):
    if fichier.endswith(".npy"):
        donnees = np.load(os.path.join(dossier_npy, fichier))

        print(f"{fichier} : {donnees.shape}")

        if donnees.shape[0] >= longueur_sequence:
            for i in range(len(donnees) - longueur_sequence):
                x.append(donnees[i:i + longueur_sequence])
                y.append(fichier.split(".")[0])
        else:
            print(f"⚠️ {fichier} ignoré : seulement {donnees.shape[0]} frames (minimum = {longueur_sequence})")


x = np.array([seq.reshape((longueur_sequence, 21, 3, 1)) for seq in x])
y = LabelEncoder().fit_transform(y)

x_entrainement, x_val, y_entrainement, y_val = train_test_split(x, y, test_size=0.2, random_state=42)

modele = Sequential([
    Conv3D(64, (2, 2, 2), padding='same', activation='relu', input_shape=(20, 21, 3, 1)),
    BatchNormalization(),
    Dropout(0.3),

    Conv3D(64, kernel_size=(2, 2, 2), padding='same', activation='relu'),
    BatchNormalization(),
    Dropout(0.3),

    Flatten(),
    Dense(64, activation='relu'),
    Dropout(0.3),

    Dense(len(np.unique(y)), activation='softmax'),
])

modele.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

modele.fit(x_entrainement, y_entrainement, validation_data=(x_val, y_val), epochs=10, batch_size=32)

modele.save("models/sign_recognition.h5")

print(f"Modèle entraîné et sauvegardé avec {len(np.unique(y))}classes")
