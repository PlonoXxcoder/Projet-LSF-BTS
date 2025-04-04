# -*- coding: utf-8 -*-
import cv2
import mediapipe as mp
import numpy as np
import tensorflow as tf
# Importer explicitement les couches/régularisateurs/etc. nécessaires
from tensorflow.keras.models import Sequential # type: ignore
from tensorflow.keras.layers import Input, LSTM, Dense, Dropout, BatchNormalization # type: ignore
from tensorflow.keras.optimizers import Adam # type: ignore
from tensorflow.keras.losses import CategoricalCrossentropy # type: ignore
from tensorflow.keras.regularizers import l2 # type: ignore
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint # type: ignore
from tensorflow.keras.utils import to_categorical # type: ignore

import os
import shutil # Pour supprimer le dossier tuner
import sys # Pour exit()
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from collections import Counter
import logging
import keras_tuner as kt # Assurez-vous que keras-tuner est installé

# Import pour SMOTE (équilibrage)
try:
    from imblearn.over_sampling import SMOTE # type: ignore
    IMBLEARN_AVAILABLE = True
except ImportError:
    logging.warning("Bibliothèque 'imbalanced-learn' non trouvée. L'équilibrage SMOTE ne sera pas disponible.")
    logging.warning("Pour l'installer : pip install imbalanced-learn")
    IMBLEARN_AVAILABLE = False
    SMOTE = None # Définir à None si non disponible

# --- Configuration du logging ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- ANSI escape codes for colors ---
class Colors:
    RESET = '\x1b[0m'
    BRIGHT_YELLOW = '\x1b[93m'
    BRIGHT_GREEN = '\x1b[92m'
    RED = '\x1b[31m'
    GREEN = '\x1b[32m'
    CYAN = '\x1b[36m'
    BOLD = '\x1b[1m'

# --- Constants ---
VOCABULARY_FILE = "vocabulaire.txt"
DATA_DIR = "data" # Dossier où traitementVideo.py a sauvegardé les .npy
MODEL_DIR = "models"
TUNER_DIR = 'keras_tuner_dir_lstm' # Dossier pour les résultats du tuner

# Noms de fichiers modèle
MODEL_TUNED_FILENAME = "model_tuned.h5" # Modèle final après tuning (Modes 1 & 3)
MODEL_BASIC_FILENAME = "model_basic.h5" # Modèle pour l'entraînement simple (Mode 2)
BEST_MODEL_TUNED_CHECKPOINT = "best_model_tuned_checkpoint.h5" # Checkpoint pendant entraînement final (Modes 1 & 3)
BEST_MODEL_BASIC_CHECKPOINT = "best_model_basic_checkpoint.h5" # Checkpoint pendant entraînement final (Mode 2)

# Noms de fichiers données accumulées (Sauvegardés dans le répertoire courant)
OLD_DATA_KEYPOINTS_FILE = "old_keypoints.npy"
OLD_DATA_LABELS_FILE = "old_labels.npy"

FIXED_LENGTH = 46 # Longueur de séquence pour le modèle LSTM

# --- Paramètres d'Extraction (DOIVENT CORRESPONDRE aux autres scripts) ---
NUM_POSE_KEYPOINTS = 4
NUM_HAND_KEYPOINTS = 21
NUM_COORDS = 3 # x, y, z

# Indices des points des lèvres (DOIVENT ÊTRE IDENTIQUES)
MOUTH_LANDMARK_INDICES = sorted(list(set([
    61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291,
    78, 95, 88, 178, 87, 14, 317, 402, 318, 324, 308,
    185, 40, 39, 37, 0, 267, 269, 270, 409,
    191, 80, 81, 82, 13, 312, 311, 310, 415
])))
NUM_MOUTH_KEYPOINTS = len(MOUTH_LANDMARK_INDICES)

# ---> MISE À JOUR FEATURES_PER_FRAME (DOIT CORRESPONDRE) <---
FEATURES_PER_FRAME = (NUM_POSE_KEYPOINTS * NUM_COORDS) + \
                     (NUM_HAND_KEYPOINTS * NUM_COORDS) * 2 + \
                     (NUM_MOUTH_KEYPOINTS * NUM_COORDS) # <-- AJOUT

# --- Paramètres par défaut pour l'entraînement ---
# Utilisés si non spécifiés ou pour Mode 2/3
DEFAULT_EPOCHS_TUNING = 60
DEFAULT_EPOCHS_FINAL = 150
DEFAULT_BATCH_SIZE = 32
DEFAULT_LEARNING_RATE = 1e-4 # Utilisé pour Mode 2

logging.info("--- Configuration Entrainement (avec Bouche) ---")
logging.info(f"Nombre points Pose: {NUM_POSE_KEYPOINTS}, Main: {NUM_HAND_KEYPOINTS}, Bouche: {NUM_MOUTH_KEYPOINTS}")
logging.info(f"FEATURES_PER_FRAME attendu : {FEATURES_PER_FRAME}")
logging.info(f"FIXED_LENGTH (séquence): {FIXED_LENGTH}")
logging.info(f"VOCABULARY_FILE: {VOCABULARY_FILE}")
logging.info(f"DATA_DIR: {DATA_DIR}")
logging.info(f"MODEL_DIR: {MODEL_DIR}")
logging.info(f"Accumulation NPY Files: {OLD_DATA_KEYPOINTS_FILE}, {OLD_DATA_LABELS_FILE} (in current dir)")
logging.info(f"SMOTE disponible: {IMBLEARN_AVAILABLE}")
logging.info("------------------------------------------------")

# --- Fonctions Utilitaires (load_vocabulary, augment_data_noise, balance_training_data, load_data, combine_data, split_data) ---
def load_vocabulary(filepath):
    """Charge le vocabulaire depuis un fichier 'mot:index'."""
    vocabulaire = {}
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            for line_num, line in enumerate(f, 1):
                parts = line.strip().split(":")
                if len(parts) == 2 and parts[0] and parts[1].isdigit():
                    mot, index = parts
                    vocabulaire[mot.lower()] = int(index) # Stocker en minuscules
                elif line.strip():
                    logging.warning(f"Format ligne incorrect (ligne {line_num}) vocabulaire: '{line.strip()}'")
    except FileNotFoundError:
        logging.error(f"Erreur: Fichier vocabulaire '{filepath}' non trouvé.")
        return None
    if not vocabulaire:
        logging.warning(f"Avertissement: Vocabulaire '{filepath}' est vide.")
    logging.info(f"Vocabulaire chargé ({len(vocabulaire)} mots) depuis {filepath}")
    return vocabulaire

def augment_data_noise(keypoints_batch, noise_std_dev=0.005):
    """Applique un bruit gaussien aux keypoints."""
    augmented_batch = keypoints_batch.copy()
    if augmented_batch.ndim != 3 or augmented_batch.shape[2] != FEATURES_PER_FRAME:
         logging.warning(f"Augmentation Bruit: Shape inattendue {augmented_batch.shape}, attendu (N, {FIXED_LENGTH}, {FEATURES_PER_FRAME}). Augmentation ignorée.")
         return keypoints_batch # Retourne l'original si la shape est mauvaise
    noise = np.random.normal(0, noise_std_dev, size=augmented_batch.shape)
    augmented_batch += noise
    return augmented_batch

def balance_training_data(x_train, y_train, strategy='auto', random_state=42):
    """Équilibre les données d'entraînement en utilisant SMOTE."""
    if not IMBLEARN_AVAILABLE or SMOTE is None:
        logging.warning("SMOTE non disponible. Retour données entraînement originales.")
        return x_train, y_train

    if x_train.shape[0] == 0:
        logging.warning("x_train est vide, impossible d'appliquer SMOTE.")
        return x_train, y_train

    n_samples, seq_len, n_features = x_train.shape
    # Assurer que y_train est one-hot
    if y_train.ndim == 1:
         logging.error("y_train doit être en format one-hot pour balance_training_data.")
         # Tentative de conversion si possible (ou retourner erreur)
         if len(np.unique(y_train)) > 1:
              try:
                   num_classes_infer = int(np.max(y_train) + 1)
                   y_train = to_categorical(y_train, num_classes=num_classes_infer)
                   logging.warning("y_train converti en one-hot.")
              except:
                   logging.error("Conversion de y_train en one-hot échouée.")
                   return x_train, y_train # Retourner original
         else:
              logging.warning("y_train a une seule classe, SMOTE non applicable.")
              return x_train, y_train


    num_classes = y_train.shape[1]
    if num_classes <= 1:
        logging.warning("Moins de 2 classes détectées dans y_train. SMOTE non applicable.")
        return x_train, y_train

    # SMOTE attend données 2D (samples, features) et labels 1D (indices)
    x_train_reshaped = x_train.reshape(n_samples, seq_len * n_features)
    y_train_indices = np.argmax(y_train, axis=1)

    logging.info("Distribution classes AVANT SMOTE:")
    unique_before, counts_before = np.unique(y_train_indices, return_counts=True)
    min_samples = float('inf')
    valid_classes_count = 0
    for cls, count in zip(unique_before, counts_before):
        logging.info(f"  Classe {cls}: {count}")
        if count > 1: # SMOTE a besoin d'au moins 2 échantillons par classe pour k>=1
            min_samples = min(min_samples, count)
            valid_classes_count += 1

    # Vérifier si SMOTE est nécessaire ou possible
    if len(unique_before) <= 1:
        logging.warning("Une seule classe présente. SMOTE non applicable.")
        return x_train, y_train

    k_neighbors = min(5, min_samples - 1) if min_samples > 1 else 0 # Ajuster k_neighbors

    if k_neighbors < 1:
        logging.warning(f"Pas assez d'échantillons dans la classe minoritaire (min_samples={min_samples}) pour SMOTE (k={k_neighbors} < 1). Retour données originales.")
        return x_train, y_train

    logging.info(f"Application de SMOTE (stratégie '{strategy}', k_neighbors={k_neighbors})...")
    try:
        # Utiliser k_neighbors ajusté
        smote = SMOTE(sampling_strategy=strategy, random_state=random_state, k_neighbors=k_neighbors, n_jobs=-1)
        x_train_resampled_reshaped, y_train_resampled_indices = smote.fit_resample(x_train_reshaped, y_train_indices)
    except ValueError as ve:
         # Cas fréquent: "Expected n_neighbors <= n_samples, but n_samples = X, n_neighbors = Y"
         logging.error(f"Erreur Valeur pendant SMOTE (peut-être lié à k_neighbors ou n_samples per class): {ve}. Retour données originales.")
         return x_train, y_train
    except Exception as e:
        logging.exception(f"Erreur pendant SMOTE : {e}. Retour données originales.")
        return x_train, y_train

    n_samples_after = x_train_resampled_reshaped.shape[0]
    logging.info(f"Distribution classes APRÈS SMOTE (Total: {n_samples_after} échantillons):")
    unique_after, counts_after = np.unique(y_train_resampled_indices, return_counts=True)
    for cls, count in zip(unique_after, counts_after): logging.info(f"  Classe {cls}: {count}")

    # Remodeler X resampled en 3D
    x_train_balanced = x_train_resampled_reshaped.reshape(n_samples_after, seq_len, n_features)
    # Re-convertir y resampled en one-hot
    y_train_balanced_onehot = to_categorical(y_train_resampled_indices, num_classes=num_classes)

    return x_train_balanced, y_train_balanced_onehot

def load_data(data_dir, new_data_files, vocabulaire, video_label_mapping, fixed_length, num_classes):
    """Charge les fichiers .npy, vérifie la shape, pad/truncate, et crée les labels one-hot."""
    logging.info(f"Chargement et prétraitement des données depuis '{data_dir}'...")
    new_keypoints_list = []
    new_labels_list = []
    total_files = len(new_data_files)
    processed_count = 0
    skipped_count = 0

    for i, data_file in enumerate(new_data_files):
        data_path = os.path.join(data_dir, data_file)
        label_name = video_label_mapping.get(data_file)

        # Vérifier si le label existe et est dans le vocabulaire (insensible à la casse)
        if label_name is None or label_name.lower() not in vocabulaire:
            logging.warning(f"[{i+1}/{total_files}] Label '{label_name}' invalide ou absent du vocabulaire pour '{data_file}'. Ignoré.")
            skipped_count += 1
            continue

        try:
            keypoints = np.load(data_path)

            # --- Vérification Shape ---
            if keypoints.ndim != 2 or keypoints.shape[1] != FEATURES_PER_FRAME:
                logging.warning(
                    f"[{i+1}/{total_files}] Shape .npy inattendue {keypoints.shape} pour '{data_file}' "
                    f"(attendu N, {FEATURES_PER_FRAME}). Ignoré."
                )
                skipped_count += 1
                continue

            # --- Padding / Truncating ---
            current_len = keypoints.shape[0]
            if current_len == 0:
                 logging.warning(f"[{i+1}/{total_files}] Fichier '{data_file}' contient 0 frame. Ignoré.")
                 skipped_count += 1
                 continue

            processed_keypoints = None
            if current_len > fixed_length:
                processed_keypoints = keypoints[-fixed_length:, :]
            elif current_len < fixed_length:
                padding_len = fixed_length - current_len
                padding = np.zeros((padding_len, FEATURES_PER_FRAME), dtype=keypoints.dtype)
                processed_keypoints = np.concatenate([padding, keypoints], axis=0)
            else: # Exactement la bonne longueur
                processed_keypoints = keypoints

            # Double vérification de la shape finale
            if processed_keypoints is not None and processed_keypoints.shape == (fixed_length, FEATURES_PER_FRAME):
                new_keypoints_list.append(processed_keypoints)
                label_index = vocabulaire[label_name.lower()]
                label_one_hot = to_categorical(label_index, num_classes=num_classes)
                new_labels_list.append(label_one_hot)
                processed_count += 1
            else:
                logging.error(f"[{i+1}/{total_files}] Shape finale incorrecte {processed_keypoints.shape if processed_keypoints is not None else 'None'} "
                              f"après padding/trunc pour '{data_file}'. Ignoré.")
                skipped_count += 1

        except FileNotFoundError:
             logging.error(f"[{i+1}/{total_files}] Fichier .npy '{data_path}' non trouvé. Ignoré.")
             skipped_count += 1
        except Exception as e:
            logging.exception(f"[{i+1}/{total_files}] Erreur chargement/traitement de '{data_file}': {e}. Ignoré.")
            skipped_count += 1

    logging.info(f"Chargement terminé: {processed_count} séquences chargées, {skipped_count} ignorées.")

    if not new_keypoints_list:
        logging.warning("Aucune nouvelle donnée valide n'a pu être chargée.")
        return None, None

    new_keypoints = np.array(new_keypoints_list)
    new_labels = np.array(new_labels_list)
    logging.info(f"Nouvelles données formatées: Keypoints shape={new_keypoints.shape}, Labels shape={new_labels.shape}")
    return new_keypoints, new_labels

def combine_data(old_keypoints_path, old_labels_path, new_keypoints, new_labels, num_classes):
    """Combine les anciennes données (si existent et compatibles) avec les nouvelles."""
    if new_keypoints is None or new_labels is None:
        logging.warning("Nouvelles données sont None, tentative de chargement des anciennes seulement.")
        try:
             old_keypoints = np.load(old_keypoints_path)
             old_labels = np.load(old_labels_path)
             if old_keypoints.ndim == 3 and old_keypoints.shape[1:] == (FIXED_LENGTH, FEATURES_PER_FRAME) and \
                old_labels.ndim == 2 and old_labels.shape[1] == num_classes:
                 logging.info(f"Nouvelles données vides, utilisation des anciennes données seules: {old_keypoints.shape[0]} séquences.")
                 return old_keypoints, old_labels
             else:
                 logging.error("Nouvelles données vides ET anciennes données incompatibles ou corrompues.")
                 return None, None
        except FileNotFoundError:
             logging.error("Nouvelles données vides ET aucune ancienne donnée trouvée.")
             return None, None
        except Exception as e:
             logging.exception(f"Erreur chargement anciennes données (nouvelles données vides): {e}")
             return None, None

    expected_keypoint_shape = (FIXED_LENGTH, FEATURES_PER_FRAME)
    expected_label_shape_dim1 = num_classes

    try:
        old_keypoints = np.load(old_keypoints_path)
        old_labels = np.load(old_labels_path)
        logging.info(f"Anciennes données trouvées: {old_keypoints.shape[0]} séquences. Tentative de combinaison...")

        old_k_shape_ok = old_keypoints.ndim == 3 and old_keypoints.shape[1:] == expected_keypoint_shape
        old_l_shape_ok = old_labels.ndim == 2 and old_labels.shape[1] == expected_label_shape_dim1

        if old_k_shape_ok and old_l_shape_ok:
            all_keypoints = np.concatenate([old_keypoints, new_keypoints], axis=0)
            all_labels = np.concatenate([old_labels, new_labels], axis=0)
            logging.info(f"Données combinées avec succès: {all_keypoints.shape[0]} séquences totales.")
        else:
            logging.error(f"Incompatibilité détectée entre anciennes et nouvelles données!")
            logging.error(f"  Anciennes K: {old_keypoints.shape}, Attendu K: (N, {FIXED_LENGTH}, {FEATURES_PER_FRAME})")
            logging.error(f"  Anciennes L: {old_labels.shape}, Attendu L: (N, {num_classes})")
            logging.warning("Utilisation des NOUVELLES données seulement.")
            all_keypoints = new_keypoints
            all_labels = new_labels

    except FileNotFoundError:
        logging.info("Pas d'anciennes données trouvées. Utilisation des nouvelles données seulement.")
        all_keypoints = new_keypoints
        all_labels = new_labels
    except Exception as e:
        logging.exception(f"Erreur chargement/concaténation anciennes données: {e}. Utilisation des nouvelles données seulement.")
        all_keypoints = new_keypoints
        all_labels = new_labels

    return all_keypoints, all_labels

def split_data(all_keypoints, all_labels):
    """Divise les données en ensembles Train, Validation, Test (stratifié si possible)."""
    if all_keypoints is None or all_labels is None or all_keypoints.shape[0] < 5:
        logging.error("Pas assez de données valides pour diviser (minimum 5 requis).")
        return None, None, None, None, None, None

    n_samples = all_keypoints.shape[0]
    # Ajuster les tailles pour être plus robuste avec peu de données
    # Viser ~20% pour validation, ~10% pour test, minimum 1 sample si possible
    if n_samples <= 5:
         val_size = 1
         test_size = 1
         train_size = n_samples - val_size - test_size
         if train_size < 1: # Cas extrême (ex: n=2)
             val_size = 1
             test_size = 0
             train_size = 1
    else:
         # Viser ~30% pour val+test
         test_val_size = min(0.3, max(2/n_samples if n_samples > 0 else 0.3, 0.1))
         # Viser ~1/3 de ça pour le test set, donc ~10% du total
         test_size_ratio = min(0.34, max(1/int(n_samples * test_val_size) if int(n_samples * test_val_size) > 0 else 0.34, 0.1))
         if int(n_samples * test_val_size * test_size_ratio) < 1: # Assurer au moins 1 pour test
             test_size_ratio = 1/int(n_samples * test_val_size) if int(n_samples * test_val_size) > 0 else 0.5


    logging.info(f"Division des données ({n_samples} échantillons): Ratio Test+Val=~{test_val_size:.2f}, Ratio Test dans Temp=~{test_size_ratio:.2f}")

    x_train, x_val, x_test, y_train, y_val, y_test = None, None, None, None, None, None

    try:
        # Essayer la division stratifiée
        y_indices = np.argmax(all_labels, axis=1)
        min_class_count = np.min(np.bincount(y_indices)) if len(y_indices) > 0 else 0
        required_for_stratify = 2

        if min_class_count < required_for_stratify:
             raise ValueError(f"Classe minoritaire a seulement {min_class_count} échantillons, insuffisant pour double split stratifié.")


        x_train, x_temp, y_train, y_temp = train_test_split(
            all_keypoints, all_labels,
            test_size=test_val_size,
            random_state=42,
            stratify=y_indices
        )

        if x_temp.shape[0] < 2:
            logging.warning("Ensemble temporaire (Val+Test) trop petit (<2), tout est assigné à la validation (stratifié).")
            x_val, x_test = x_temp, np.array([]).reshape(0, *x_temp.shape[1:])
            y_val, y_test = y_temp, np.array([]).reshape(0, *y_temp.shape[1:])
        else:
            # Diviser l'ensemble temporaire en validation et test (stratifié aussi)
            y_temp_indices = np.argmax(y_temp, axis=1)
            min_temp_class_count = np.min(np.bincount(y_temp_indices)) if len(y_temp_indices) > 0 else 0
            if min_temp_class_count < required_for_stratify:
                 raise ValueError(f"Classe minoritaire dans Temp a seulement {min_temp_class_count} échantillons, insuffisant pour split Val/Test stratifié.")

            x_val, x_test, y_val, y_test = train_test_split(
                x_temp, y_temp,
                test_size=test_size_ratio,
                random_state=42,
                stratify=y_temp_indices
            )

        logging.info(f"Split stratifié réussi: Train={x_train.shape[0]}, Val={x_val.shape[0]}, Test={x_test.shape[0] if x_test is not None and x_test.size > 0 else '0'}")

    except ValueError as e:
        # Si la stratification échoue (ex: classe avec 1 seul échantillon)
        logging.warning(f"Échec division stratifiée ({e}). Tentative sans stratification...")
        try:
            x_train, x_temp, y_train, y_temp = train_test_split(
                all_keypoints, all_labels,
                test_size=test_val_size,
                random_state=42
            )
            if x_temp.shape[0] < 2:
                 logging.warning("Ensemble temporaire (Val+Test) trop petit (<2), tout est assigné à la validation (non stratifié).")
                 x_val, x_test = x_temp, np.array([]).reshape(0, *x_temp.shape[1:])
                 y_val, y_test = y_temp, np.array([]).reshape(0, *y_temp.shape[1:])
            else:
                 x_val, x_test, y_val, y_test = train_test_split(
                    x_temp, y_temp,
                    test_size=test_size_ratio,
                    random_state=42
                 )
            logging.info(f"Split non stratifié réussi: Train={x_train.shape[0]}, Val={x_val.shape[0]}, Test={x_test.shape[0] if x_test is not None and x_test.size > 0 else '0'}")
        except Exception as e_split:
            logging.exception(f"Erreur finale lors de la division des données: {e_split}")
            return None, None, None, None, None, None

    return x_train, x_val, x_test, y_train, y_val, y_test

# --- Fonction pour Keras Tuner (build_model) ---
def build_model(hp, input_shape, num_classes):
    """Fonction de construction de modèle pour Keras Tuner."""
    model = Sequential(name="SignLanguageLSTM_Tuned_withMouth")
    model.add(Input(shape=input_shape, name='input_layer'))

    hp_units_1 = hp.Int('units_1', min_value=32, max_value=128, step=32)
    hp_units_2 = hp.Int('units_2', min_value=64, max_value=256, step=64)
    hp_units_3 = hp.Int('units_3', min_value=32, max_value=128, step=32)

    hp_dropout_1 = hp.Float('dropout_1', min_value=0.3, max_value=0.7, step=0.1)
    hp_dropout_2 = hp.Float('dropout_2', min_value=0.3, max_value=0.7, step=0.1)
    hp_dropout_3 = hp.Float('dropout_3', min_value=0.3, max_value=0.7, step=0.1)
    hp_dropout_4 = hp.Float('dropout_4', min_value=0.4, max_value=0.8, step=0.1)

    hp_dense_units = hp.Int('dense_units', min_value=32, max_value=128, step=32)

    hp_learning_rate = hp.Choice('learning_rate', values=[5e-4, 1e-4, 5e-5, 1e-5])

    hp_lstm_l2 = hp.Float('lstm_l2', min_value=1e-5, max_value=1e-2, sampling='log', default=1e-4)
    hp_dense_l2 = hp.Float('dense_l2', min_value=1e-4, max_value=1e-2, sampling='log', default=1e-3)

    hp_use_bn = hp.Boolean("use_bn", default=True)

    model.add(LSTM(hp_units_1, return_sequences=True, name='lstm_1',
                   kernel_regularizer=l2(hp_lstm_l2), recurrent_regularizer=l2(hp_lstm_l2)))
    if hp_use_bn: model.add(BatchNormalization(name='bn_1'))
    model.add(Dropout(hp_dropout_1, name='dropout_1'))

    model.add(LSTM(hp_units_2, return_sequences=True, name='lstm_2',
                   kernel_regularizer=l2(hp_lstm_l2), recurrent_regularizer=l2(hp_lstm_l2)))
    if hp_use_bn: model.add(BatchNormalization(name='bn_2'))
    model.add(Dropout(hp_dropout_2, name='dropout_2'))

    model.add(LSTM(hp_units_3, return_sequences=False, name='lstm_3',
                   kernel_regularizer=l2(hp_lstm_l2), recurrent_regularizer=l2(hp_lstm_l2)))
    if hp_use_bn: model.add(BatchNormalization(name='bn_3'))
    model.add(Dropout(hp_dropout_3, name='dropout_3'))

    model.add(Dense(hp_dense_units, activation='relu', name='dense_1',
                  kernel_regularizer=l2(hp_dense_l2)))
    if hp_use_bn: model.add(BatchNormalization(name='bn_4'))
    model.add(Dropout(hp_dropout_4, name='dropout_4'))

    model.add(Dense(num_classes, activation='softmax', name='output_layer'))

    model.compile(
        optimizer=Adam(learning_rate=hp_learning_rate),
        loss=CategoricalCrossentropy(),
        metrics=['accuracy']
    )
    return model

# --- NOUVELLE Fonction pour Modèle Simple (Mode 2) ---
def build_basic_model(input_shape, num_classes):
    """Construit un modèle LSTM simple et fixe, sans hyperparamètres."""
    logging.info("Construction du modèle de base (fixe)...")
    model = Sequential(name="SignLanguageLSTM_Basic_withMouth")
    model.add(Input(shape=input_shape, name='input_layer'))

    # Architecture simple et fixe
    model.add(LSTM(64, return_sequences=True, name='lstm_1')) # 1ère couche LSTM
    model.add(Dropout(0.5, name='dropout_1'))                 # Dropout

    model.add(LSTM(128, return_sequences=False, name='lstm_2'))# 2ème couche LSTM (pas return_sequences)
    model.add(Dropout(0.5, name='dropout_2'))                 # Dropout

    model.add(Dense(64, activation='relu', name='dense_1'))    # Couche Dense
    model.add(Dropout(0.5, name='dropout_3'))                 # Dropout

    model.add(Dense(num_classes, activation='softmax', name='output_layer')) # Couche de sortie

    # Pas de Batch Normalization ou L2 pour ce modèle simple
    # La compilation se fera dans la boucle principale avec un LR fixe

    logging.info("Modèle de base construit.")
    return model

# --- Fonctions d'Entraînement Final et Évaluation (compile_and_train_model, evaluate_model) ---
def compile_and_train_model(model, x_train, y_train, x_val, y_val, epochs, batch_size, best_model_save_path, learning_rate=None, model_name="modèle"):
    """Compile (si nécessaire avec LR optionnel) et entraîne le modèle."""
    if not model.optimizer: # Vérifier si un optimiseur existe déjà
         lr = learning_rate if learning_rate is not None else DEFAULT_LEARNING_RATE # Utilise LR fourni ou défaut
         logging.warning(f"{model_name} non compilé, compilation avec lr={lr:.0e}.")
         model.compile(optimizer=Adam(learning_rate=lr), loss=CategoricalCrossentropy(), metrics=['accuracy'])
    elif learning_rate is not None and model.optimizer.learning_rate.numpy() != learning_rate: # Comparer les valeurs
         logging.warning(f"Re-compilation de {model_name} avec un nouveau learning_rate: {learning_rate:.0e}")
         # Conserver l'état de l'optimiseur si possible? Plus simple de recompiler
         model.compile(optimizer=Adam(learning_rate=learning_rate), loss=CategoricalCrossentropy(), metrics=['accuracy'])
    elif not model.optimizer: # Si l'optimiseur est None pour une raison quelconque
        lr = learning_rate if learning_rate is not None else DEFAULT_LEARNING_RATE
        logging.warning(f"Optimiseur manquant pour {model_name}, compilation avec lr={lr:.0e}.")
        model.compile(optimizer=Adam(learning_rate=lr), loss=CategoricalCrossentropy(), metrics=['accuracy'])


    model.summary(line_length=120) # Afficher le résumé du modèle

    # Callbacks
    early_stopping = EarlyStopping(monitor='val_loss', patience=20, # Garder patience assez longue
                                   restore_best_weights=True, verbose=1)
    # Assurer que le dossier pour le checkpoint existe (pourrait être dans MODEL_DIR)
    checkpoint_dir = os.path.dirname(best_model_save_path)
    if checkpoint_dir and not os.path.exists(checkpoint_dir):
        logging.info(f"Création du dossier pour le checkpoint : {checkpoint_dir}")
        os.makedirs(checkpoint_dir, exist_ok=True)

    checkpoint = ModelCheckpoint(best_model_save_path, save_best_only=True,
                                 monitor='val_loss', mode='min', verbose=1)

    logging.info(f"Début entraînement '{model_name}' pour {epochs} epochs (Batch Size: {batch_size})...")
    history = model.fit(
        x_train, y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_data=(x_val, y_val),
        callbacks=[early_stopping, checkpoint],
        shuffle=True,
        verbose=1
        )
    logging.info(f"Entraînement '{model_name}' terminé.")
    return history

def evaluate_model(model, x_test, y_test, best_model_save_path, history=None):
    """Évalue le modèle sur l'ensemble de test après avoir chargé les meilleurs poids."""
    logging.info(f"Chargement meilleurs poids depuis '{best_model_save_path}' pour évaluation finale...")
    test_accuracy = 0.0
    model_to_evaluate = None

    if os.path.exists(best_model_save_path):
        try:
            # Utiliser load_model est plus sûr pour recharger l'état complet
            model_to_evaluate = tf.keras.models.load_model(best_model_save_path)
            logging.info("Meilleur modèle (checkpoint) chargé avec succès pour évaluation.")
        except Exception as e:
            logging.exception(f"Erreur chargement modèle depuis checkpoint '{best_model_save_path}': {e}. Fallback sur modèle en mémoire.")
            # Attention: model peut ne pas avoir les meilleurs poids si restore_best_weights=False ou a échoué
            model_to_evaluate = model
    else:
         logging.warning(f"Fichier checkpoint '{best_model_save_path}' non trouvé. Utilisation du modèle actuel en mémoire (peut ne pas être le meilleur).")
         model_to_evaluate = model

    if model_to_evaluate is None:
         logging.error("Aucun modèle disponible pour l'évaluation.")
         return 0.0 # Retourner 0 si aucun modèle

    # Évaluer sur l'ensemble de test s'il existe
    if x_test is not None and y_test is not None and x_test.shape[0] > 0:
        logging.info("Évaluation sur l'ensemble de test...")
        try:
            loss, accuracy = model_to_evaluate.evaluate(x_test, y_test, verbose=0)
            logging.info(f"{Colors.BRIGHT_GREEN}Performance Test: Loss={loss:.4f}, Accuracy={accuracy:.4f}{Colors.RESET}")
            test_accuracy = accuracy
        except Exception as e_eval:
             logging.exception(f"Erreur pendant model.evaluate sur le test set: {e_eval}")
    else:
        logging.info("Pas d'ensemble de test valide fourni pour l'évaluation.")

    # --- Visualisation ---
    if history and hasattr(history, 'history') and 'loss' in history.history and 'val_loss' in history.history:
        plt.figure(figsize=(10, 4))
        plt.subplot(1, 2, 1)
        plt.plot(history.history['loss'], label='Train Loss')
        plt.plot(history.history['val_loss'], label='Val Loss')
        plt.title('Loss')
        plt.xlabel('Epoch'); plt.ylabel('Loss'); plt.legend(); plt.grid(True)
        plt.subplot(1, 2, 2)
        if 'accuracy' in history.history and 'val_accuracy' in history.history:
             plt.plot(history.history['accuracy'], label='Train Accuracy')
             plt.plot(history.history['val_accuracy'], label='Val Accuracy')
             plt.title('Accuracy')
             plt.xlabel('Epoch'); plt.ylabel('Accuracy'); plt.legend(); plt.grid(True)
        else: plt.title('Accuracy (Data Missing)')
        plt.tight_layout()
        # Note: Le plot sera affiché à la toute fin du script dans main()

    return test_accuracy

# --- Fonction de Sauvegarde (save_model_and_data) ---
# ---> MODIFICATION: Supprime les makedirs pour les fichiers .npy <---
def save_model_and_data(model, keypoints_to_save, labels_to_save, keypoints_save_path, labels_save_path, model_save_path):
    """Sauvegarde les données combinées (pour la prochaine exécution) et le modèle final."""
    # S'assurer que le dossier POUR LE MODELE existe
    model_dir = os.path.dirname(model_save_path)
    if model_dir and not os.path.exists(model_dir): # Vérifie si model_dir n'est pas vide
         logging.info(f"Création du dossier pour le modèle : {model_dir}")
         os.makedirs(model_dir, exist_ok=True)
    # Note: On ne crée PAS de dossier pour keypoints_save_path et labels_save_path
    # car ils sont définis comme des fichiers dans le répertoire courant.

    # Sauvegarder les données (celles utilisées AVANT augmentation/SMOTE pour l'accumulation)
    if keypoints_to_save is not None and labels_to_save is not None:
        logging.info(f"Sauvegarde des données accumulées ({keypoints_to_save.shape[0]} séquences) dans:")
        logging.info(f"  Keypoints: {keypoints_save_path}")
        logging.info(f"  Labels:    {labels_save_path}")
        try:
            np.save(keypoints_save_path, keypoints_to_save)
            np.save(labels_save_path, labels_to_save)
        except Exception as e:
            logging.exception(f"Erreur sauvegarde fichiers .npy des données accumulées: {e}")
    else:
        logging.warning("Aucune donnée à sauvegarder pour accumulation (keypoints ou labels sont None).")


    # Sauvegarder le modèle final Keras
    if model is not None:
        logging.info(f"Sauvegarde du modèle final entraîné dans {model_save_path}")
        try:
            model.save(model_save_path) # Utilise le format Keras natif (dossier ou .h5)
            logging.info(f"Modèle final sauvegardé avec succès.")
        except Exception as e:
            logging.exception(f"Erreur lors de la sauvegarde du modèle final Keras: {e}")
    else:
        logging.error("Le modèle final est None, impossible de le sauvegarder.")

# --- Fonction Principale (main) ---
def main():
    print(f"{Colors.BOLD}{Colors.CYAN}--- Options d'Entraînement ---{Colors.RESET}")
    print("1. Mode 'Seuil et Tentatives': Recherche hyperparamètres, puis entraîne en visant un seuil d'accuracy minimum avec N tentatives.")
    print("2. Mode 'Basique': Entraîne un modèle simple et fixe, sans tuning ni seuil.")
    print("3. Mode 'Tuning Unique': Recherche hyperparamètres, entraîne le meilleur modèle trouvé une seule fois.")
    print("-" * 30)

    mode = None
    while mode not in ['1', '2', '3']:
        mode = input("Choisissez le mode (1, 2, ou 3) : ").strip()
        if mode not in ['1', '2', '3']:
            print(f"{Colors.RED}Entrée invalide. Veuillez choisir 1, 2, ou 3.{Colors.RESET}")
    mode = int(mode)

    # Paramètres spécifiques au mode
    accuracy_threshold = 0.0
    max_retrain_attempts = 1

    if mode == 1:
        print(f"\n{Colors.BOLD}{Colors.CYAN}--- Mode 1 : Seuil et Tentatives ---{Colors.RESET}")
        while True:
            try:
                threshold_input = input(f"Entrez le seuil d'accuracy TEST désiré (ex: 0.85 pour 85%, laissez vide pour 0.0): ")
                accuracy_threshold = float(threshold_input) if threshold_input else 0.0
                if 0.0 <= accuracy_threshold <= 1.0:
                    break
                else:
                    print(f"{Colors.RED}Le seuil doit être entre 0.0 et 1.0.{Colors.RESET}")
            except ValueError:
                print(f"{Colors.RED}Entrée invalide. Veuillez entrer un nombre.{Colors.RESET}")

        while True:
            try:
                attempts_input = input(f"Entrez le nombre maximum de tentatives d'entraînement (ex: 3, minimum 1): ")
                max_retrain_attempts = int(attempts_input) if attempts_input else 1
                if max_retrain_attempts >= 1:
                    break
                else:
                    print(f"{Colors.RED}Le nombre de tentatives doit être au moins 1.{Colors.RESET}")
            except ValueError:
                print(f"{Colors.RED}Entrée invalide. Veuillez entrer un entier.{Colors.RESET}")
        logging.info(f"Mode 1 sélectionné: Seuil={accuracy_threshold:.2f}, Tentatives max={max_retrain_attempts}")

    elif mode == 2:
        print(f"\n{Colors.BOLD}{Colors.CYAN}--- Mode 2 : Basique ---{Colors.RESET}")
        logging.info("Mode 2 sélectionné: Entraînement simple sans tuning ni seuil.")

    elif mode == 3:
        print(f"\n{Colors.BOLD}{Colors.CYAN}--- Mode 3 : Tuning Unique ---{Colors.RESET}")
        logging.info("Mode 3 sélectionné: Recherche HP et entraînement unique du meilleur modèle.")
        max_retrain_attempts = 1 # Assure une seule tentative d'entraînement final
        accuracy_threshold = 0.0 # Pas de seuil requis

    # --- Début du processus commun ---

    # Supprimer anciens fichiers de données accumulées (si demandé)
    # Ces fichiers sont maintenant attendus dans le répertoire courant
    old_keypoints_path = OLD_DATA_KEYPOINTS_FILE
    old_labels_path = OLD_DATA_LABELS_FILE
    if os.path.exists(old_keypoints_path) or os.path.exists(old_labels_path):
        logging.warning("-" * 60)
        logging.warning(f"Fichiers d'accumulation '{old_keypoints_path}' / '{old_labels_path}' existent dans le répertoire courant.")
        user_input = input("Voulez-vous supprimer ces fichiers et continuer ? (o/N) : ").strip().lower()
        if user_input == 'o':
            logging.info("Suppression des anciens fichiers .npy d'accumulation...")
            try:
                if os.path.exists(old_keypoints_path): os.remove(old_keypoints_path)
                if os.path.exists(old_labels_path): os.remove(old_labels_path)
                logging.info("Anciens fichiers d'accumulation supprimés.")
            except Exception as e:
                logging.error(f"Erreur suppression: {e}. Veuillez supprimer manuellement."); sys.exit(1)
        else:
             logging.info("Anciens fichiers d'accumulation conservés.")

    # Charger Vocabulaire
    vocabulaire = load_vocabulary(VOCABULARY_FILE)
    if vocabulaire is None or not vocabulaire:
        logging.error("Erreur critique: Vocabulaire non chargé ou vide. Arrêt."); sys.exit(1)
    num_classes = len(vocabulaire)
    logging.info(f"Nombre de classes (mots) détectées: {num_classes}")
    index_to_word = {idx: word for word, idx in vocabulaire.items()}

    # Identifier les fichiers .npy et mapper aux labels
    video_label_mapping = {}
    new_data_files = []
    logging.info(f"Recherche fichiers '*_keypoints.npy' dans : '{DATA_DIR}'")
    if not os.path.isdir(DATA_DIR):
        logging.error(f"Erreur: Dossier de données '{DATA_DIR}' n'existe pas."); sys.exit(1)

    sorted_vocab = sorted(vocabulaire.keys(), key=len, reverse=True)
    found_files_count = 0
    for filename in sorted(os.listdir(DATA_DIR)):
        if filename.lower().endswith("_keypoints.npy"):
            full_path = os.path.join(DATA_DIR, filename)
            if os.path.isfile(full_path): # S'assurer que c'est un fichier
                found_files_count += 1
                base_name = filename[:-len("_keypoints.npy")]
                found_label = None
                # Essayer de faire correspondre le début du nom de fichier avec le vocabulaire
                for vocab_word in sorted_vocab:
                    if base_name.lower().startswith(vocab_word.lower()):
                         # Vérifier que ce qui suit est un séparateur, un chiffre ou rien
                         suffix_part = base_name[len(vocab_word):]
                         if not suffix_part or suffix_part.startswith("_") or suffix_part.isdigit():
                              found_label = vocab_word
                              break
                if found_label:
                    video_label_mapping[filename] = found_label
                    new_data_files.append(filename)
                    # logging.debug(f"  Mappé '{filename}' -> '{found_label}'")
                else:
                     logging.warning(f"  -> Avertissement: Aucun label du vocabulaire trouvé pour '{filename}'. Ignoré.")

    logging.info(f"{found_files_count} fichiers '*_keypoints.npy' trouvés au total.")
    if not new_data_files:
        logging.error(f"Erreur: Aucun fichier .npy mappable trouvé dans '{DATA_DIR}'. Vérifiez les noms de fichiers et le vocabulaire."); sys.exit(1)
    logging.info(f"{len(new_data_files)} fichiers .npy valides et mappés trouvés pour traitement.")

    # Chemins des fichiers
    model_filename = MODEL_TUNED_FILENAME if mode != 2 else MODEL_BASIC_FILENAME
    model_path = os.path.join(MODEL_DIR, model_filename)

    checkpoint_filename = BEST_MODEL_TUNED_CHECKPOINT if mode != 2 else BEST_MODEL_BASIC_CHECKPOINT
    best_model_checkpoint_path = os.path.join(MODEL_DIR, checkpoint_filename)
    # old_keypoints_path et old_labels_path sont déjà définis (fichiers locaux)

    # Charger et Préparer les Données (Commun à tous les modes)
    new_keypoints, new_labels = load_data(DATA_DIR, new_data_files, vocabulaire, video_label_mapping, FIXED_LENGTH, num_classes)
    all_keypoints_orig, all_labels_orig = combine_data(old_keypoints_path, old_labels_path, new_keypoints, new_labels, num_classes)

    if all_keypoints_orig is None or all_labels_orig is None or all_keypoints_orig.shape[0] == 0:
        logging.error("Erreur: Aucune donnée valide à traiter après combinaison/chargement. Arrêt.")
        sys.exit(1)

    logging.info(f"Total données combinées (avant augmentation/split): {all_keypoints_orig.shape[0]} séquences.")
    logging.info("Distribution classes AVANT Augmentation/Équilibrage:")
    label_indices_orig = np.argmax(all_labels_orig, axis=1)
    class_counts_orig = Counter(label_indices_orig)
    for class_index, count in sorted(class_counts_orig.items()):
        class_name = index_to_word.get(class_index, f"Classe_{class_index}")
        logging.info(f"  Classe '{class_name}' ({class_index}): {count} échantillons")

    logging.info("Application de l'augmentation par bruit gaussien...")
    augmented_keypoints = augment_data_noise(all_keypoints_orig)
    logging.info(f"Augmentation par bruit terminée. Shape: {augmented_keypoints.shape}")

    logging.info("Division des données augmentées en ensembles Train, Validation, Test...")
    x_train_aug, x_val, x_test, y_train, y_val, y_test = split_data(augmented_keypoints, all_labels_orig)

    if x_train_aug is None or y_train is None or x_val is None or y_val is None : # Test peut être vide
        logging.error("Échec de la division des données. Impossible de continuer.")
        save_model_and_data(None, all_keypoints_orig, all_labels_orig, old_keypoints_path, old_labels_path, model_path) # Sauve données même si split échoue
        sys.exit(1)
    logging.info(f"Shapes après split: Train={x_train_aug.shape}, Val={x_val.shape}, Test={x_test.shape if x_test is not None and x_test.size > 0 else 'Vide'}")

    logging.info("--- Équilibrage de l'ensemble d'entraînement (SMOTE) ---")
    x_train_balanced, y_train_balanced = balance_training_data(x_train_aug, y_train)
    logging.info(f"Shape Train Set après SMOTE: {x_train_balanced.shape if x_train_balanced is not None else 'N/A'}")


    # --- Logique spécifique au mode ---

    final_model = None
    final_history = None
    final_test_accuracy = 0.0
    input_shape = (FIXED_LENGTH, FEATURES_PER_FRAME)

    # --- MODE 2: Basique ---
    if mode == 2:
        logging.info(f"{Colors.CYAN}--- Exécution Mode 2 : Entraînement Basique ---{Colors.RESET}")
        basic_model = build_basic_model(input_shape, num_classes)

        # Entraîner le modèle de base
        final_history = compile_and_train_model(
            basic_model,
            x_train_balanced, y_train_balanced,
            x_val, y_val,
            epochs=DEFAULT_EPOCHS_FINAL, # Utiliser les epochs finales par défaut
            batch_size=DEFAULT_BATCH_SIZE,
            best_model_save_path=best_model_checkpoint_path,
            learning_rate=DEFAULT_LEARNING_RATE, # LR fixe
            model_name="Modèle Basique"
        )

        # Évaluer sur le test set
        final_test_accuracy = evaluate_model(
            basic_model, # Le modèle en mémoire *devrait* avoir les meilleurs poids grâce à restore_best_weights=True
            x_test, y_test,
            best_model_checkpoint_path, # Chemin pour log et fallback si restore a échoué
            final_history
        )

        # Charger le meilleur modèle depuis le checkpoint pour sauvegarde finale (plus sûr)
        if os.path.exists(best_model_checkpoint_path):
             try:
                  final_model = tf.keras.models.load_model(best_model_checkpoint_path)
                  logging.info(f"Meilleur modèle basique chargé depuis {best_model_checkpoint_path} pour sauvegarde finale.")
             except Exception as e:
                  logging.error(f"Erreur chargement checkpoint basique {best_model_checkpoint_path}: {e}. Sauvegarde du modèle en mémoire.")
                  final_model = basic_model # Fallback (peut ne pas être le meilleur)
        else:
             logging.warning(f"Checkpoint {best_model_checkpoint_path} non trouvé. Sauvegarde du modèle en mémoire (peut ne pas être le meilleur).")
             final_model = basic_model


    # --- MODES 1 et 3: Tuning + Entraînement ---
    elif mode == 1 or mode == 3:
        if mode == 1:
             logging.info(f"{Colors.CYAN}--- Exécution Mode 1 : Tuning avec Seuil ({accuracy_threshold:.2f}) et Tentatives ({max_retrain_attempts}) ---{Colors.RESET}")
        else: # Mode 3
             logging.info(f"{Colors.CYAN}--- Exécution Mode 3 : Tuning Unique ---{Colors.RESET}")

        # Supprimer l'ancien dossier de tuning s'il existe
        if os.path.exists(TUNER_DIR):
            logging.warning(f"Suppression de l'ancien dossier de tuning '{TUNER_DIR}'...")
            try:
                shutil.rmtree(TUNER_DIR)
            except Exception as e_rm:
                logging.error(f"Erreur suppression dossier tuner: {e_rm}. Risque d'utiliser anciens résultats.")

        # Configurer et lancer le Tuner
        tuner = kt.Hyperband(
            lambda hp: build_model(hp, input_shape=input_shape, num_classes=num_classes),
            objective='val_accuracy',
            max_epochs=DEFAULT_EPOCHS_TUNING,
            factor=3,
            hyperband_iterations=1,
            directory=TUNER_DIR,
            project_name='sign_language_lstm_with_tuning',
            overwrite=True # Assure qu'on écrase si rmtree a échoué
        )
        tuner.search_space_summary()
        logging.info(f"--- Début Recherche Hyperparamètres (sur données équilibrées, {DEFAULT_EPOCHS_TUNING} max epochs) ---")
        search_callbacks = [ EarlyStopping(monitor='val_loss', patience=10, verbose=1) ]

        # Vérifier si les données d'entraînement/validation existent avant search
        if x_train_balanced is None or y_train_balanced is None or x_val is None or y_val is None:
             logging.error("Données d'entraînement ou validation manquantes avant tuner.search. Arrêt.")
             save_model_and_data(None, all_keypoints_orig, all_labels_orig, old_keypoints_path, old_labels_path, model_path)
             sys.exit(1)

        tuner.search(
            x_train_balanced, y_train_balanced,
            epochs=DEFAULT_EPOCHS_TUNING,
            batch_size=DEFAULT_BATCH_SIZE,
            validation_data=(x_val, y_val),
            callbacks=search_callbacks,
            verbose=1
        )
        logging.info("--- Fin Recherche Hyperparamètres ---")

        # Récupérer les meilleurs hyperparamètres
        try:
            best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]
            logging.info(f"{Colors.BRIGHT_GREEN}Meilleurs hyperparamètres trouvés par Keras Tuner:{Colors.RESET}")
            for hp_name in best_hps.space:
                 logging.info(f"  - {hp_name.name}: {best_hps.get(hp_name.name)}")
        except IndexError:
             logging.error("Keras Tuner n'a retourné aucun résultat valide. Vérifiez les logs. Arrêt.")
             save_model_and_data(None, all_keypoints_orig, all_labels_orig, old_keypoints_path, old_labels_path, model_path)
             sys.exit(1)
        except Exception as e:
            logging.exception(f"Erreur récupération meilleurs HP: {e}. Arrêt.")
            save_model_and_data(None, all_keypoints_orig, all_labels_orig, old_keypoints_path, old_labels_path, model_path)
            sys.exit(1)

        # --- Entraînement Final (avec ou sans boucle de retry selon le mode) ---
        retrain_count = 0
        best_overall_accuracy = -1.0 # Initialiser à -1 pour s'assurer qu'on sauvegarde au moins un modèle
        best_model_overall = None
        history_for_best_model = None

        if mode == 1:
            logging.info(f"{Colors.BRIGHT_YELLOW}--- Début Entraînement Final Mode 1 (max {max_retrain_attempts} tentatives, objectif Test Acc > {accuracy_threshold:.2f}) ---{Colors.RESET}")
        else: # Mode 3
            logging.info(f"{Colors.BRIGHT_YELLOW}--- Début Entraînement Final Mode 3 (1 tentative) ---{Colors.RESET}")


        while retrain_count < max_retrain_attempts:
            retrain_count += 1
            logging.info(f"{Colors.BRIGHT_YELLOW}>>> Tentative d'entraînement #{retrain_count}/{max_retrain_attempts} <<<{Colors.RESET}")

            logging.info("Reconstruction du modèle avec les meilleurs hyperparamètres...")
            # Utiliser tuner.hypermodel.build(best_hps) pour obtenir une instance fraîche
            current_model = tuner.hypermodel.build(best_hps)
            # Le learning rate est déjà défini dans les best_hps et utilisé par build/compile

            current_history = compile_and_train_model(
                current_model,
                x_train_balanced, y_train_balanced,
                x_val, y_val,
                epochs=DEFAULT_EPOCHS_FINAL,
                batch_size=DEFAULT_BATCH_SIZE,
                best_model_save_path=best_model_checkpoint_path,
                # Pas besoin de passer learning_rate ici, il est dans best_hps
                model_name=f"Modèle Tuned (Tentative {retrain_count})"
            )

            logging.info(f"--- Évaluation Tentative #{retrain_count} sur Test Set ---")
            # current_model devrait avoir les meilleurs poids grâce à restore_best_weights
            current_test_accuracy = evaluate_model(
                current_model,
                x_test, y_test,
                best_model_checkpoint_path, # Chemin pour log et fallback
                current_history
            )

            if current_test_accuracy > best_overall_accuracy:
                logging.info(f"{Colors.BRIGHT_GREEN}Nouvelle meilleure accuracy ({current_test_accuracy:.4f}) trouvée lors de la tentative {retrain_count}. Mise à jour du meilleur modèle global.{Colors.RESET}")
                best_overall_accuracy = current_test_accuracy
                history_for_best_model = current_history
                # Recharger explicitement depuis le checkpoint est la méthode la plus sûre
                if os.path.exists(best_model_checkpoint_path):
                    try:
                        best_model_overall = tf.keras.models.load_model(best_model_checkpoint_path)
                        logging.info("Meilleur modèle global mis à jour depuis le checkpoint.")
                    except Exception as e_load:
                         logging.error(f"Impossible de recharger le meilleur modèle depuis {best_model_checkpoint_path}: {e_load}")
                         best_model_overall = current_model # Fallback (potentiellement pas le meilleur)
                else:
                     logging.warning(f"Checkpoint {best_model_checkpoint_path} non trouvé APRÈS entraînement. Modèle en mémoire sera utilisé pour sauvegarde.")
                     best_model_overall = current_model # Fallback (potentiellement pas le meilleur)
            # elif best_model_overall is None: # Si c'est la première tentative et qu'elle a échoué à s'améliorer, sauvegarder qd même
            #      logging.warning("Première tentative terminée, sauvegarde du modèle même si accuracy faible.")
            #      best_overall_accuracy = current_test_accuracy
            #      history_for_best_model = current_history
            #      if os.path.exists(best_model_checkpoint_path):
            #          # ... (même logique de chargement que ci-dessus) ...
            #      else: best_model_overall = current_model


            # Conditions d'arrêt (Mode 1 seulement)
            if mode == 1:
                if best_overall_accuracy >= accuracy_threshold and best_overall_accuracy > 0: # Vérifier > 0 pour éviter seuil 0.0
                    logging.info(f"{Colors.BRIGHT_GREEN}Objectif d'accuracy ({accuracy_threshold:.2f}) atteint ou dépassé ({best_overall_accuracy:.4f})! Arrêt.{Colors.RESET}")
                    break
                elif retrain_count >= max_retrain_attempts:
                     logging.warning(f"{Colors.RED}Max tentatives ({max_retrain_attempts}) atteint sans atteindre {accuracy_threshold:.2f}. Meilleure Acc: {best_overall_accuracy:.4f}{Colors.RESET}")
                # else: continue loop (si pas atteint et pas max tentatives)

        # Assigner le meilleur modèle trouvé à final_model pour sauvegarde
        # S'assurer que même si la boucle ne s'exécute pas ou échoue, on a une valeur
        if best_model_overall is None and 'current_model' in locals():
             logging.warning("Aucun 'meilleur modèle global' n'a été défini (potentiellement 0 tentative réussie). Utilisation du dernier modèle entraîné.")
             final_model = current_model
             final_history = current_history
             final_test_accuracy = current_test_accuracy if 'current_test_accuracy' in locals() else 0.0
        elif best_model_overall is None:
             logging.error("Aucun modèle n'a pu être entraîné ou sélectionné comme 'meilleur'. Impossible de sauvegarder.")
             final_model = None # Assurer que c'est None
             final_history = None
             final_test_accuracy = 0.0
        else:
             final_model = best_model_overall
             final_history = history_for_best_model
             final_test_accuracy = best_overall_accuracy


    # --- Sauvegarde Finale (Commun, utilise final_model et model_path définis plus tôt) ---
    logging.info("--- Sauvegarde Finale Données et Meilleur Modèle Obtenu ---")
    save_model_and_data(
        final_model,              # Le meilleur modèle trouvé (basic ou tuned)
        all_keypoints_orig,       # Données originales K (AVANT augmentation/SMOTE)
        all_labels_orig,          # Données originales L (AVANT augmentation/SMOTE)
        old_keypoints_path,       # Fichier .npy K accumulés (local)
        old_labels_path,          # Fichier .npy L accumulés (local)
        model_path                # Chemin final du modèle (.h5 ou dossier) dans MODEL_DIR
    )

    # Afficher le graphique d'entraînement final
    if final_history:
        logging.info("Affichage du graphique d'entraînement du modèle final...")
        plt.figure(figsize=(12, 5))
        mode_str = f"Mode {mode}" + (f" (Test Acc: {final_test_accuracy:.3f})" if final_test_accuracy > 0 else "")
        plt.suptitle(f'Modèle Final ({mode_str}) - Historique Entraînement', fontsize=14)
        plt.subplot(1, 2, 1)
        plt.plot(final_history.history['loss'], label='Train Loss')
        plt.plot(final_history.history['val_loss'], label='Val Loss')
        plt.xlabel('Epoch'); plt.ylabel('Loss'); plt.legend(); plt.grid(True)
        plt.title('Loss vs. Epoch')
        plt.subplot(1, 2, 2)
        if 'accuracy' in final_history.history and 'val_accuracy' in final_history.history:
             plt.plot(final_history.history['accuracy'], label='Train Accuracy')
             plt.plot(final_history.history['val_accuracy'], label='Val Accuracy')
             plt.xlabel('Epoch'); plt.ylabel('Accuracy'); plt.legend(); plt.grid(True)
             plt.title('Accuracy vs. Epoch')
        else: plt.title('Accuracy data non disponible')
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        logging.info("Fermez la fenêtre du graphique pour terminer le script.")
        plt.show() # Bloquant
    else:
        logging.warning("Aucun historique d'entraînement trouvé pour le modèle final.")

    logging.info(f"{Colors.BRIGHT_GREEN}--- Fin du script d'entraînement (Mode {mode}) ---{Colors.RESET}")

if __name__ == "__main__":
    # Configuration TF pour la croissance mémoire GPU
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            logging.info(f"GPU(s) détecté(s) et configuré(s) pour Memory Growth: {gpus}")
        except RuntimeError as e:
            logging.error(f"Erreur configuration Memory Growth GPU: {e}")
    else:
        logging.warning("Aucun GPU détecté par TensorFlow. Utilisation CPU.")

    # Point d'entrée principal
    main()

    # Fermer toutes les figures matplotlib au cas où show() ne bloquerait pas
    plt.close('all')
    print(f"{Colors.GREEN}Script terminé.{Colors.RESET}")