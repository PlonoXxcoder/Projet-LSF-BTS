# -*- coding: utf-8 -*-
import cv2
import mediapipe as mp
import numpy as np
import tensorflow as tf
# <<< MODIFICATION: Importer explicitement les couches/régularisateurs nécessaires >>>
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import CategoricalCrossentropy
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.layers import Input, LSTM, Dense, Dropout, BatchNormalization
from tensorflow.keras.regularizers import l2
# <<< FIN MODIFICATION >>>
import os
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from collections import Counter
import logging
import keras_tuner as kt

# --- Configuration du logging ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- ANSI escape codes for colors ---
class Colors:
    RESET = '\x1b[0m'
    BRIGHT_YELLOW = '\x1b[93m'
    BRIGHT_GREEN = '\x1b[92m'
    RED = '\x1b[31m'
    GREEN = '\x1b[32m'
    CV_GREEN = (0, 255, 0)
    CV_YELLOW = (0, 255, 255)
    CV_RED = (0, 0, 255)
    CV_WHITE = (255, 255, 255)

# --- Constants ---
VOCABULARY_FILE = "vocabulaire.txt"
DATA_DIR = "data"
MODEL_DIR = "models"
MODEL_FILENAME = "model.h5" # Modèle final après tuning
BEST_MODEL_FILENAME = "best_model.h5" # Meilleur checkpoint PENDANT l'entraînement final
OLD_DATA_KEYPOINTS_FILE = "old_keypoints.npy"
OLD_DATA_LABELS_FILE = "old_labels.npy"
FIXED_LENGTH = 46
NUM_POSE_KEYPOINTS = 4
NUM_HAND_KEYPOINTS = 21
NUM_COORDS = 3
FEATURES_PER_FRAME = (NUM_POSE_KEYPOINTS * NUM_COORDS) + \
                     (NUM_HAND_KEYPOINTS * NUM_COORDS) * 2
logging.info(f"Nombre de features par frame attendu : {FEATURES_PER_FRAME}")

# --- Fonctions Utilitaires (inchangées) ---
def load_vocabulary(filepath):
    vocabulaire = {}
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            for line_num, line in enumerate(f, 1):
                parts = line.strip().split(":")
                if len(parts) == 2 and parts[0] and parts[1].isdigit():
                    mot, index = parts
                    vocabulaire[mot.lower()] = int(index)
                elif line.strip():
                    logging.warning(f"Avertissement (ligne {line_num}): Ligne ignorée dans '{filepath}' (format incorrect): '{line.strip()}'")
    except FileNotFoundError:
        logging.error(f"Erreur: Fichier vocabulaire '{filepath}' non trouvé.")
        return None
    if not vocabulaire:
        logging.warning(f"Avertissement: Vocabulaire chargé depuis '{filepath}' est vide.")
    logging.info(f"Vocabulaire chargé depuis '{filepath}' avec {len(vocabulaire)} mots.")
    return vocabulaire

def augment_data(keypoints_batch):
    augmented_batch = keypoints_batch.copy()
    num_samples, seq_len, num_features = augmented_batch.shape
    if num_features != FEATURES_PER_FRAME:
        logging.warning(f"Augmentation: Attend {FEATURES_PER_FRAME} features, reçu {num_features}. Augmentation ignorée.")
        return augmented_batch
    noise_std_dev = 0.005 # Conserver un bruit faible
    noise = np.random.normal(0, noise_std_dev, size=augmented_batch.shape)
    augmented_batch += noise
    return augmented_batch

def load_data(data_dir, new_data_files, vocabulaire, video_label_mapping, fixed_length, num_classes):
    logging.info("Chargement des nouvelles données...")
    new_keypoints_list = []
    new_labels_list = []
    for data_file in new_data_files:
        data_path = os.path.join(data_dir, data_file)
        label_name = video_label_mapping.get(data_file)
        if label_name is None or label_name.lower() not in vocabulaire:
            logging.warning(f"Label '{label_name}' invalide ou non trouvé dans vocabulaire pour {data_file}. Ignoré.")
            continue
        try:
            keypoints = np.load(data_path)
            if keypoints.ndim == 3 and keypoints.shape[1] == (NUM_POSE_KEYPOINTS + 2 * NUM_HAND_KEYPOINTS) and keypoints.shape[2] == NUM_COORDS:
                 logging.warning(f"Conversion ancienne shape {keypoints.shape} pour {data_file}")
                 keypoints = keypoints.reshape(keypoints.shape[0], -1)
            if keypoints.ndim != 2 or keypoints.shape[1] != FEATURES_PER_FRAME:
                logging.warning(f"Shape .npy inattendue {keypoints.shape} pour {data_file} (attendu N,{FEATURES_PER_FRAME}). Ignoré.")
                continue
            current_len = keypoints.shape[0]
            if current_len > fixed_length:
                keypoints = keypoints[:fixed_length, :]
            elif current_len < fixed_length:
                padding = np.zeros((fixed_length - current_len, keypoints.shape[1]), dtype=keypoints.dtype)
                keypoints = np.concatenate([padding, keypoints], axis=0) # Padding au début
            if keypoints.shape == (fixed_length, FEATURES_PER_FRAME):
                new_keypoints_list.append(keypoints)
                label_index = vocabulaire[label_name.lower()]
                label_one_hot = tf.keras.utils.to_categorical(label_index, num_classes=num_classes)
                new_labels_list.append(label_one_hot)
            else:
                logging.warning(f"Shape incorrecte {keypoints.shape} après padding/trunc pour {data_file}. Ignoré.")
        except Exception as e:
            logging.exception(f"Erreur chargement/traitement {data_file}: {e}. Ignoré.")
    if not new_keypoints_list:
        logging.warning("Aucune nouvelle donnée valide.")
        return None, None
    new_keypoints = np.array(new_keypoints_list); new_labels = np.array(new_labels_list)
    logging.info(f"Nouvelles données chargées: {new_keypoints.shape[0]} séquences.")
    return new_keypoints, new_labels

def combine_data(old_keypoints_path, old_labels_path, new_keypoints, new_labels, num_classes):
    try:
        old_keypoints = np.load(old_keypoints_path); old_labels = np.load(old_labels_path)
        logging.info(f"Anciennes données chargées: {old_keypoints.shape[0]}. Shape K: {old_keypoints.shape}, L: {old_labels.shape}")
        old_k_shape_ok = len(old_keypoints.shape) > 1 and old_keypoints.shape[1:] == new_keypoints.shape[1:]
        old_l_shape_ok = len(old_labels.shape) > 1 and old_labels.shape[1] == new_labels.shape[1]
        num_classes_compatible = old_labels.shape[1] == num_classes if len(old_labels.shape) > 1 else False
        if old_k_shape_ok and old_l_shape_ok and num_classes_compatible:
            all_keypoints = np.concatenate([old_keypoints, new_keypoints], axis=0)
            all_labels = np.concatenate([old_labels, new_labels], axis=0)
            logging.info(f"Données combinées: {all_keypoints.shape[0]} séquences.")
        else:
            logging.error("Incompatibilité détectée entre anciennes et nouvelles données."); # ... (logs détaillés d'erreur) ...
            logging.warning("Utilisation des nouvelles données seulement.")
            all_keypoints = new_keypoints; all_labels = new_labels
    except FileNotFoundError:
        logging.info("Pas d'anciennes données trouvées. Utilisation des nouvelles données seulement.")
        all_keypoints = new_keypoints; all_labels = new_labels
    except Exception as e:
        logging.exception(f"Erreur chargement/concat anciennes données: {e}. Utilisation nouvelles données.")
        all_keypoints = new_keypoints; all_labels = new_labels
    return all_keypoints, all_labels

def split_data(all_keypoints, all_labels):
    if all_keypoints.shape[0] < 5:
        logging.error("Pas assez de données pour diviser."); return None, None, None, None, None, None
    try:
        x_train, x_temp, y_train, y_temp = train_test_split(all_keypoints, all_labels, test_size=0.3, random_state=42, stratify=np.argmax(all_labels, axis=1))
        logging.info(f"Split initial: Train {x_train.shape[0]}, Temp (Val+Test) {x_temp.shape[0]}")
        if x_temp.shape[0] < 2:
             logging.warning("Pas assez d'échantillons temporaires pour créer ensemble test. Tout en validation.")
             x_val, x_test, y_val, y_test = x_temp, np.array([]).reshape(0, *x_temp.shape[1:]), y_temp, np.array([]).reshape(0, *y_temp.shape[1:])
        else:
             x_val, x_test, y_val, y_test = train_test_split(x_temp, y_temp, test_size=0.5, random_state=42, stratify=np.argmax(y_temp, axis=1))
        logging.info(f"Split final: Train {x_train.shape[0]}, Val {x_val.shape[0]}, Test {x_test.shape[0]}")
        return x_train, x_val, x_test, y_train, y_val, y_test
    except ValueError as e:
        logging.warning(f"Erreur division stratifiée ({e}). Tentative sans stratification...")
        try:
            x_train, x_temp, y_train, y_temp = train_test_split(all_keypoints, all_labels, test_size=0.3, random_state=42)
            if x_temp.shape[0] < 2:
                 x_val, x_test, y_val, y_test = x_temp, np.array([]).reshape(0, *x_temp.shape[1:]), y_temp, np.array([]).reshape(0, *y_temp.shape[1:])
            else:
                 x_val, x_test, y_val, y_test = train_test_split(x_temp, y_temp, test_size=0.5, random_state=42)
            logging.info(f"Split final (non stratifié): Train {x_train.shape[0]}, Val {x_val.shape[0]}, Test {x_test.shape[0]}")
            return x_train, x_val, x_test, y_train, y_val, y_test
        except Exception as e_split:
            logging.exception(f"Erreur finale division données: {e_split}")
            return None, None, None, None, None, None

# --- Fonction pour Keras Tuner (MODIFIÉE) ---
def build_model(hp, input_shape, num_classes):
    """Fonction de construction de modèle pour Keras Tuner (avec plus de régularisation)."""
    model = tf.keras.Sequential()
    model.add(Input(shape=input_shape, name='lstm_input'))

    # --- Hyperparamètres à tester (plages modifiées/ajoutées) ---
    hp_units_1 = hp.Int('units_1', min_value=32, max_value=96, step=32) # Plage un peu réduite
    hp_units_2 = hp.Int('units_2', min_value=64, max_value=192, step=64) # Plage un peu réduite
    hp_units_3 = hp.Int('units_3', min_value=32, max_value=96, step=32) # Plage un peu réduite

    # <<< MODIFICATION: Plages de Dropout augmentées >>>
    hp_dropout_1 = hp.Float('dropout_1', min_value=0.3, max_value=0.6, step=0.1)
    hp_dropout_2 = hp.Float('dropout_2', min_value=0.3, max_value=0.6, step=0.1)
    hp_dropout_3 = hp.Float('dropout_3', min_value=0.3, max_value=0.6, step=0.1)
    hp_dropout_4 = hp.Float('dropout_4', min_value=0.4, max_value=0.7, step=0.1) # Dropout final plus fort

    hp_dense_units = hp.Int('dense_units', min_value=32, max_value=96, step=32) # Plage un peu réduite

    hp_learning_rate = hp.Choice('learning_rate', values=[5e-4, 1e-4, 5e-5]) # Taux plus faibles

    # <<< AJOUT: Régularisation L2 pour les couches LSTM >>>
    # Utiliser 'sampling="log"' est bien pour les régularisations/learning rates
    hp_lstm_l2 = hp.Float('lstm_l2', min_value=1e-4, max_value=1e-2, sampling='log', default=1e-4)
    hp_dense_l2 = hp.Float('dense_l2', min_value=1e-4, max_value=1e-2, sampling='log', default=1e-3)
    # <<< FIN AJOUT >>>

    # --- Construction du modèle avec les hyperparamètres ---
    model.add(LSTM(hp_units_1, return_sequences=True, name='lstm_1',
                   kernel_regularizer=l2(hp_lstm_l2), recurrent_regularizer=l2(hp_lstm_l2))) # Ajout L2
    model.add(Dropout(hp_dropout_1, name='dropout_1'))

    model.add(LSTM(hp_units_2, return_sequences=True, name='lstm_2',
                   kernel_regularizer=l2(hp_lstm_l2), recurrent_regularizer=l2(hp_lstm_l2))) # Ajout L2
    model.add(Dropout(hp_dropout_2, name='dropout_2'))

    model.add(LSTM(hp_units_3, return_sequences=False, name='lstm_3',
                   kernel_regularizer=l2(hp_lstm_l2), recurrent_regularizer=l2(hp_lstm_l2))) # Ajout L2
    model.add(Dropout(hp_dropout_3, name='dropout_3'))

    model.add(Dense(hp_dense_units, activation='relu', name='dense_1',
                  kernel_regularizer=l2(hp_dense_l2))) # Ajout L2
    model.add(Dropout(hp_dropout_4, name='dropout_4'))

    model.add(Dense(num_classes, activation='softmax', name='output_layer'))

    # Compilation du modèle
    model.compile(
        optimizer=Adam(learning_rate=hp_learning_rate),
        loss=CategoricalCrossentropy(),
        metrics=['accuracy']
    )
    return model

# --- Fonctions d'Entraînement Final et Évaluation (MODIFIÉES) ---
def compile_and_train_model(model, x_train, y_train, x_val, y_val, epochs, batch_size, best_model_save_path):
    """Compile et entraîne le modèle (pour l'entraînement FINAL après tuning)."""
    if not model._is_compiled:
         # Tenter de récupérer le LR des HP si possible, sinon utiliser défaut
         try:
              lr = model.optimizer.learning_rate.numpy() # Si déjà partiellement entraîné par tuner
         except AttributeError:
              lr = 5e-4 # Valeur par défaut raisonnable si non compilé
              logging.warning(f"Modèle non compilé. Compilation avec lr={lr} (défaut)...")
              optimizer = Adam(learning_rate=lr)
              loss_fn = CategoricalCrossentropy()
              model.compile(optimizer=optimizer, loss=loss_fn, metrics=['accuracy'])

    model.summary()
    # <<< MODIFICATION: Patience réduite pour EarlyStopping >>>
    early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True, verbose=1)
    os.makedirs(os.path.dirname(best_model_save_path), exist_ok=True)
    # <<< MODIFICATION: Assurer que ModelCheckpoint surveille val_loss et mode min >>>
    checkpoint = ModelCheckpoint(best_model_save_path, save_best_only=True, monitor='val_loss', mode='min', verbose=1)
    logging.info("Début entraînement FINAL du meilleur modèle...")
    history = model.fit(
        x_train, y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_data=(x_val, y_val),
        callbacks=[early_stopping, checkpoint], # Passer les callbacks
        shuffle=True
        )
    return history

def evaluate_model(model, x_test, y_test, best_model_save_path, history):
    """Évalue le modèle sur l'ensemble de test et affiche les résultats."""
    logging.info(f"Chargement meilleurs poids depuis {best_model_save_path} pour évaluation finale...")
    try:
        if os.path.exists(best_model_save_path):
            # Le plus sûr est de recharger le modèle entier sauvegardé par le checkpoint
            # car il contient l'état de l'optimiseur au meilleur moment aussi
            model = tf.keras.models.load_model(best_model_save_path)
            logging.info("Meilleur modèle (checkpoint) chargé.")
        else:
             logging.warning(f"Fichier checkpoint '{best_model_save_path}' non trouvé pour évaluation.")
             # Le modèle actuel a les poids de la fin de l'entraînement,
             # ou ceux restaurés par EarlyStopping si restore_best_weights=True

    except Exception as e:
        logging.exception(f"Avertissement: Erreur chargement poids/modèle pour évaluation: {e}")

    if x_test is not None and x_test.size > 0:
        logging.info("Évaluation sur l'ensemble de test...")
        loss, accuracy = model.evaluate(x_test, y_test, verbose=0)
        logging.info(f"Loss Test: {loss:.4f}, Accuracy Test: {accuracy:.4f}")
    else:
        logging.info("Pas d'ensemble de test pour l'évaluation.")

    # --- Visualisation (inchangée) ---
    if history and history.history and 'loss' in history.history and 'val_loss' in history.history:
        plt.figure(figsize=(12, 5))
        plt.subplot(1, 2, 1)
        plt.plot(history.history['loss'], label='Train Loss')
        plt.plot(history.history['val_loss'], label='Val Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss'); plt.legend(); plt.title('Loss')
        plt.subplot(1, 2, 2)
        if 'accuracy' in history.history and 'val_accuracy' in history.history:
            plt.plot(history.history['accuracy'], label='Train Accuracy')
            plt.plot(history.history['val_accuracy'], label='Val Accuracy')
            plt.xlabel('Epoch')
            plt.ylabel('Accuracy'); plt.legend(); plt.title('Accuracy')
        else: plt.title('Accuracy non disponible')
        plt.tight_layout(); plt.show()
    else: logging.warning("Pas d'historique d'entraînement à afficher.")


def save_model_and_data(model, all_keypoints, old_keypoints_path, all_labels, old_labels_path, model_save_path):
    """Sauvegarde les données et le modèle final."""
    logging.info(f"Sauvegarde données NON AUGMENTÉES ({all_keypoints.shape[0]}) dans {old_keypoints_path}, {old_labels_path}")
    np.save(old_keypoints_path, all_keypoints) # Sauvegarder les données originales
    np.save(old_labels_path, all_labels)
    logging.info(f"Sauvegarde modèle final dans {model_save_path}")
    try:
        model.save(model_save_path)
        logging.info("Modèle final sauvegardé.")
    except Exception as e:
        logging.exception(f"Erreur sauvegarde modèle final: {e}")

# --- Exécution Principale (inchangée dans sa structure globale) ---
def main():
    # --- Supprimer anciens fichiers ---
    if os.path.exists(OLD_DATA_KEYPOINTS_FILE) or os.path.exists(OLD_DATA_LABELS_FILE):
        logging.warning("-" * 60 + "\nATTENTION : Fichiers old_*.npy existent.\n" + "-" * 60)
        # user_input = input("Voulez-vous les supprimer et continuer ? (o/N) : ").lower()
        # if user_input == 'o': # Pour tests, on supprime automatiquement
        if True: # ATTENTION: Supprime automatiquement pour ce test
            logging.warning("Suppression automatique des anciens fichiers .npy...")
            try:
                if os.path.exists(OLD_DATA_KEYPOINTS_FILE): os.remove(OLD_DATA_KEYPOINTS_FILE)
                if os.path.exists(OLD_DATA_LABELS_FILE): os.remove(OLD_DATA_LABELS_FILE)
                logging.info("Anciens fichiers supprimés.")
            except Exception as e:
                logging.error(f"Erreur suppression: {e}. Veuillez supprimer manuellement."); exit()
        # else: logging.info("Arrêt."); exit()

    # --- Charger Vocabulaire ---
    vocabulaire = load_vocabulary(VOCABULARY_FILE)
    if vocabulaire is None: exit()
    logging.info(f"Vocabulaire ({len(vocabulaire)} mots): {list(vocabulaire.keys())}")
    num_classes = len(vocabulaire)

    # --- Mapping Automatique ---
    video_label_mapping = {}
    new_data_files = []
    logging.info(f"Recherche fichiers dans : '{DATA_DIR}'")
    if not os.path.exists(DATA_DIR): logging.error(f"Erreur: Dossier '{DATA_DIR}' n'existe pas."); exit()
    for filename in sorted(os.listdir(DATA_DIR)):
        if filename.endswith("_keypoints.npy"):
            found_label = None
            sorted_vocab = sorted(vocabulaire.keys(), key=len, reverse=True)
            for vocab_word in sorted_vocab:
                if filename.lower().startswith(vocab_word + "_"):
                    found_label = vocab_word
                    break
            if found_label:
                video_label_mapping[filename] = found_label
                new_data_files.append(filename)
            else:
                 logging.warning(f"  -> Avertissement: Aucun label pour '{filename}'. Ignoré.")
    if not new_data_files: logging.info("\nAucun fichier *_keypoints.npy correspondant trouvé. Arrêt."); exit()
    logging.info(f"{len(new_data_files)} fichiers .npy trouvés pour traitement.")

    # --- Paramètres ---
    model_path = os.path.join(MODEL_DIR, MODEL_FILENAME)
    best_model_path = os.path.join(MODEL_DIR, BEST_MODEL_FILENAME)
    old_keypoints_path = OLD_DATA_KEYPOINTS_FILE
    old_labels_path = OLD_DATA_LABELS_FILE
    epochs_tuning = 50
    epochs_final = 100
    batch_size = 16

    # --- Charger et préparer TOUTES les données AVANT le tuning ---
    new_keypoints, new_labels = load_data(DATA_DIR, new_data_files, vocabulaire, video_label_mapping, FIXED_LENGTH, num_classes)
    if new_keypoints is None or new_labels is None: logging.error("Aucune donnée valide. Arrêt."); return
    all_keypoints, all_labels = combine_data(old_keypoints_path, old_labels_path, new_keypoints, new_labels, num_classes)
    if all_keypoints is None or all_labels is None: logging.error("Erreur combinaison données. Arrêt."); return

    logging.info("Distribution classes après combinaison:")
    label_indices = np.argmax(all_labels, axis=1); class_counts = Counter(label_indices)
    for class_index, count in sorted(class_counts.items()):
        class_name = next((name for name, idx in vocabulaire.items() if idx == class_index), f"Classe_{class_index}")
        logging.info(f"  Classe '{class_name}' ({class_index}): {count} échantillons")

    logging.info("Application augmentation (bruit)...")
    augmented_keypoints = augment_data(all_keypoints)
    logging.info(f"Augmentation terminée. Shape: {augmented_keypoints.shape}")

    x_train, x_val, x_test, y_train, y_val, y_test = split_data(augmented_keypoints, all_labels)
    if x_train is None or x_val is None:
        logging.error("Échec division données. Arrêt.")
        np.save(old_keypoints_path, all_keypoints); np.save(old_labels_path, all_labels)
        logging.info(f"Données brutes ({all_keypoints.shape[0]}) sauvegardées."); return

    # --- Phase de Tuning avec Keras Tuner ---
    input_shape = (FIXED_LENGTH, FEATURES_PER_FRAME)

    # Utiliser Hyperband pour potentiellement accélérer
    tuner = kt.Hyperband(
        lambda hp: build_model(hp, input_shape, num_classes),
        objective='val_accuracy',
        max_epochs=epochs_tuning, # Max epochs par essai DANS Hyperband
        factor=3, # Facteur de réduction pour chaque round d'Hyperband
        hyperband_iterations=1, # Nombre de fois où lancer le processus Hyperband
        directory='keras_tuner_dir_hyperband', # Dossier différent
        project_name='sign_language_lstm_reg' # Nom projet
    )

    # tuner = kt.RandomSearch( # Alternative si Hyperband pose problème
    #     lambda hp: build_model(hp, input_shape, num_classes),
    #     objective='val_accuracy',
    #     max_trials=25, # Nombre d'essais pour RandomSearch
    #     executions_per_trial=1,
    #     directory='keras_tuner_dir_random',
    #     project_name='sign_language_lstm_reg'
    # )

    tuner.search_space_summary()

    logging.info("--- Début Recherche Hyperparamètres ---")
    # EarlyStopping pour la RECHERCHE (peut être moins stricte que pour l'entraînement final)
    search_callbacks = [ EarlyStopping(monitor='val_loss', patience=10, verbose=1) ] # Surveiller val_loss
    tuner.search(
        x_train, y_train,
        epochs=epochs_tuning, # Redondant avec max_epochs pour Hyperband, mais nécessaire pour l'API
        batch_size=batch_size,
        validation_data=(x_val, y_val),
        callbacks=search_callbacks
    )
    logging.info("--- Fin Recherche Hyperparamètres ---")

    # Récupérer les meilleurs hyperparamètres
    try:
        best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]
        logging.info("Meilleurs hyperparamètres trouvés :")
        for hp_name in best_hps.space:
             logging.info(f"  - {hp_name.name}: {best_hps.get(hp_name.name)}")
    except Exception as e:
        logging.exception(f"Erreur récupération meilleurs HP: {e}. Impossible de continuer l'entraînement final.")
        return # Arrêter si on ne peut pas obtenir les meilleurs HP

    # --- Phase d'Entraînement Final ---
    logging.info("--- Début Entraînement Final avec Meilleurs HP ---")
    best_model = tuner.hypermodel.build(best_hps)

    final_history = compile_and_train_model(
        best_model,
        x_train, y_train,
        x_val, y_val,
        epochs=epochs_final,
        batch_size=batch_size,
        best_model_save_path=best_model_path # Sauvegarde le meilleur de CET entraînement
    )

    # --- Évaluation Finale ---
    logging.info("--- Évaluation Finale sur Ensemble Test ---")
    evaluate_model(best_model, x_test, y_test, best_model_path, final_history)

    # --- Sauvegarde Finale ---
    logging.info("--- Sauvegarde Finale Données et Modèle ---")
    # Le modèle 'best_model' contient les poids chargés par ModelCheckpoint ou EarlyStopping
    save_model_and_data(best_model, all_keypoints, old_keypoints_path, all_labels, old_labels_path, model_path)

    logging.info("--- Fin du script ---")

if __name__ == "__main__":
    main()