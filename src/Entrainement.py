import numpy as np
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import CategoricalCrossentropy
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import os
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from collections import Counter
import logging  # Import du module logging

# --- Configuration du logging ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Constants ---
VOCABULARY_FILE = "vocabulaire.txt"
DATA_DIR = "data"
MODEL_DIR = "models"
MODEL_FILENAME = "model.h5"
BEST_MODEL_FILENAME = "best_model.h5"
OLD_DATA_KEYPOINTS_FILE = "old_keypoints.npy"
OLD_DATA_LABELS_FILE = "old_labels.npy"

FIXED_LENGTH = 46
NUM_POSE_KEYPOINTS = 4
NUM_HAND_KEYPOINTS = 21
NUM_COORDS = 3
FEATURES_PER_FRAME = (NUM_POSE_KEYPOINTS * NUM_COORDS) + \
                     (NUM_HAND_KEYPOINTS * NUM_COORDS) * 2

logging.info(f"Nombre de features par frame attendu : {FEATURES_PER_FRAME}")

# --- Utility Functions ---
def create_sequential_model(input_shape, num_classes):
    """Crée un modèle basé sur LSTM pour la reconnaissance de séquences."""
    logging.info(f"Création d'un modèle séquentiel avec input_shape={input_shape} et num_classes={num_classes}")
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=input_shape, name='lstm_input'),
        tf.keras.layers.LSTM(64, return_sequences=True, name='lstm_1'),
        tf.keras.layers.Dropout(0.3, name='dropout_1'),
        tf.keras.layers.LSTM(128, return_sequences=True, name='lstm_2'),
        tf.keras.layers.Dropout(0.3, name='dropout_2'),
        tf.keras.layers.LSTM(64, return_sequences=False, name='lstm_3'),
        tf.keras.layers.Dropout(0.3, name='dropout_3'),
        tf.keras.layers.Dense(64, activation='relu', name='dense_1', kernel_regularizer=tf.keras.regularizers.l2(0.001)),
        tf.keras.layers.Dropout(0.4, name='dropout_4'),
        tf.keras.layers.Dense(num_classes, activation='softmax', name='output_layer')
    ])
    return model

def load_vocabulary(filepath):
    """Charge le vocabulaire à partir d'un fichier texte."""
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
    """Applique du bruit simple aux points clés d'un batch."""
    augmented_batch = keypoints_batch.copy()
    num_samples, seq_len, num_features = augmented_batch.shape

    if num_features != FEATURES_PER_FRAME:
        logging.warning(f"Avertissement (Augmentation): Attend {FEATURES_PER_FRAME} features, reçu {num_features}. Augmentation ignorée.")
        return augmented_batch

    noise_std_dev = 0.005
    noise = np.random.normal(0, noise_std_dev, size=augmented_batch.shape)
    augmented_batch += noise
    return augmented_batch

def load_data(data_dir, new_data_files, vocabulaire, video_label_mapping, fixed_length, num_classes):
    """Charge les nouvelles données à partir des fichiers .npy."""
    logging.info("Chargement des nouvelles données...")
    new_keypoints_list = []
    new_labels_list = []

    for data_file in new_data_files:
        data_path = os.path.join(data_dir, data_file)
        label_name = video_label_mapping.get(data_file)

        if label_name is None or label_name.lower() not in vocabulaire:
            logging.warning(f"Label '{label_name}' invalide pour {data_file}. Ignoré.")
            continue

        try:
            keypoints = np.load(data_path)
            if keypoints.ndim == 3 and keypoints.shape[1] == (NUM_POSE_KEYPOINTS + 2 * NUM_HAND_KEYPOINTS) and keypoints.shape[2] == NUM_COORDS:
                keypoints = keypoints.reshape(keypoints.shape[0], -1)
            elif keypoints.ndim != 2 or keypoints.shape[1] != FEATURES_PER_FRAME:
                logging.warning(f"Forme initiale inattendue {keypoints.shape} pour {data_file}. Ignoré.")
                continue

            current_len = keypoints.shape[0]
            if current_len > fixed_length:
                keypoints = keypoints[:fixed_length, :]
            elif current_len < fixed_length:
                padding = np.zeros((fixed_length - current_len, keypoints.shape[1]), dtype=keypoints.dtype)
                keypoints = np.concatenate([keypoints, padding], axis=0)

            if keypoints.shape == (fixed_length, FEATURES_PER_FRAME):
                new_keypoints_list.append(keypoints)
                label_index = vocabulaire[label_name.lower()]
                label_one_hot = tf.keras.utils.to_categorical(label_index, num_classes=num_classes)
                new_labels_list.append(label_one_hot)
                logging.info(f"  -> Chargé: {data_file} (Label: {label_name}), Shape: {keypoints.shape}")
            else:
                logging.warning(f"Forme incorrecte {keypoints.shape} après padding/truncation pour {data_file}. Ignoré.")
        except FileNotFoundError:
            logging.error(f"Fichier {data_path} non trouvé. Ignoré.")
        except ValueError as ve:
            logging.error(f"Erreur valeur traitement {data_file}: {ve}. Ignoré.")
        except Exception as e:
            logging.exception(f"Erreur générale chargement {data_file}: {e}. Ignoré.")

    if not new_keypoints_list:
        logging.warning("Aucune nouvelle donnée valide.")
        return None, None

    new_keypoints = np.array(new_keypoints_list)
    new_labels = np.array(new_labels_list)
    logging.info(f"Nouvelles données chargées: {new_keypoints.shape[0]} séquences.")
    return new_keypoints, new_labels

def combine_data(old_keypoints_path, old_labels_path, new_keypoints, new_labels, num_classes):
    """Combine les anciennes et nouvelles données."""
    try:
        old_keypoints = np.load(old_keypoints_path)
        old_labels = np.load(old_labels_path)
        logging.info(f"Anciennes données chargées: {old_keypoints.shape[0]}. Shape K: {old_keypoints.shape}, L: {old_labels.shape}")
        shapes_compatible = old_keypoints.shape[1:] == new_keypoints.shape[1:]
        num_classes_compatible = old_labels.shape[1] == num_classes
        if shapes_compatible and num_classes_compatible:
            all_keypoints = np.concatenate([old_keypoints, new_keypoints], axis=0)
            all_labels = np.concatenate([old_labels, new_labels], axis=0)
            logging.info(f"Données combinées: {all_keypoints.shape[0]}.")
        else:
            if not shapes_compatible:
                logging.error(f"Incompatibilité forme K: Anciens {old_keypoints.shape} vs Nouveaux {new_keypoints.shape}.")
            if not num_classes_compatible:
                logging.error(f"Incompatibilité classes L: Anciens {old_labels.shape[1]} vs Actuel {num_classes}.")
            logging.warning("Utilisation nouvelles données seulement.")
            all_keypoints = new_keypoints
            all_labels = new_labels
    except FileNotFoundError:
        logging.info("Pas d'anciennes données. Utilisation des nouvelles données seulement.")
        all_keypoints = new_keypoints
        all_labels = new_labels
    except Exception as e:
        logging.exception(f"Erreur chargement/concat anciennes données: {e}. Utilisation nouvelles données.")
        all_keypoints = new_keypoints
        all_labels = new_labels

    return all_keypoints, all_labels

def split_data(all_keypoints, all_labels):
    """Divise les données en ensembles d'entraînement, de validation et de test."""
    if all_keypoints.shape[0] < 5:
        logging.warning("Pas assez de données pour diviser.")
        return None, None, None, None, None, None

    try:
        x_train, x_temp, y_train, y_temp = train_test_split(
            all_keypoints, all_labels, test_size=0.3, random_state=42
        )

        if x_temp.shape[0] < 2:
            logging.warning("Pas assez de données pour diviser en validation et test. Utilisation pour la validation.")
            x_val, x_test, y_val, y_test = x_temp, np.array([]), y_temp, np.array([])
        else:
            x_val, x_test, y_val, y_test = train_test_split(
                x_temp, y_temp, test_size=0.5, random_state=42
            )
        logging.info(f"Taille Train: {x_train.shape[0]}, Val: {x_val.shape[0]}, Test: {x_test.shape[0]}")
        return x_train, x_val, x_test, y_train, y_val, y_test

    except ValueError as e:
        logging.exception(f"Erreur lors de la division des données: {e}")
        return None, None, None, None, None, None

def compile_and_train_model(model, x_train, y_train, x_val, y_val, epochs, batch_size, best_model_save_path):
    """Compile et entraîne le modèle."""
    optimizer = Adam(learning_rate=0.0005)
    loss_fn = CategoricalCrossentropy()
    model.compile(optimizer=optimizer, loss=loss_fn, metrics=['accuracy'])
    model.summary()
    early_stopping = EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True, verbose=1)
    os.makedirs(os.path.dirname(best_model_save_path), exist_ok=True)
    checkpoint = ModelCheckpoint(best_model_save_path, save_best_only=True, monitor='val_loss', mode='min', verbose=1)
    logging.info("Début entraînement...")
    history = model.fit(x_train, y_train, epochs=epochs, batch_size=batch_size, validation_data=(x_val, y_val), callbacks=[early_stopping, checkpoint], shuffle=True)
    return history

def evaluate_model(model, x_test, y_test, best_model_save_path, history):
    """Évalue le modèle sur l'ensemble de test et affiche les résultats."""
    # --- Charger Meilleurs Poids ---
    logging.info(f"Chargement meilleurs poids depuis {best_model_save_path}")
    try:
        if os.path.exists(best_model_save_path):
            model.load_weights(best_model_save_path)
        elif not EarlyStopping.restore_best_weights: #early_stopping.restore_best_weights ne fonctionne pas car earlystopping n'est pas global
            logging.warning("Avertissement: Pas de checkpoint et restore_best_weights=False.")
    except Exception as e:
        logging.exception(f"Avertissement: Erreur chargement poids: {e}")

    # --- Évaluer ---
    if x_test.size > 0:
        logging.info("Évaluation test...")
        loss, accuracy = model.evaluate(x_test, y_test, verbose=0)
        logging.info(f"Loss test: {loss:.4f}, Acc test: {accuracy:.4f}")
    else:
        logging.info("Pas d'ensemble test.")

    # --- Visualiser ---
    if history and history.history and 'loss' in history.history and 'val_loss' in history.history:
        plt.figure(figsize=(12, 5))
        plt.subplot(1, 2, 1)
        plt.plot(history.history['loss'], label='Train Loss')
        plt.plot(history.history['val_loss'], label='Val Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.title('Loss')

        plt.subplot(1, 2, 2)
        if 'accuracy' in history.history and 'val_accuracy' in history.history:
            plt.plot(history.history['accuracy'], label='Train Accuracy')
            plt.plot(history.history['val_accuracy'], label='Val Accuracy')
            plt.xlabel('Epoch')
            plt.ylabel('Accuracy')
            plt.legend()
            plt.title('Accuracy')
        else:
            plt.title('Accuracy non disponible')

        plt.tight_layout()
        plt.show()
        logging.info(f"Historique: {history.history}")
    else:
        logging.warning("Pas d'historique à afficher.")

def save_model_and_data(model, all_keypoints, old_keypoints_path, all_labels, old_labels_path, model_save_path):
    """Sauvegarde les données et le modèle."""
    # --- Sauvegarder Données et Modèle ---
    logging.info(f"Sauvegarde données ({all_keypoints.shape[0]}) dans {old_keypoints_path}, {old_labels_path}")
    np.save(old_keypoints_path, all_keypoints)
    np.save(old_labels_path, all_labels)
    logging.info(f"Sauvegarde modèle final dans {model_save_path}")
    try:
        model.save(model_save_path)
        logging.info("Modèle sauvegardé.")
    except Exception as e:
        logging.exception(f"Erreur sauvegarde modèle: {e}")

def train_model(data_dir, model_save_path, best_model_save_path, old_keypoints_path, old_labels_path, new_data_files, vocabulaire, video_label_mapping, epochs=10, batch_size=32, fixed_length=46):
    """Entraîne le modèle de manière incrémentale avec validation et augmentation."""

    num_classes = len(vocabulaire)
    input_shape = (fixed_length, FEATURES_PER_FRAME)

    # --- Charger/Créer Modèle ---
    if os.path.exists(model_save_path):
        try:
            model = tf.keras.models.load_model(model_save_path)
            logging.info(f"Modèle chargé depuis {model_save_path}")
            if model.input_shape[1:] != input_shape:
                logging.warning(f"Avertissement: Input shape modèle {model.input_shape[1:]} != attendu {input_shape}. Nouveau modèle créé.")
                model = create_sequential_model(input_shape, num_classes)
            elif model.layers[-1].output_shape[-1] != num_classes:
                logging.warning(f"Avertissement: Incompatibilité couche sortie ({model.layers[-1].output_shape[-1]}) vs vocabulaire ({num_classes}). Nouveau modèle créé.")
                model = create_sequential_model(input_shape, num_classes)
            else:
                logging.info("Structure modèle compatible.")
        except Exception as e:
            logging.exception(f"Erreur chargement modèle: {e}. Nouveau modèle créé.")
            model = create_sequential_model(input_shape, num_classes)
    else:
        logging.info("Nouveau modèle créé.")
        model = create_sequential_model(input_shape, num_classes)

    # --- Charger Nouvelles Données ---
    new_keypoints, new_labels = load_data(data_dir, new_data_files, vocabulaire, video_label_mapping, fixed_length, num_classes)
    if new_keypoints is None or new_labels is None:
        logging.warning("Aucune nouvelle donnée valide. Arrêt de l'entraînement.")
        return model

    # --- Combiner Données ---
    all_keypoints, all_labels = combine_data(old_keypoints_path, old_labels_path, new_keypoints, new_labels, num_classes)

    # --- Afficher Distribution (pour info) ---
    logging.info("Distribution classes avant division:")
    label_indices = np.argmax(all_labels, axis=1)
    class_counts = Counter(label_indices)
    for class_index, count in sorted(class_counts.items()):
        class_name = next((name for name, idx in vocabulaire.items() if idx == class_index), f"Classe_{class_index}")
        logging.info(f"  Classe '{class_name}' ({class_index}): {count} échantillons")

    # --- Augmentation Données ---
    logging.info("Application augmentation (bruit)...")
    augmented_keypoints = augment_data(all_keypoints)
    logging.info(f"Augmentation terminée. Shape: {augmented_keypoints.shape}")

    # --- Diviser Données ---
    x_train, x_val, x_test, y_train, y_val, y_test = split_data(augmented_keypoints, all_labels)

    if x_train is None:
        logging.warning("Pas assez de données pour diviser. Sauvegarde des données et arrêt de l'entraînement.")
        save_model_and_data(model, all_keypoints, old_keypoints_path, all_labels, old_labels_path, model_save_path)
        logging.info(f"Données ({all_keypoints.shape[0]}) sauvegardées.")
        return model

    # --- Compiler et Entraîner ---
    history = compile_and_train_model(model, x_train, y_train, x_val, y_val, epochs, batch_size, best_model_save_path)

    # --- Évaluer ---
    evaluate_model(model, x_test, y_test, best_model_save_path, history)

    # --- Sauvegarder ---
    save_model_and_data(model, all_keypoints, old_keypoints_path, all_labels, old_labels_path, model_save_path)

    return model

# --- Exécution Principale ---
def main():
    # --- Supprimer anciens fichiers si nécessaire ---
    if os.path.exists(OLD_DATA_KEYPOINTS_FILE) or os.path.exists(OLD_DATA_LABELS_FILE):
        logging.warning("-" * 60 + "\nATTENTION : Fichiers old_*.npy existent.\nLe format a peut-être changé (138 features).\n" + "-" * 60)
        user_input = input("Voulez-vous les supprimer et continuer ? (o/N) : ").lower()
        if user_input == 'o':
            try:
                if os.path.exists(OLD_DATA_KEYPOINTS_FILE):
                    os.remove(OLD_DATA_KEYPOINTS_FILE)
                    logging.info("Ancien fichier keypoints supprimé.")
                if os.path.exists(OLD_DATA_LABELS_FILE):
                    os.remove(OLD_DATA_LABELS_FILE)
                    logging.info("Ancien fichier labels supprimé.")
                logging.info("Anciens fichiers supprimés.")
            except Exception as e:
                logging.error(f"Erreur suppression: {e}. Veuillez supprimer manuellement.")
                exit()
        else:
            logging.info("Arrêt. Veuillez gérer les anciens fichiers.")
            exit()

    # --- Charger Vocabulaire ---
    vocabulaire = load_vocabulary(VOCABULARY_FILE)
    if vocabulaire is None:
        exit()
    logging.info(f"Vocabulaire ({len(vocabulaire)} mots): {vocabulaire}")
    num_classes = len(vocabulaire)

    # --- Mapping Automatique ---
    video_label_mapping = {}
    new_data_files = []
    logging.info(f"Recherche fichiers dans : '{DATA_DIR}'")
    if not os.path.exists(DATA_DIR):
        logging.error(f"Erreur: Dossier '{DATA_DIR}' n'existe pas.")
        exit()

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
                logging.info(f"  -> Trouvé: '{filename}' -> Label: '{found_label}'")
            else:
                logging.warning(f"  -> Avertissement: Aucun label pour '{filename}'. Ignoré.")
    if not new_data_files:
        logging.info("\nAucun fichier *_keypoints.npy correspondant trouvé. Arrêt.")
        exit()
    logging.info(f"Mapping: {video_label_mapping}")
    logging.info(f"{len(new_data_files)} fichiers pour entraînement.")

    # --- Paramètres ---
    model_path = os.path.join(MODEL_DIR, MODEL_FILENAME)
    best_model_path = os.path.join(MODEL_DIR, BEST_MODEL_FILENAME)
    old_keypoints_path = OLD_DATA_KEYPOINTS_FILE
    old_labels_path = OLD_DATA_LABELS_FILE
    epochs = 50
    batch_size = 16

    # --- Entraîner ---
    trained_model = train_model(
        DATA_DIR, model_path, best_model_path, old_keypoints_path, old_labels_path,
        new_data_files, vocabulaire, video_label_mapping,
        epochs, batch_size, FIXED_LENGTH
    )

    logging.info("--- Fin du script ---")

if __name__ == "__main__":
    main()