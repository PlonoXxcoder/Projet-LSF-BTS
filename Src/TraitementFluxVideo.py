# RealTimePredict.py (ou TraitementFluxVIdeo.py)
# -*- coding: utf-8 -*-
import cv2
import numpy as np
import tensorflow as tf
# Importer les fonctions CNN nécessaires (adaptez si vous changez de modèle CNN)
from tensorflow.keras.applications import MobileNetV2, ResNet50, EfficientNetB0 # type: ignore
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input as mobilenet_preprocess # type: ignore
from tensorflow.keras.applications.resnet50 import preprocess_input as resnet_preprocess # type: ignore
from tensorflow.keras.applications.efficientnet import preprocess_input as efficientnet_preprocess # type: ignore
from tensorflow.keras.preprocessing import image # type: ignore
from tensorflow.keras.models import Model # type: ignore
from tensorflow.keras.layers import GlobalAveragePooling2D # type: ignore
import os
import logging
import time
from collections import deque, Counter
try:
    import config # Importe la configuration globale
except ImportError:
    print("ERREUR: Impossible d'importer config.py. Assurez-vous qu'il existe et qu'il est accessible.")
    import sys
    sys.exit(1)


# --- Configuration du logging ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - [%(threadName)s] - %(message)s')
print("DEBUG: Logging configured for TraitementFluxVideo.py")

# --- ANSI escape codes (utilisé si exécuté en console directe) ---
class Colors:
    RESET = '\x1b[0m'
    BRIGHT_YELLOW = '\x1b[93m'
    BRIGHT_GREEN = '\x1b[92m'
    RED = '\x1b[31m'
    GREEN = '\x1b[32m'
    BLUE = '\x1b[34m'
    # Couleurs pour OpenCV (BGR)
    CV_BLUE = (255, 0, 0)
    CV_GREEN = (0, 255, 0)
    CV_RED = (0, 0, 255)
    CV_WHITE = (255, 255, 255)
    CV_BLACK = (0, 0, 0)
    CV_YELLOW = (0, 255, 255)
    CV_CYAN = (255, 255, 0)
    CV_MAGENTA = (255, 0, 255)
    CV_ORANGE = (0, 165, 255)
    CV_PURPLE = (128, 0, 128)
    CV_LIGHT_GREY = (211, 211, 211)
    CV_DARK_GREY = (169, 169, 169)

# --- Vérification Mode ---
if not config.USE_CNN_FEATURES:
    logging.error("ERREUR: Ce script est conçu pour USE_CNN_FEATURES=True dans config.py.")
    print("DEBUG: USE_CNN_FEATURES is False, exiting.")
    exit()
else:
    print("DEBUG: USE_CNN_FEATURES is True.")

# --- Constants & Configuration (Utilisation de config ACTIVE) ---
MODEL_PATH = os.path.join(config.BASE_DIR, config.MODEL_DIR, config.ACTIVE_MODEL_FILENAME) # Chemin complet modèle
VOCABULARY_PATH = config.VOCABULARY_PATH
FIXED_LENGTH = config.FIXED_LENGTH
FEATURE_DIM = config.ACTIVE_FEATURE_DIM
CNN_MODEL_CHOICE = config.CNN_MODEL_CHOICE
CNN_INPUT_SHAPE = config.CNN_INPUT_SHAPE
CAPTURE_SOURCE = config.CAPTURE_SOURCE
FRAMES_TO_SKIP = config.FRAMES_TO_SKIP
PREDICTION_THRESHOLD = config.PREDICTION_THRESHOLD
SMOOTHING_WINDOW_SIZE = config.CAPTURE_SMOOTHING_WINDOW_SIZE
TOP_N = config.CAPTURE_TOP_N
CONF_THRESH_GREEN = config.CAPTURE_CONF_THRESH_GREEN
CONF_THRESH_YELLOW = config.CAPTURE_CONF_THRESH_YELLOW
MAX_FRAME_WIDTH = config.CAPTURE_MAX_FRAME_WIDTH # Pour redimensionner l'affichage si webcam trop grande

# --- Logging de la configuration active ---
logging.info("--- Configuration Prédiction Temps Réel (CNN+LSTM) ---")
logging.info(f"Modèle LSTM utilisé: {MODEL_PATH}")
logging.info(f"Vocabulaire: {VOCABULARY_PATH}")
logging.info(f"Source Capture: {CAPTURE_SOURCE}")
logging.info(f"Modèle CNN Base: {CNN_MODEL_CHOICE}")
logging.info(f"Shape Entrée CNN: {CNN_INPUT_SHAPE}")
logging.info(f"Dimension Features CNN: {FEATURE_DIM}")
logging.info(f"Longueur Séquence LSTM: {FIXED_LENGTH}")
logging.info(f"Frames à Sauter (Performance): {FRAMES_TO_SKIP}")
logging.info(f"Seuil Confiance Prédiction: {PREDICTION_THRESHOLD}")
logging.info(f"Fenêtre Lissage Affichage: {SMOOTHING_WINDOW_SIZE}")
logging.warning("!!! La performance dépendra fortement de la présence d'un GPU et de FRAMES_TO_SKIP !!!")
logging.info("-----------------------------------------------")
print("DEBUG: Configuration loaded and logged.")

# --- Variables Globales pour Modèles ---
cnn_feature_extractor_model = None
preprocess_function = None
cnn_target_size = None
lstm_prediction_model = None
vocabulaire = None
index_to_word = None

# --- Fonctions Utilitaires ---

def load_vocabulary(filepath=VOCABULARY_PATH):
    """Charge le vocabulaire depuis un fichier."""
    print(f"DEBUG: Attempting to load vocabulary from {filepath}")
    vocab = {}
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            for line_num, line in enumerate(f, 1):
                parts = line.strip().split(":")
                if len(parts) == 2 and parts[0] and parts[1].isdigit():
                    mot, index = parts
                    vocab[mot.lower()] = int(index)
                elif line.strip():
                    logging.warning(f"Ligne {line_num} ignorée dans '{filepath}': '{line.strip()}'")
                    print(f"DEBUG: Vocab line {line_num} ignored: '{line.strip()}'")
    except FileNotFoundError:
        logging.error(f"Erreur: Fichier vocabulaire '{filepath}' non trouvé.")
        print(f"DEBUG: Vocab file not found: {filepath}")
        return None
    except Exception as e:
        logging.exception(f"Erreur inattendue lors du chargement du vocabulaire '{filepath}': {e}")
        print(f"DEBUG: Unexpected error loading vocab: {e}")
        return None
    if not vocab:
        logging.warning(f"Avertissement: Vocabulaire chargé depuis '{filepath}' est vide.")
        print(f"DEBUG: Loaded vocabulary is empty: {filepath}")
        # Ne retourne pas None ici, un vocabulaire vide peut être voulu dans certains cas tests
        # Mais on le signale.
    logging.info(f"Vocabulaire chargé depuis '{filepath}' avec {len(vocab)} mots.")
    print(f"DEBUG: Vocabulary loaded successfully ({len(vocab)} words).")
    return vocab

def load_models_and_preprocessing():
    """ Charge le modèle CNN d'extraction et le modèle LSTM de prédiction. """
    global cnn_feature_extractor_model, preprocess_function, cnn_target_size, lstm_prediction_model
    print("DEBUG: Attempting to load models...")

    # --- Charger Modèle CNN pour Extraction ---
    model_name = CNN_MODEL_CHOICE
    input_shape = CNN_INPUT_SHAPE
    cnn_target_size = input_shape[:2] # (height, width)
    logging.info(f"Chargement CNN base model: {model_name} pour extraction...")
    print(f"DEBUG: Loading CNN base model: {model_name}")
    try:
        if model_name == 'MobileNetV2':
            base = MobileNetV2(input_shape=input_shape, include_top=False, weights='imagenet')
            preprocess_function = mobilenet_preprocess
        elif model_name == 'ResNet50':
             base = ResNet50(input_shape=input_shape, include_top=False, weights='imagenet')
             preprocess_function = resnet_preprocess
        elif model_name == 'EfficientNetB0':
             base = EfficientNetB0(input_shape=input_shape, include_top=False, weights='imagenet')
             preprocess_function = efficientnet_preprocess
        else:
            print(f"DEBUG: Unsupported CNN model: {model_name}")
            raise ValueError(f"Modèle CNN non supporté: {model_name}")

        output = GlobalAveragePooling2D()(base.output)
        cnn_feature_extractor_model = Model(inputs=base.input, outputs=output, name=f"{model_name}_FeatureExtractor")
        print(f"DEBUG: CNN model {model_name} structure created. Initializing...")
        dummy_input = np.zeros((1,) + input_shape, dtype=np.float32)
        cnn_feature_extractor_model.predict(dummy_input, verbose=0)
        logging.info(f"Modèle CNN {model_name} chargé.")
        print(f"DEBUG: CNN model {model_name} loaded and initialized.")

    except Exception as e:
        logging.exception(f"ERREUR CRITIQUE chargement modèle CNN {model_name}: {e}")
        print(f"DEBUG: CRITICAL ERROR loading CNN model {model_name}: {e}")
        return False

    # --- Charger Modèle LSTM pour Prédiction ---
    logging.info(f"Chargement modèle prédiction LSTM depuis: {MODEL_PATH}...")
    print(f"DEBUG: Loading LSTM model from: {MODEL_PATH}")
    try:
        if not os.path.exists(MODEL_PATH):
             logging.error(f"Fichier modèle LSTM non trouvé : {MODEL_PATH}")
             print(f"DEBUG: LSTM model file not found: {MODEL_PATH}")
             return False
        lstm_prediction_model = tf.keras.models.load_model(MODEL_PATH)
        logging.info(f"Modèle LSTM chargé depuis {MODEL_PATH}")
        print(f"DEBUG: LSTM model loaded from {MODEL_PATH}. Checking shape...")
        # Vérification shape entrée LSTM
        expected_lstm_shape = lstm_prediction_model.input_shape
        logging.info(f"Shape entrée attendue par modèle LSTM chargé: {expected_lstm_shape}")
        print(f"DEBUG: Expected LSTM input shape: {expected_lstm_shape}")
        if len(expected_lstm_shape) != 3 or \
           expected_lstm_shape[1] is not None and expected_lstm_shape[1] != FIXED_LENGTH or \
           expected_lstm_shape[2] is not None and expected_lstm_shape[2] != FEATURE_DIM:
             logging.error(
                 f"{Colors.RED}Incohérence Shape LSTM! Modèle attend {expected_lstm_shape}, "
                 f"Script Config (None, {FIXED_LENGTH}, {FEATURE_DIM}){Colors.RESET}"
             )
             print(f"DEBUG: LSTM Shape Mismatch! Model={expected_lstm_shape}, Config=(None, {FIXED_LENGTH}, {FEATURE_DIM})")
             return False
        else:
             logging.info("Shape entrée modèle LSTM OK vs constantes script.")
             print(f"DEBUG: LSTM input shape OK. Initializing...")
             # Prédiction dummy pour forcer l'initialisation
             dummy_lstm_input = np.zeros((1, FIXED_LENGTH, FEATURE_DIM), dtype=np.float32)
             lstm_prediction_model.predict(dummy_lstm_input, verbose=0)
             logging.info("Modèle LSTM initialisé.")
             print("DEBUG: LSTM model initialized.")

    except Exception as e:
        logging.exception(f"Erreur chargement/init modèle LSTM {MODEL_PATH}: {e}")
        print(f"DEBUG: Error loading/initializing LSTM model: {e}")
        return False

    print("DEBUG: Models loaded successfully.")
    return True

def extract_cnn_features_realtime(frame):
    """ Extrait les features CNN d'une frame. """
    global cnn_feature_extractor_model, preprocess_function, cnn_target_size

    if cnn_feature_extractor_model is None or preprocess_function is None or cnn_target_size is None:
        logging.error("Modèle CNN ou fonction de prétraitement non initialisé.")
        print("DEBUG: CNN feature extractor not initialized in extract_cnn_features_realtime")
        return None
    try:
        # Redimensionne à la taille attendue par le CNN (e.g., 224x224)
        img = cv2.resize(frame, cnn_target_size, interpolation=cv2.INTER_AREA)
        img_array = image.img_to_array(img)
        img_batch = np.expand_dims(img_array, axis=0)
        img_preprocessed = preprocess_function(img_batch)
        features = cnn_feature_extractor_model.predict(img_preprocessed, verbose=0)[0]
        return features
    except Exception as e:
        logging.exception(f"Erreur extraction CNN temps réel: {e}")
        print(f"DEBUG: Error during CNN feature extraction: {e}")
        return None

# --- Fonction Principale Temps Réel ---
def main_realtime():
    global vocabulaire, index_to_word
    print("DEBUG: main_realtime() started.")

    # --- Initialisation TF et GPU ---
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus: tf.config.experimental.set_memory_growth(gpu, True)
            logging.info(f"GPU(s) configuré(s): {gpus}")
            print(f"DEBUG: GPU(s) configured: {gpus}")
        except RuntimeError as e:
             logging.error(f"Erreur config GPU: {e}")
             print(f"DEBUG: GPU config error: {e}")
    else:
        logging.warning("Aucun GPU détecté par TensorFlow. L'inférence CNN sera lente.")
        print("DEBUG: No GPU detected by TensorFlow.")
    print(f"DEBUG: GPU check done.")

    # --- Chargement Modèles et Vocabulaire ---
    if not load_models_and_preprocessing():
        logging.error("Échec du chargement des modèles nécessaires. Arrêt.")
        print("DEBUG: Exiting main_realtime() due to model load failure.")
        return

    vocabulaire = load_vocabulary()
    if not vocabulaire:
        # Même si load_vocabulary n'arrête pas pour un vocab vide, on peut décider ici
        # que c'est une erreur critique pour la prédiction.
        logging.error("Erreur critique : Vocabulaire non chargé ou vide. Arrêt.")
        print("DEBUG: Exiting main_realtime() due to vocab load failure or empty vocab.")
        return
    index_to_word = {i: word for word, i in vocabulaire.items()}
    logging.info(f"Vocabulaire inversé créé ({len(index_to_word)} entrées).")
    print(f"DEBUG: Inverse vocabulary created ({len(index_to_word)} entries).")

    # --- Initialisation Webcam ---
    logging.info(f"Ouverture de la source de capture : {CAPTURE_SOURCE}")
    print(f"DEBUG: Attempting to open camera source: {CAPTURE_SOURCE} (type: {type(CAPTURE_SOURCE)})")
    cap = None # Initialiser à None
    try:
        cap = cv2.VideoCapture(CAPTURE_SOURCE)
        print("DEBUG: Waiting 1 sec for camera initialization...")
        time.sleep(1.0)
        print(f"DEBUG: cv2.VideoCapture returned: {cap}")
        is_opened = cap.isOpened() if cap else False
        print(f"DEBUG: Camera is opened: {is_opened}")
    except Exception as e_cap:
         logging.error(f"Exception lors de l'ouverture de la capture {CAPTURE_SOURCE}: {e_cap}", exc_info=True)
         print(f"DEBUG: Exception during cv2.VideoCapture(): {e_cap}")
         print(f"DEBUG: Exiting main_realtime() because camera failed to open (Exception)")
         if cap: cap.release() # Essayer de libérer même en cas d'erreur
         return

    if not is_opened:
        logging.error(f"Impossible d'ouvrir la source de capture {CAPTURE_SOURCE}. Vérifiez l'index de la caméra ou les permissions.")
        print(f"DEBUG: Exiting main_realtime() because camera failed to open (Source: {CAPTURE_SOURCE})")
        if cap: cap.release()
        return
    logging.info("Webcam ouverte avec succès.")
    print("DEBUG: Webcam opened successfully.")

    # --- Initialisation des variables pour la boucle temps réel ---
    sequence_window = deque(maxlen=FIXED_LENGTH)
    prediction_display_buffer = deque(maxlen=SMOOTHING_WINDOW_SIZE)
    last_top_n_text = ["Initialisation..."]
    frame_processing_times = deque(maxlen=30)
    frame_count = 0
    print("DEBUG: Real-time loop variables initialized.")

    # Calcul redimensionnement affichage si nécessaire
    target_width = None; target_height = None; resize_needed = False
    try:
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        logging.info(f"Résolution native webcam: {frame_width}x{frame_height}")
        print(f"DEBUG: Native camera resolution: {frame_width}x{frame_height}")
        if MAX_FRAME_WIDTH and frame_width > MAX_FRAME_WIDTH:
             scale = MAX_FRAME_WIDTH / frame_width
             target_width = MAX_FRAME_WIDTH
             target_height = int(frame_height * scale)
             resize_needed = True
             logging.info(f"Affichage sera redimensionné à -> {target_width}x{target_height}")
             print(f"DEBUG: Display will be resized to {target_width}x{target_height}")
        else:
             print("DEBUG: No display resizing needed.")
    except Exception as e:
        logging.warning(f"Impossible de lire la résolution de la webcam: {e}")
        print(f"DEBUG: Could not get camera resolution: {e}")

    print(f"\n{Colors.BRIGHT_YELLOW}Appuyez sur 'q' dans la fenêtre vidéo pour quitter.{Colors.RESET}")
    print("DEBUG: Entering main real-time loop...")
    loop_count = 0

    try:
        # --- Boucle Principale Temps Réel ---
        while True:
            loop_count += 1
            try:
                ret, frame = cap.read()
            except Exception as e_read:
                 logging.error(f"Exception during cap.read() (iteration {loop_count}): {e_read}", exc_info=True)
                 print(f"DEBUG: Breaking loop at iteration {loop_count} due to exception in cap.read(): {e_read}")
                 break

            # print(f"DEBUG: Frame read attempt {loop_count}: success={ret}") # Verbeux
            if not ret:
                logging.error(f"Impossible de lire la frame (iteration {loop_count}). Arrêt.")
                is_still_opened = cap.isOpened() if cap else False
                print(f"DEBUG: Breaking loop at iteration {loop_count}, cannot read frame. Camera still opened: {is_still_opened}")
                break
            frame_count += 1
            display_frame = frame.copy() # Copie pour l'affichage

            # Redimensionner pour affichage si nécessaire AVANT de traiter
            if resize_needed and target_width and target_height:
                try:
                    display_frame = cv2.resize(display_frame, (target_width, target_height), interpolation=cv2.INTER_AREA)
                except Exception as e_resize:
                    logging.warning(f"Erreur redimensionnement affichage: {e_resize}")
                    print(f"DEBUG: Error resizing display frame: {e_resize}")
                    # Continuer avec la frame non redimensionnée si erreur

            # --- Logique de Saut de Frames (pour CNN) ---
            if frame_count % (FRAMES_TO_SKIP + 1) == 0:
                # print(f"DEBUG: Processing frame {frame_count} for CNN.") # Verbeux
                start_time = time.time()

                # --- Extraction Features CNN (utilise la frame originale 'frame', pas display_frame redim.) ---
                cnn_features = extract_cnn_features_realtime(frame)

                end_time = time.time()
                proc_time = (end_time - start_time) * 1000 # en ms
                frame_processing_times.append(proc_time)

                if cnn_features is not None:
                    sequence_window.append(cnn_features)

                    # --- Logique de Prédiction LSTM ---
                    current_sequence_len = len(sequence_window)
                    if current_sequence_len > 0:
                        padded_sequence = None
                        if current_sequence_len < FIXED_LENGTH:
                            # print(f"DEBUG: Padding sequence. Current len: {current_sequence_len}") # Verbeux
                            padding = np.zeros((FIXED_LENGTH - current_sequence_len, FEATURE_DIM), dtype=np.float32)
                            padded_sequence = np.concatenate((padding, np.array(sequence_window, dtype=np.float32)), axis=0)
                        else:
                            # print(f"DEBUG: Sequence full. Length: {current_sequence_len}") # Verbeux
                            padded_sequence = np.array(sequence_window, dtype=np.float32)

                        if padded_sequence.shape == (FIXED_LENGTH, FEATURE_DIM):
                            reshaped_sequence = np.expand_dims(padded_sequence, axis=0)
                            # print("DEBUG: Predicting with LSTM...") # Verbeux
                            try:
                                res = lstm_prediction_model.predict(reshaped_sequence, verbose=0)[0]
                                top_n_indices = np.argsort(res)[-TOP_N:][::-1]
                                top_n_confidences = res[top_n_indices]
                                top_n_words = [index_to_word.get(idx, f"Idx_{idx}?") for idx in top_n_indices]
                                # print(f"DEBUG: Top prediction: {top_n_words[0]} ({top_n_confidences[0]:.2f})") # Verbeux

                                # Mettre à jour l'affichage seulement si confiance suffisante?
                                top_pred_idx = top_n_indices[0]
                                top_pred_conf = top_n_confidences[0]
                                if top_pred_conf >= PREDICTION_THRESHOLD:
                                    prediction_display_buffer.append(top_pred_idx) # Ajouter au buffer de lissage

                                # Préparer le texte pour affichage (Top N)
                                last_top_n_text = [f"{w} ({c:.2f})" for w, c in zip(top_n_words, top_n_confidences)]

                            except Exception as e_pred:
                                logging.exception(f"Erreur lstm_model.predict: {e_pred}")
                                print(f"DEBUG: Error during LSTM prediction: {e_pred}")
                                last_top_n_text = ["Erreur Prediction"]
                        else:
                            logging.warning(f"Shape seq incorrecte ({padded_sequence.shape}) avant prédic.")
                            print(f"DEBUG: Incorrect sequence shape before LSTM: {padded_sequence.shape}")
                            last_top_n_text = ["Erreur Seq Shape"]
                else:
                     # print("DEBUG: CNN Feature extraction returned None.") # Verbeux
                     last_top_n_text = ["Erreur Extrac. CNN"]
            # --- Fin du traitement CNN/LSTM (si frame non skippée) ---

            # --- Affichage (sur chaque frame lue) ---
            try:
                # Couleur basée sur la confiance de la MEILLEURE prédiction
                top_conf = 0.0
                if last_top_n_text and '(' in last_top_n_text[0] and 'Erreur' not in last_top_n_text[0]:
                     try:
                         top_conf = float(last_top_n_text[0].split('(')[1].split(')')[0])
                     except (IndexError, ValueError) as e_parse:
                         # print(f"DEBUG: Error parsing confidence from '{last_top_n_text[0]}': {e_parse}") # Verbeux
                         pass # Ignore parsing errors, keep top_conf = 0.0

                text_color = Colors.CV_RED
                if top_conf >= CONF_THRESH_GREEN: text_color = Colors.CV_GREEN
                elif top_conf >= CONF_THRESH_YELLOW: text_color = Colors.CV_YELLOW

                # Afficher les Top N prédictions
                y_offset = 30
                for line_idx, line in enumerate(last_top_n_text):
                    current_color = text_color if line_idx == 0 else Colors.CV_WHITE
                    cv2.putText(display_frame, line, (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.7, current_color, 2, cv2.LINE_AA)
                    y_offset += 25

                # Affichage mot lissé
                smoothed_word = "?"
                if prediction_display_buffer:
                    try:
                        smoothed_index_count = Counter(prediction_display_buffer).most_common(1)
                        if smoothed_index_count:
                            smoothed_index = smoothed_index_count[0][0]
                            smoothed_word = index_to_word.get(smoothed_index, "?")
                    except IndexError: pass # Peut arriver si buffer vide
                cv2.putText(display_frame, f"Pred. lissée: {smoothed_word}", (10, display_frame.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, Colors.CV_WHITE, 2, cv2.LINE_AA)

                # Afficher temps de traitement moyen frame CNN
                if frame_processing_times:
                    avg_proc_time = np.mean(frame_processing_times)
                    fps_approx = 1000 / avg_proc_time if avg_proc_time > 0 else 0
                    cv2.putText(display_frame, f"CNN Proc: {avg_proc_time:.1f} ms (~{fps_approx:.1f} FPS)", (10, display_frame.shape[0] - 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, Colors.CV_WHITE, 1, cv2.LINE_AA)

                # Afficher la frame résultante
                window_title = "Real-Time Sign Prediction (CNN+LSTM) - Press 'q' to quit"
                cv2.imshow(window_title, display_frame)

                # Gestion de la touche pour quitter
                key = cv2.waitKey(1) & 0xFF # Le waitKey est crucial pour que imshow fonctionne
                if key == ord('q'):
                    logging.info("Touche 'q' pressée, arrêt du programme.")
                    print("DEBUG: 'q' key pressed, breaking loop.")
                    break

            except Exception as e_display:
                logging.exception(f"Erreur lors de l'affichage OpenCV: {e_display}")
                print(f"DEBUG: Exception during OpenCV display (putText/imshow): {e_display}")
                # Si imshow échoue, la boucle risque de ne plus répondre à 'q'
                # On pourrait choisir de break ici.
                # break

            # Petite pause optionnelle pour ne pas surcharger le CPU à 100% inutilement
            # time.sleep(0.001) # 1ms

    except KeyboardInterrupt:
        logging.info("\nArrêt du programme demandé par l'utilisateur (Ctrl+C).")
        print("\nDEBUG: KeyboardInterrupt caught.")
    except Exception as e_main:
        logging.exception(f"Erreur majeure inattendue dans la boucle principale: {e_main}")
        print(f"DEBUG: Major unexpected exception in main loop: {e_main}")
    finally:
        # --- Nettoyage ---
        print(f"DEBUG: Exited main loop after {loop_count} iterations.")
        logging.info("Fermeture de la webcam et nettoyage...")
        print("DEBUG: Cleaning up...")
        if cap and cap.isOpened():
            try:
                cap.release()
                print("DEBUG: Camera released.")
            except Exception as e_release:
                 logging.error(f"Erreur lors de cap.release(): {e_release}")
                 print(f"DEBUG: Error releasing camera: {e_release}")
        else:
            print("DEBUG: Camera was not open or already released.")

        try:
            cv2.destroyAllWindows()
            # Ajouter un petit waitKey après destroyAllWindows peut aider sur certains systèmes
            cv2.waitKey(1)
            print("DEBUG: OpenCV windows destroyed.")
        except Exception as e_destroy:
             logging.error(f"Erreur lors de cv2.destroyAllWindows(): {e_destroy}")
             print(f"DEBUG: Error destroying OpenCV windows: {e_destroy}")

        try:
            print("DEBUG: Attempting to clear Keras session...")
            tf.keras.backend.clear_session()
            logging.info("Session Keras/TensorFlow libérée.") # Utiliser logging.info ici
            print("DEBUG: Keras session cleared.")
        except Exception as e_clear:
            logging.warning(f"Erreur lors de la libération de la session Keras: {e_clear}")
            print(f"DEBUG: Error clearing Keras session: {e_clear}")

        logging.info(f"{Colors.BRIGHT_GREEN}Programme de prédiction temps réel terminé.{Colors.RESET}")
        print("DEBUG: main_realtime() finished.")


if __name__ == "__main__":
    print("DEBUG: TraitementFluxVideo.py executed as main script.")
    main_realtime()