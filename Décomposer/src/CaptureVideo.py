# -*- coding: utf-8 -*-
import cv2
import mediapipe as mp
import numpy as np
import tensorflow as tf
import threading
import queue
import os
import logging
import time
import csv
from collections import deque, Counter
from tqdm import tqdm
import matplotlib.pyplot as plt # Pour le graphique

# --- Configuration du logging ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - [%(threadName)s] - %(message)s')

# --- ANSI escape codes for colors ---
class Colors:
    RESET = '\x1b[0m'
    BRIGHT_YELLOW = '\x1b[93m'
    BRIGHT_GREEN = '\x1b[92m'
    RED = '\x1b[31m'
    GREEN = '\x1b[32m' # Pour le démarrage du thread
    CV_GREEN = (0, 255, 0)
    CV_YELLOW = (0, 255, 255)
    CV_RED = (0, 0, 255)
    CV_WHITE = (255, 255, 255)

# --- Constants ---
# !!! IMPORTANT: Assurez-vous que MODEL_PATH pointe vers le modèle
#     entraîné avec les features incluant la bouche !!!
MODEL_PATH = "models/model_basic_mouth.h5"
VOCABULARY_FILE = "vocabulaire.txt"
FIXED_LENGTH = 46 # Doit correspondre à l'entraînement
VIDEOS_DIR = "D:/bonneaup.SNIRW/Test2/video" # CHEMIN ABSOLU (à adapter)

# --- CONFIGURATION ---
PREDICTION_CSV_FILE = "prediction_log_with_mouth.csv" # Nom de fichier CSV mis à jour
CSV_HEADER = ["Timestamp", "VideoFile", "MostFrequentWord", "Frequency", "TotalPredictions", "AvgConfidenceMostFrequent", "MaxConfidenceSeen", "ProcessingTimeSec"]
SAVE_KEYPOINTS = True # Mettre à False si vous ne voulez pas sauvegarder les keypoints pendant la capture
KEYPOINTS_SAVE_DIR = "extracted_keypoints_capture" # Dossier différent pour la capture vs traitement?
TOP_N = 3
SMOOTHING_WINDOW_SIZE = 15
CONF_THRESH_GREEN = 0.80
CONF_THRESH_YELLOW = 0.50
FRAMES_TO_SKIP = 3
MAX_FRAME_WIDTH = 1280
DEADLOCK_TIMEOUT = 10

# --- Paramètres d'Extraction (DOIVENT CORRESPONDRE à traitementVideo.py et entrainement.py) ---
NUM_POSE_KEYPOINTS = 4
NUM_HAND_KEYPOINTS = 21
NUM_COORDS = 3 # x, y, z

# Indices des points des lèvres (DOIVENT ÊTRE IDENTIQUES à ceux utilisés pour l'entraînement)
MOUTH_LANDMARK_INDICES = sorted(list(set([
    61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291,
    78, 95, 88, 178, 87, 14, 317, 402, 318, 324, 308,
    185, 40, 39, 37, 0, 267, 269, 270, 409,
    191, 80, 81, 82, 13, 312, 311, 310, 415
])))
NUM_MOUTH_KEYPOINTS = len(MOUTH_LANDMARK_INDICES)

# ---> MISE À JOUR FEATURES_PER_FRAME (DOIT CORRESPONDRE à l'entraînement) <---
FEATURES_PER_FRAME = (NUM_POSE_KEYPOINTS * NUM_COORDS) + \
                     (NUM_HAND_KEYPOINTS * NUM_COORDS) * 2 + \
                     (NUM_MOUTH_KEYPOINTS * NUM_COORDS) # <-- AJOUT

# --- Logging des configurations ---
logging.info("--- Configuration CaptureVideo (avec Bouche) ---")
logging.info(f"MODEL_PATH: {MODEL_PATH} (Doit être entraîné avec bouche!)")
logging.info(f"VOCABULARY_FILE: {VOCABULARY_FILE}")
logging.info(f"FIXED_LENGTH: {FIXED_LENGTH}")
logging.info(f"VIDEOS_DIR: {VIDEOS_DIR}")
logging.info(f"PREDICTION_CSV_FILE: {PREDICTION_CSV_FILE}")
logging.info(f"SAVE_KEYPOINTS: {SAVE_KEYPOINTS}")
if SAVE_KEYPOINTS: logging.info(f"KEYPOINTS_SAVE_DIR: {KEYPOINTS_SAVE_DIR}")
logging.info(f"TOP_N Predictions: {TOP_N}")
logging.info(f"SMOOTHING_WINDOW_SIZE: {SMOOTHING_WINDOW_SIZE}")
logging.info(f"FRAMES_TO_SKIP: {FRAMES_TO_SKIP}")
if MAX_FRAME_WIDTH: logging.info(f"MAX_FRAME_WIDTH: {MAX_FRAME_WIDTH}")
else: logging.info("MAX_FRAME_WIDTH: Désactivé")
logging.info(f"DEADLOCK_TIMEOUT: {DEADLOCK_TIMEOUT}s")
logging.info(f"Nombre points Pose: {NUM_POSE_KEYPOINTS}, Main: {NUM_HAND_KEYPOINTS}, Bouche: {NUM_MOUTH_KEYPOINTS}")
logging.info(f"FEATURES_PER_FRAME attendu: {FEATURES_PER_FRAME}") # <-- Affichage mis à jour
logging.info("-----------------------------------------------")


# --- Utility Functions ---
def load_vocabulary(filepath):
    vocabulaire = {}
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split(":")
                if len(parts) == 2 and parts[0] and parts[1].isdigit():
                    vocabulaire[parts[0]] = int(parts[1])
                elif line.strip():
                    logging.warning(f"Format ligne incorrect vocabulaire: '{line.strip()}'")
    except (FileNotFoundError, ValueError) as e:
        logging.error(f"Erreur chargement vocabulaire {filepath}: {e}")
        return {}
    logging.info(f"Vocabulaire chargé ({len(vocabulaire)} mots) depuis {filepath}")
    return vocabulaire

def extract_keypoints(results):
    """
    Extracts POSE (4) + LEFT HAND (21) + RIGHT HAND (21) + MOUTH (NUM_MOUTH_KEYPOINTS) keypoints.
    Retourne un array numpy de taille FEATURES_PER_FRAME ou un array de zéros si erreur.
    """
    # Pose (4 points)
    pose = np.array([[res.x, res.y, res.z] for res in results.pose_landmarks.landmark[:NUM_POSE_KEYPOINTS]]).flatten() \
        if results.pose_landmarks and len(results.pose_landmarks.landmark) >= NUM_POSE_KEYPOINTS else np.zeros(NUM_POSE_KEYPOINTS * NUM_COORDS)

    # Mains (21 points chacune)
    lh = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]).flatten() \
        if results.left_hand_landmarks and len(results.left_hand_landmarks.landmark) == NUM_HAND_KEYPOINTS else np.zeros(NUM_HAND_KEYPOINTS * NUM_COORDS)
    rh = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]).flatten() \
        if results.right_hand_landmarks and len(results.right_hand_landmarks.landmark) == NUM_HAND_KEYPOINTS else np.zeros(NUM_HAND_KEYPOINTS * NUM_COORDS)

    # Bouche (NUM_MOUTH_KEYPOINTS points)
    mouth = np.zeros(NUM_MOUTH_KEYPOINTS * NUM_COORDS) # Initialise avec des zéros
    if results.face_landmarks:
        try:
            # Vérifier si tous les indices demandés existent dans les landmarks détectés
            if all(idx < len(results.face_landmarks.landmark) for idx in MOUTH_LANDMARK_INDICES):
                 mouth_points = [results.face_landmarks.landmark[i] for i in MOUTH_LANDMARK_INDICES]
                 mouth = np.array([[res.x, res.y, res.z] for res in mouth_points]).flatten()
            else:
                 logging.warning(f"Indices de bouche manquants dans face_landmarks (détectés: {len(results.face_landmarks.landmark)}). Retour zéros pour la bouche.")
        except IndexError as ie:
            logging.error(f"Erreur d'indice lors de l'extraction des points de bouche: {ie}. Retour zéros pour la bouche.")
            mouth = np.zeros(NUM_MOUTH_KEYPOINTS * NUM_COORDS)
        except Exception as e:
            logging.error(f"Erreur inattendue lors de l'extraction des points de bouche: {e}. Retour zéros pour la bouche.")
            mouth = np.zeros(NUM_MOUTH_KEYPOINTS * NUM_COORDS)

    # Concaténation de toutes les features
    extracted = np.concatenate([pose, lh, rh, mouth]) # <-- 'mouth' est ajouté ici

    # Vérification finale de la taille totale des features
    if extracted.shape[0] != FEATURES_PER_FRAME:
        logging.warning(f"L'extraction a produit un nombre inattendu de features ({extracted.shape[0]}), attendu {FEATURES_PER_FRAME}. Retour de zéros.")
        return np.zeros(FEATURES_PER_FRAME) # Retourner des zéros de la bonne taille

    return extracted

def get_expected_word_from_filename(filename):
    # ... (fonction inchangée) ...
    name_without_ext = os.path.splitext(filename)[0]
    parts = name_without_ext.split('_', 1)
    expected_word = parts[0]
    return expected_word.strip().lower()


# --- Keypoint Extractor Class ---
class KeypointExtractor:
    def __init__(self):
        self.mp_holistic = mp.solutions.holistic
        self.frame_queue = queue.Queue(maxsize=50)
        self.keypoint_queue = queue.Queue(maxsize=100)
        self.display_queue = queue.Queue(maxsize=10)
        self.running = threading.Event()
        self.video_capture = None
        self.capture_thread = None
        self.extraction_thread = None
        self.video_path = None

    def capture_frames(self):
        """Thread function to capture frames from video."""
        # Lit les frames, redimensionne si nécessaire, saute des frames,
        # et met les frames dans frame_queue et display_queue.
        # Aucune modification directe nécessaire ici car elle ne fait pas l'extraction.
        frame_count_read = 0
        frame_count_queued_extract = 0
        frame_count_queued_display = 0
        capture_start_time = time.time()
        self.video_capture = None

        try:
            logging.info(f"{Colors.GREEN}>>> DÉMARRAGE THREAD CAPTURE pour {os.path.basename(self.video_path)}{Colors.RESET}")
            # ... (ouverture vidéo, calcul redimensionnement etc. inchangé) ...
            self.video_capture = cv2.VideoCapture(self.video_path)
            if not self.video_capture.isOpened():
                logging.error(f"Impossible d'ouvrir vidéo : {self.video_path}")
                raise ValueError(f"Impossible d'ouvrir vidéo : {self.video_path}")

            frame_width = int(self.video_capture.get(cv2.CAP_PROP_FRAME_WIDTH))
            frame_height = int(self.video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = self.video_capture.get(cv2.CAP_PROP_FPS)
            total_frames = int(self.video_capture.get(cv2.CAP_PROP_FRAME_COUNT))
            logging.info(f"Vidéo ouverte: {os.path.basename(self.video_path)} ({frame_width}x{frame_height} @ {fps:.2f} FPS, Total Frames: {total_frames})")

            target_width = frame_width; target_height = frame_height; resize_needed = False
            if MAX_FRAME_WIDTH and frame_width > MAX_FRAME_WIDTH:
                scale = MAX_FRAME_WIDTH / frame_width
                target_width = MAX_FRAME_WIDTH
                target_height = int(frame_height * scale)
                resize_needed = True
                logging.info(f"Redimensionnement activé: {frame_width}x{frame_height} -> {target_width}x{target_height}")

            while self.running.is_set():
                # ... (lecture frame inchangée) ...
                ret, frame = self.video_capture.read()
                if not ret:
                    logging.info(f"Fin vidéo ou erreur lecture après {frame_count_read} frames lues: {os.path.basename(self.video_path)}.")
                    break
                frame_count_read += 1

                # Skip frames logic (inchangée)
                if (frame_count_read - 1) % FRAMES_TO_SKIP == 0:
                    frame_to_process = frame
                    if resize_needed:
                         # ... (redimensionnement inchangé) ...
                         try: frame_to_process = cv2.resize(frame, (target_width, target_height), interpolation=cv2.INTER_AREA)
                         except Exception as resize_err: logging.warning(f"[Capture] Erreur redim. frame {frame_count_read}: {resize_err}"); frame_to_process = frame

                    if frame_to_process is not None:
                         # Mettre dans les queues (inchangé)
                         try: self.frame_queue.put(frame_to_process, timeout=2.0); frame_count_queued_extract += 1
                         except queue.Full: logging.warning(f"[Capture] Queue frames (extract) pleine. Attente..."); self.frame_queue.put(frame_to_process); frame_count_queued_extract += 1 # Blocage
                         except Exception as e: logging.exception(f"[Capture] Erreur frame_queue.put : {e}"); self.running.clear(); break

                         try: self.display_queue.put_nowait(frame_to_process); frame_count_queued_display += 1
                         except queue.Full:
                             try: self.display_queue.get_nowait(); self.display_queue.put_nowait(frame_to_process)
                             except queue.Empty: pass
                             except Exception as e: logging.warning(f"[Capture] Erreur remplacement display queue: {e}")
                         except Exception as e: logging.warning(f"[Capture] Erreur display_queue.put : {e}")
                    else:
                         logging.warning(f"[Capture] frame_to_process est None (frame #{frame_count_read}), skip.")

            logging.debug(f"[Capture] Sortie de la boucle while pour {os.path.basename(self.video_path)}.")
        # ... (Gestion erreurs et finally inchangée) ...
        except ValueError: pass # Already logged
        except Exception as e_globale_capture:
            logging.exception(f"{Colors.RED}!!! ERREUR GLOBALE CAPTURÉE dans capture_frames pour {os.path.basename(self.video_path)} !!! : {e_globale_capture}{Colors.RESET}")
            self.running.clear()
        finally:
            logging.debug(f"{Colors.RED}>>> Entrée FINALLY capture_frames pour {os.path.basename(self.video_path)}{Colors.RESET}")
            if self.video_capture:
                 if self.video_capture.isOpened(): self.video_capture.release(); logging.info(f"Capture vidéo relâchée pour {os.path.basename(self.video_path)}")
            try: self.frame_queue.put(None, timeout=5.0); logging.info("[Capture - Finally] Signal fin (None) envoyé à frame_queue.")
            except queue.Full: logging.error("[Capture - Finally] Échec envoi signal fin (None) - Queue frames pleine.")
            except Exception as e: logging.error(f"[Capture - Finally] Erreur envoi signal fin (None) : {e}")
            total_time = time.time() - capture_start_time
            logging.info(f"Thread capture terminé pour {os.path.basename(self.video_path)}. {frame_count_queued_extract}/{frame_count_read} frames vers extraction, {frame_count_queued_display} vers affichage en {total_time:.2f}s.")
            logging.info(f"{Colors.RED}### FIN THREAD CAPTURE pour {os.path.basename(self.video_path)} ###{Colors.RESET}")


    def extract_keypoints_loop(self):
        """Thread function to extract keypoints using MediaPipe."""
        # ... (Initialisation inchangée) ...
        logging.info(f"{Colors.GREEN}>>> DÉMARRAGE THREAD EXTRACTION pour {os.path.basename(self.video_path)}{Colors.RESET}")
        frames_processed = 0
        extraction_start_time = time.time()
        holistic_instance = None
        try:
            # Initialiser Holistic DANS le thread
            holistic_instance = self.mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5)
            logging.info("[Extraction] Instance MediaPipe Holistic créée.")

            while True:
                # ... (Vérification running flag et get frame inchangé) ...
                if not self.running.is_set():
                    # ... (Vidage queue si arrêt demandé) ...
                    logging.info("[Extraction] Arrêt demandé (running=False).")
                    break

                frame = None
                try:
                    frame = self.frame_queue.get(timeout=1.0)
                    if frame is None:
                        logging.info(f"[Extraction] Signal fin (None) reçu de frame_queue. Fin normale.")
                        break
                except queue.Empty:
                    if self.capture_thread and not self.capture_thread.is_alive():
                        logging.warning("[Extraction] Queue frames vide et thread capture mort. Arrêt extraction.")
                        break
                    else: continue

                # Traitement avec MediaPipe (inchangé)
                try:
                    process_start = time.time()
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    frame_rgb.flags.writeable = False
                    results = holistic_instance.process(frame_rgb)
                    process_time = time.time() - process_start

                    # ---> APPEL À LA FONCTION MISE À JOUR <---
                    # Utilise la fonction extract_keypoints définie plus haut, qui inclut la bouche
                    keypoints = extract_keypoints(results)
                    frames_processed += 1

                    # Mettre les keypoints (avec bouche) dans la queue (inchangé)
                    try:
                        self.keypoint_queue.put(keypoints, timeout=2.0)
                    except queue.Full:
                        logging.warning(f"[Extraction] Queue keypoints pleine. Attente...")
                        self.keypoint_queue.put(keypoints) # Blocage
                    except Exception as e_put_kp:
                        logging.exception(f"[Extraction] Erreur keypoint_queue.put : {e_put_kp}")
                        self.running.clear(); break

                except Exception as e_process:
                    logging.exception(f"[Extraction] Erreur traitement MediaPipe/extraction frame {frames_processed + 1}: {e_process}")
                    # On continue potentiellement au prochain frame
                    pass

            logging.debug(f"[Extraction] Sortie de la boucle while pour {os.path.basename(self.video_path)}.")
        # ... (Gestion erreurs et finally inchangée) ...
        except Exception as e_init_loop:
             logging.exception(f"[Extraction] Erreur majeure initialisation ou boucle extraction: {e_init_loop}")
             self.running.clear()
        finally:
             logging.debug(f"{Colors.RED}>>> Entrée FINALLY extract_keypoints_loop pour {os.path.basename(self.video_path)}{Colors.RESET}")
             if holistic_instance:
                 try: holistic_instance.close(); logging.debug("[Extraction - Finally] Instance Holistic fermée.")
                 except Exception as e: logging.warning(f"[Extraction - Finally] Erreur fermeture Holistic: {e}")
             try: self.keypoint_queue.put(None, timeout=5.0); logging.info("[Extraction - Finally] Signal fin (None) envoyé à keypoint_queue.")
             except queue.Full: logging.error("[Extraction - Finally] Échec envoi signal fin (None) keypoints - Queue pleine.")
             except Exception as e: logging.error(f"[Extraction - Finally] Erreur envoi signal fin (None) keypoints : {e}")
             total_time = time.time() - extraction_start_time
             logging.info(f"Fin boucle extraction pour {os.path.basename(self.video_path)}. Traité {frames_processed} frames en {total_time:.2f}s.")
             logging.info(f"{Colors.RED}### FIN THREAD EXTRACTION pour {os.path.basename(self.video_path)} ###{Colors.RESET}")


    def start(self, video_path):
        # ... (fonction inchangée) ...
        if self.capture_thread is not None and self.capture_thread.is_alive() or \
           self.extraction_thread is not None and self.extraction_thread.is_alive():
            logging.warning(f"{Colors.BRIGHT_YELLOW}Tentative de démarrer alors que threads actifs pour {os.path.basename(self.video_path)}. Appel stop() d'abord...{Colors.RESET}")
            self.stop()

        self.video_path = video_path
        self.running.set()

        logging.debug("Vidage queues avant démarrage...")
        queues_to_clear = [self.frame_queue, self.keypoint_queue, self.display_queue]
        for q in queues_to_clear:
            while not q.empty():
                try: q.get_nowait()
                except queue.Empty: break
                except Exception as e_clear: logging.warning(f"Erreur vidage queue {q}: {e_clear}")
        logging.debug("Queues vidées.")

        self.capture_thread = threading.Thread(target=self.capture_frames, name=f"Capture-{os.path.basename(video_path)}")
        self.extraction_thread = threading.Thread(target=self.extract_keypoints_loop, name=f"Extract-{os.path.basename(video_path)}")
        self.extraction_thread.start()
        self.capture_thread.start()
        logging.info(f"Threads démarrés pour {os.path.basename(video_path)}")

    def stop(self):
        # ... (fonction inchangée) ...
        video_name = os.path.basename(self.video_path) if self.video_path else "Unknown Video"
        logging.info(f"{Colors.BRIGHT_YELLOW}>>> Entrée dans stop() pour {video_name}{Colors.RESET}")
        self.running.clear()
        logging.info(f"Flag 'running' mis à False pour {video_name}.")

        join_timeout_capture = 10; join_timeout_extract = 20

        if self.capture_thread is not None:
            thread_name = self.capture_thread.name
            if self.capture_thread.is_alive():
                logging.info(f"Attente fin thread capture '{thread_name}' (max {join_timeout_capture}s)...")
                self.capture_thread.join(timeout=join_timeout_capture)
                if self.capture_thread.is_alive(): logging.warning(f"{Colors.RED}TIMEOUT!{Colors.RESET} Thread capture '{thread_name}' non terminé.")
                else: logging.info(f"Thread capture '{thread_name}' terminé.")
            else: logging.debug(f"Thread capture '{thread_name}' déjà terminé.")
        else: logging.debug("stop(): Thread capture non trouvé (None).")

        if self.extraction_thread is not None:
            thread_name = self.extraction_thread.name
            if self.extraction_thread.is_alive():
                logging.info(f"Attente fin thread extraction '{thread_name}' (max {join_timeout_extract}s)...")
                self.extraction_thread.join(timeout=join_timeout_extract)
                if self.extraction_thread.is_alive(): logging.warning(f"{Colors.RED}TIMEOUT!{Colors.RESET} Thread extraction '{thread_name}' non terminé.")
                else: logging.info(f"Thread extraction '{thread_name}' terminé.")
            else: logging.debug(f"Thread extraction '{thread_name}' déjà terminé.")
        else: logging.debug("stop(): Thread extraction non trouvé (None).")

        self.capture_thread = None; self.extraction_thread = None
        logging.info(f"Vérification arrêt threads terminée pour {video_name}.")
        logging.info(f"{Colors.BRIGHT_YELLOW}<<< Sortie de stop() pour {video_name}{Colors.RESET}")


# --- Main Function ---
def main():
    global SAVE_KEYPOINTS # Déclarer l'intention de potentiellement modifier la globale

    # --- Initialisation TF et Modèle ---
    model = None
    try:
        # ... (Configuration GPU inchangée) ...
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            try:
                for gpu in gpus: tf.config.experimental.set_memory_growth(gpu, True)
                logging.info(f"GPU(s) configuré(s) pour Memory Growth : {gpus}")
            except RuntimeError as e: logging.error(f"Erreur configuration Memory Growth GPU: {e}")
        else: logging.warning("Aucun GPU détecté par TensorFlow.")

        # Charger le modèle Keras (DOIT ÊTRE ENTRAÎNÉ AVEC LA BOUCHE)
        if not os.path.exists(MODEL_PATH):
             logging.error(f"Fichier modèle non trouvé : {MODEL_PATH}"); return
        model = tf.keras.models.load_model(MODEL_PATH)
        logging.info(f"Modèle chargé depuis {MODEL_PATH}")
        try:
            # Vérifier la forme d'entrée attendue par le modèle chargé
            expected_shape = model.input_shape
            logging.info(f"Forme entrée attendue par le modèle chargé: {expected_shape}")
            # Valider que la forme du modèle correspond aux constantes du script
            # expected_shape est typiquement (None, FIXED_LENGTH, FEATURES_PER_FRAME)
            if len(expected_shape) != 3 or \
               expected_shape[1] is not None and expected_shape[1] != FIXED_LENGTH or \
               expected_shape[2] is not None and expected_shape[2] != FEATURES_PER_FRAME:
                 logging.warning(
                     f"{Colors.RED}Incohérence Shape! Modèle attend (batch, {expected_shape[1]}, {expected_shape[2]}), "
                     f"Script Config (batch, {FIXED_LENGTH}, {FEATURES_PER_FRAME}){Colors.RESET}"
                 )
                 # Optionnel: Sortir si l'incohérence est critique
                 # return
            else:
                 logging.info("La forme d'entrée du modèle chargé correspond aux constantes du script.")
        except Exception as e:
             logging.warning(f"Impossible de vérifier/valider la forme d'entrée du modèle: {e}")
    except Exception as e:
        logging.exception(f"Erreur majeure lors de l'initialisation TensorFlow/Modèle : {e}"); return

    # --- Chargement Vocabulaire ---
    vocabulaire = load_vocabulary(VOCABULARY_FILE)
    if not vocabulaire: logging.error("Erreur critique : Vocabulaire vide/non chargé. Arrêt."); return
    index_to_word = {i: word for word, i in vocabulaire.items()}

    # --- Vérification Dossier Vidéos et Listing ---
    # ... (inchangé) ...
    if not os.path.isdir(VIDEOS_DIR): logging.error(f"Chemin vidéo invalide: {VIDEOS_DIR}"); return
    try:
        video_files_to_process = [f for f in os.listdir(VIDEOS_DIR) if f.lower().endswith(('.mp4', '.avi', '.mov', '.mkv', '.webm'))]
        video_files_to_process.sort()
        logging.debug(f"Fichiers vidéo trouvés: {video_files_to_process}")
    except Exception as e: logging.exception(f"Erreur listage fichiers dans {VIDEOS_DIR}: {e}"); return
    if not video_files_to_process: logging.info(f"Aucune vidéo trouvée dans {VIDEOS_DIR}."); return
    logging.info(f"Trouvé {len(video_files_to_process)} vidéos à traiter.")

    # --- Préparation Sauvegarde Keypoints et CSV ---
    if SAVE_KEYPOINTS:
        try:
            os.makedirs(KEYPOINTS_SAVE_DIR, exist_ok=True)
            logging.info(f"Sauvegarde keypoints activée -> Dossier: '{KEYPOINTS_SAVE_DIR}'")
        except OSError as e:
            logging.error(f"Impossible créer dossier keypoints '{KEYPOINTS_SAVE_DIR}': {e}")
            SAVE_KEYPOINTS = False # Modifie la globale si erreur
            logging.warning("Sauvegarde keypoints désactivée.")

    try:
        # ... (Préparation CSV inchangée, utilise PREDICTION_CSV_FILE mis à jour) ...
        file_exists = os.path.isfile(PREDICTION_CSV_FILE)
        write_header = not file_exists or os.path.getsize(PREDICTION_CSV_FILE) == 0
        with open(PREDICTION_CSV_FILE, 'a', newline='', encoding='utf-8') as csvfile:
            writer = csv.writer(csvfile)
            if write_header: writer.writerow(CSV_HEADER); logging.info(f"CSV '{PREDICTION_CSV_FILE}' prêt (en-tête écrit).")
            else: logging.info(f"CSV '{PREDICTION_CSV_FILE}' prêt (ajout données).")
    except IOError as e:
        logging.error(f"Impossible ouvrir/écrire en-tête CSV dans {PREDICTION_CSV_FILE}: {e}")
        # return # Optionnel: sortir si le CSV est critique

    main_start_time = time.time()
    extractor = None
    word_counts = {}; correct_predictions = {} # Pour suivi précision

    try:
        # --- Boucle Principale sur les Vidéos ---
        for video_index, video_file in enumerate(tqdm(video_files_to_process, desc="Traitement Vidéos", unit="video")):
            video_path = os.path.join(VIDEOS_DIR, video_file)
            base_video_name = os.path.basename(video_path)
            window_name = f"Video - {base_video_name}"
            logging.info(f"{Colors.BRIGHT_YELLOW}--- [{video_index+1}/{len(video_files_to_process)}] Début: {base_video_name} ---{Colors.RESET}")
            video_start_time = time.time()

            extractor = KeypointExtractor()
            extractor.start(video_path)

            # Structures de données pour la vidéo courante
            sequence_window = deque(maxlen=FIXED_LENGTH)
            all_keypoints_for_video = [] # Si SAVE_KEYPOINTS
            all_predictions_details = []
            prediction_display_buffer = deque(maxlen=SMOOTHING_WINDOW_SIZE)
            frame_display_buffer = None
            processing_active = True
            frames_processed_main = 0; predictions_made = 0
            max_confidence_seen_video = 0.0
            last_top_n_text = ["Initialisation..."]
            last_keypoint_time = time.time(); deadlock_timeout_occurred = False

            try:
                # --- Boucle Interne: Traitement Keypoints & Affichage ---
                while processing_active:
                    keypoints = None
                    try:
                        # Récupérer les keypoints (qui incluent maintenant la bouche)
                        keypoints = extractor.keypoint_queue.get(timeout=0.5)
                        if keypoints is None:
                            logging.info(f"[Main] Signal fin (None) reçu keypoint_queue. Fin {base_video_name}.")
                            processing_active = False
                        else:
                            frames_processed_main += 1
                            last_keypoint_time = time.time()
                            if SAVE_KEYPOINTS:
                                all_keypoints_for_video.append(keypoints)
                            # Ajouter le vecteur de keypoints (plus long) à la fenêtre
                            sequence_window.append(keypoints)

                    except queue.Empty:
                        # ... (Logique détection deadlock/fin inchangée) ...
                        capture_alive = extractor.capture_thread and extractor.capture_thread.is_alive()
                        extract_alive = extractor.extraction_thread and extractor.extraction_thread.is_alive()
                        if not extractor.running.is_set() and extractor.keypoint_queue.empty() and not capture_alive and not extract_alive:
                            logging.info(f"[Main] Arrêt/threads terminés et queue vide. Fin {base_video_name}.")
                            processing_active = False
                        elif not capture_alive and extract_alive:
                            time_since_last = time.time() - last_keypoint_time
                            if time_since_last > DEADLOCK_TIMEOUT:
                                logging.error(f"{Colors.RED}[Main] DEADLOCK TIMEOUT ({DEADLOCK_TIMEOUT}s) détecté pour {base_video_name}! Forçage arrêt.{Colors.RESET}")
                                deadlock_timeout_occurred = True
                                processing_active = False
                            # else: logging.warning(f"[Main] Deadlock potentiel? Inactivité: {time_since_last:.1f}s")
                        elif extractor.running.is_set() or capture_alive or extract_alive:
                             pass # Attente
                        else: # Ni running, ni threads vivants
                             if not extractor.keypoint_queue.empty(): logging.info("[Main] Threads morts mais queue pas vide? Tentative vidage...")
                             else: logging.info("[Main] Threads morts et queue vide. Arrêt."); processing_active = False

                    # --- Logique de Prédiction (utilise les keypoints avec bouche) ---
                    if keypoints is not None and not deadlock_timeout_occurred:
                        current_sequence_len = len(sequence_window)
                        padded_sequence = None

                        if current_sequence_len > 0:
                            if current_sequence_len < FIXED_LENGTH:
                                # ---> Padding utilise le nouveau FEATURES_PER_FRAME <---
                                padding = np.zeros((FIXED_LENGTH - current_sequence_len, FEATURES_PER_FRAME))
                                padded_sequence = np.concatenate((padding, np.array(sequence_window)), axis=0)
                            else:
                                padded_sequence = np.array(sequence_window)

                            # ---> Vérification shape utilise le nouveau FEATURES_PER_FRAME <---
                            if padded_sequence is not None and padded_sequence.shape == (FIXED_LENGTH, FEATURES_PER_FRAME):
                                reshaped_sequence = np.expand_dims(padded_sequence, axis=0)
                                try:
                                    predict_start = time.time()
                                    # Le modèle chargé DOIT avoir été entraîné avec cette shape d'entrée
                                    res = model.predict(reshaped_sequence, verbose=0)[0]
                                    predict_time = time.time() - predict_start
                                    predictions_made += 1

                                    # ... (Analyse top N, buffer lissage inchangés) ...
                                    top_n_indices = np.argsort(res)[-TOP_N:][::-1]
                                    top_n_confidences = res[top_n_indices]
                                    top_n_words = [index_to_word.get(idx, f"Idx_{idx}?") for idx in top_n_indices]
                                    top_pred_idx = top_n_indices[0]; top_pred_conf = top_n_confidences[0]
                                    all_predictions_details.append((top_pred_idx, top_pred_conf))
                                    prediction_display_buffer.append(top_pred_idx)
                                    max_confidence_seen_video = max(max_confidence_seen_video, top_pred_conf)
                                    last_top_n_text = [f"{w} ({c:.2f})" for w, c in zip(top_n_words, top_n_confidences)]

                                except tf.errors.InvalidArgumentError as e_tf_shape:
                                     # Cette erreur peut survenir si le modèle chargé n'a pas la bonne shape d'entrée !
                                     logging.error(f"[Main] {Colors.RED}Erreur TensorFlow (mauvaise shape modèle?): {e_tf_shape}. Shape fournie: {reshaped_sequence.shape}. Modèle attendait probablement autre chose.{Colors.RESET}")
                                     last_top_n_text = ["Erreur Shape Modèle?"]
                                     processing_active = False # Arrêter si le modèle est incompatible
                                except Exception as e_pred:
                                    logging.exception(f"[Main] Erreur inconnue model.predict: {e_pred}")
                                    last_top_n_text = ["Erreur Prediction"]
                            else:
                                # Cette erreur indique un problème dans la logique de padding/fenêtrage du script
                                logging.warning(f"[Main] Shape incorrecte ({padded_sequence.shape if padded_sequence is not None else 'None'}) avant prédiction. Attendu: ({FIXED_LENGTH}, {FEATURES_PER_FRAME})")
                                last_top_n_text = ["Erreur Seq Shape"]

                    # --- Affichage (inchangé) ---
                    try:
                        new_frame = extractor.display_queue.get_nowait()
                        if new_frame is not None: frame_display_buffer = new_frame
                    except queue.Empty: pass

                    if frame_display_buffer is not None and frame_display_buffer.size > 0:
                        display_frame = frame_display_buffer.copy()
                        try:
                            top_conf = all_predictions_details[-1][1] if all_predictions_details else 0.0
                            text_color = Colors.CV_RED
                            if top_conf >= CONF_THRESH_GREEN: text_color = Colors.CV_GREEN
                            elif top_conf >= CONF_THRESH_YELLOW: text_color = Colors.CV_YELLOW

                            y_offset = 30
                            for line_idx, line in enumerate(last_top_n_text):
                                current_color = text_color if line_idx == 0 else Colors.CV_WHITE
                                cv2.putText(display_frame, line, (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.7, current_color, 2, cv2.LINE_AA)
                                y_offset += 25

                            if prediction_display_buffer:
                                try:
                                    smoothed_index = Counter(prediction_display_buffer).most_common(1)[0][0]
                                    smoothed_word = index_to_word.get(smoothed_index, "?")
                                    cv2.putText(display_frame, f"Lisse ({SMOOTHING_WINDOW_SIZE}f): {smoothed_word}", (10, display_frame.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, Colors.CV_WHITE, 2, cv2.LINE_AA)
                                except IndexError: pass

                            cv2.imshow(window_name, display_frame)
                            key = cv2.waitKey(1) & 0xFF
                            if key == ord('q'):
                                logging.info("Touche 'q' pressée, arrêt global.")
                                processing_active = False
                                extractor.running.clear()
                                raise KeyboardInterrupt("Arrêt utilisateur via 'q'")

                        except cv2.error as e_cv:
                            if "NULL window" in str(e_cv) or "Invalid window handle" in str(e_cv):
                                logging.warning(f"[Main] Fenêtre '{window_name}' fermée? Arrêt vidéo.")
                                processing_active = False
                            else: logging.warning(f"[Main] Erreur cv2: {e_cv}")
                        except Exception as e_show:
                             logging.exception(f"[Main] Erreur affichage/texte: {e_show}")

                    if not processing_active:
                         logging.debug(f"[Main] processing_active=False, sortie boucle vidéo {base_video_name}.")
                         break
                # --- Fin boucle interne ---
                logging.info(f"Fin boucle traitement principale pour {base_video_name}.")

            except KeyboardInterrupt:
                 logging.info(f"KeyboardInterrupt pendant {base_video_name}. Arrêt...")
                 if extractor: extractor.running.clear()
                 raise
            except Exception as e_inner_loop:
                logging.exception(f"Erreur inattendue boucle interne {base_video_name}: {e_inner_loop}")
                if extractor: extractor.running.clear()
            finally:
                # --- Nettoyage après chaque vidéo ---
                logging.info(f"{Colors.BRIGHT_YELLOW}--- Nettoyage pour {base_video_name}... ---{Colors.RESET}")
                if extractor: extractor.stop()
                try:
                     if cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) >= 1:
                         cv2.destroyWindow(window_name); logging.info(f"Fenêtre '{window_name}' fermée.")
                     # else: logging.debug(f"Fenêtre '{window_name}' déjà fermée.")
                     cv2.waitKey(1)
                except Exception as e_close:
                    logging.warning(f"Erreur fermeture fenêtre '{window_name}': {e_close}"); cv2.waitKey(1)

                video_end_time = time.time(); processing_time_sec = video_end_time - video_start_time
                logging.info(f"Vidéo {base_video_name}: {frames_processed_main} keypoints traités, {predictions_made} prédictions.")
                logging.info(f"Temps traitement vidéo: {processing_time_sec:.2f} sec.")

                # --- Sauvegarde Keypoints (.npy) si activée ---
                if SAVE_KEYPOINTS and all_keypoints_for_video:
                    npy_filename = os.path.splitext(base_video_name)[0] + "_capture.npy" # Suffixe pour distinguer
                    npy_filepath = os.path.join(KEYPOINTS_SAVE_DIR, npy_filename)
                    try:
                        np.save(npy_filepath, np.array(all_keypoints_for_video))
                        logging.info(f"Keypoints sauvegardés: {npy_filepath} ({len(all_keypoints_for_video)} frames)")
                    except Exception as e_save:
                        logging.error(f"Erreur sauvegarde keypoints {npy_filepath}: {e_save}")

                # --- Analyse Prédictions & Log CSV & Suivi Précision (logique inchangée) ---
                final_word = "N/A"; final_word_freq = 0; avg_conf_final_word = 0.0; is_correct = False
                expected_word = get_expected_word_from_filename(base_video_name)
                if not expected_word: logging.warning(f"Mot attendu non extrait de '{base_video_name}'")
                else: word_counts[expected_word] = word_counts.get(expected_word, 0) + 1

                if deadlock_timeout_occurred:
                     final_word = "TIMEOUT_DEADLOCK"; final_word_freq = 0; avg_conf_final_word = 0.0
                     logging.warning(f"Enregistrement CSV: {Colors.RED}{final_word}{Colors.RESET} pour {base_video_name}.")
                elif all_predictions_details:
                    try:
                        prediction_indices = [idx for idx, conf in all_predictions_details]
                        if prediction_indices:
                            index_counts = Counter(prediction_indices)
                            most_common_index, final_word_freq = index_counts.most_common(1)[0]
                            final_word = index_to_word.get(most_common_index, f"Idx_{most_common_index}?")
                            confidences_for_final_word = [conf for idx, conf in all_predictions_details if idx == most_common_index]
                            if confidences_for_final_word: avg_conf_final_word = sum(confidences_for_final_word) / len(confidences_for_final_word)

                            if expected_word and final_word.lower() == expected_word:
                                is_correct = True
                                correct_predictions[expected_word] = correct_predictions.get(expected_word, 0) + 1
                                logging.info(f"-> Mot final: {Colors.BRIGHT_GREEN}{final_word}{Colors.RESET} ({final_word_freq}/{predictions_made}, conf avg: {avg_conf_final_word:.2f}) - CORRECT")
                            else:
                                logger_func = logging.info if expected_word else logging.warning
                                logger_func(f"-> Mot final: {Colors.RED}{final_word}{Colors.RESET} ({final_word_freq}/{predictions_made}, conf avg: {avg_conf_final_word:.2f}) - INCORRECT (Attendu: {expected_word if expected_word else 'N/A'})")
                        else: final_word = "Erreur_Analyse_Indices"
                    except Exception as e_analyze:
                        logging.exception(f"Erreur analyse finale prédictions {base_video_name}: {e_analyze}")
                        final_word = "Erreur_Analyse_Exception"
                else:
                    logging.warning(f"{Colors.RED}-> Aucune prédiction générée pour {base_video_name} (pas de deadlock).{Colors.RESET}")

                try:
                    current_timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
                    with open(PREDICTION_CSV_FILE, 'a', newline='', encoding='utf-8') as csvfile:
                        writer = csv.writer(csvfile)
                        writer.writerow([current_timestamp, base_video_name, final_word, final_word_freq, predictions_made,
                                         f"{avg_conf_final_word:.4f}", f"{max_confidence_seen_video:.4f}", f"{processing_time_sec:.2f}"])
                    logging.info(f"Résultat '{final_word}' ajouté à {PREDICTION_CSV_FILE}")
                except IOError as e_io_csv: logging.error(f"Impossible écrire CSV {PREDICTION_CSV_FILE}: {e_io_csv}")
                except Exception as e_csv: logging.exception(f"Erreur écriture CSV: {e_csv}")

                logging.info(f"{Colors.BRIGHT_YELLOW}--- Fin traitement complet {base_video_name} ---{Colors.RESET}")

        # --- Fin boucle principale vidéos ---
        total_main_time = time.time() - main_start_time
        logging.info(f"{Colors.BRIGHT_GREEN}=== Traitement des {len(video_files_to_process)} vidéos terminé en {total_main_time:.2f} secondes. ===")

        # --- Calcul & Affichage Précision Globale & par Mot (inchangé) ---
        total_videos_processed_for_accuracy = sum(word_counts.values())
        total_correct_overall = sum(correct_predictions.values())
        if total_videos_processed_for_accuracy > 0:
            overall_accuracy = (total_correct_overall / total_videos_processed_for_accuracy) * 100
            logging.info(f"=== Précision Globale: {total_correct_overall}/{total_videos_processed_for_accuracy} ({overall_accuracy:.2f}%) ===")
        else: logging.info("=== Aucune vidéo traitable pour calculer précision globale. ===")

        word_accuracies = {}
        logging.info("--- Précision par Mot ---")
        sorted_expected_words = sorted(word_counts.keys())
        if not sorted_expected_words: logging.info("Aucun mot attendu extrait.")
        else:
            for word in sorted_expected_words:
                total = word_counts[word]
                correct = correct_predictions.get(word, 0)
                accuracy = (correct / total) * 100 if total > 0 else 0
                word_accuracies[word] = accuracy
                logging.info(f"- {word}: {correct}/{total} ({accuracy:.2f}%)")
        logging.info("------------------------")

        # --- Génération Graphique (inchangé) ---
        if word_accuracies:
            try:
                words = list(word_accuracies.keys()); accuracies = list(word_accuracies.values())
                plt.figure(figsize=(max(10, len(words) * 0.8), 6))
                bars = plt.bar(words, accuracies, color='skyblue')
                plt.xlabel("Mot Attendu (normalisé)"); plt.ylabel("Précision (%)"); plt.title("Précision par Mot")
                plt.ylim(0, 105); plt.xticks(rotation=45, ha='right')
                for bar in bars: yval = bar.get_height(); plt.text(bar.get_x() + bar.get_width()/2.0, yval + 1, f'{yval:.1f}%', va='bottom', ha='center', fontsize=9)
                plt.tight_layout()
                graph_filename = "prediction_accuracy_per_word_capture.png"
                plt.savefig(graph_filename); logging.info(f"Graphique précision sauvegardé: '{graph_filename}'")
                # plt.show() # Décommenter pour afficher
            except Exception as e_plot: logging.error(f"Erreur génération graphique: {e_plot}")
        else: logging.info("Aucune donnée pour générer graphique.")

    except KeyboardInterrupt:
         logging.info(f"{Colors.RED}Arrêt programme (KeyboardInterrupt).{Colors.RESET}")
         if extractor is not None: logging.info("Arrêt propre threads..."); extractor.stop()
    except Exception as e_main_loop:
         logging.exception(f"{Colors.RED}Erreur non gérée boucle principale: {e_main_loop}{Colors.RESET}")
         if extractor is not None: logging.info("Arrêt propre threads après erreur..."); extractor.stop()
    finally:
         # --- Nettoyage final global ---
         logging.info("Nettoyage final: Fermeture fenêtres OpenCV...")
         cv2.destroyAllWindows(); cv2.waitKey(1); cv2.waitKey(1)
         if 'model' in locals() and model is not None:
             try: logging.debug("Libération session Keras globale..."); tf.keras.backend.clear_session(); del model; logging.debug("Session Keras libérée.")
             except Exception as e: logging.warning(f"Erreur nettoyage final Keras: {e}")
         time.sleep(0.5)
         logging.info(f"{Colors.BRIGHT_GREEN}Programme terminé.{Colors.RESET}")

if __name__ == "__main__":
    try: main()
    except Exception as e: logging.exception(f"{Colors.RED}Erreur non gérée au niveau __main__: {e}{Colors.RESET}")
    finally: cv2.destroyAllWindows(); cv2.waitKey(1); logging.info("Sortie finale script.")