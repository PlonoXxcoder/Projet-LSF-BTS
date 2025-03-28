# -*- coding: utf-8 -*- # Ajout pour encodage si nécessaire
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

# --- Configuration du logging ---
# Garder DEBUG pour voir tous les messages lors du test du timeout
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - [%(threadName)s] - %(message)s')

# --- ANSI escape codes for colors ---
class Colors:
    RESET = '\x1b[0m'
    BRIGHT_YELLOW = '\x1b[93m'
    BRIGHT_GREEN = '\x1b[92m'
    RED = '\x1b[31m'
    GREEN = '\x1b[32m' # Ajout pour le démarrage du thread
    CV_GREEN = (0, 255, 0)
    CV_YELLOW = (0, 255, 255)
    CV_RED = (0, 0, 255)
    CV_WHITE = (255, 255, 255)

# --- Constants ---
MODEL_PATH = "models/model.h5"
VOCABULARY_FILE = "vocabulaire.txt"
FIXED_LENGTH = 46
VIDEOS_DIR = "D:/bonneaup.SNIRW/Test2/video" # CHEMIN ABSOLU (à adapter)

# --- CONFIGURATION ---
PREDICTION_CSV_FILE = "prediction_log.csv"
CSV_HEADER = ["Timestamp", "VideoFile", "MostFrequentWord", "Frequency", "TotalPredictions", "AvgConfidenceMostFrequent", "MaxConfidenceSeen", "ProcessingTimeSec"]
SAVE_KEYPOINTS = True
KEYPOINTS_SAVE_DIR = "extracted_keypoints"
TOP_N = 3
SMOOTHING_WINDOW_SIZE = 15
CONF_THRESH_GREEN = 0.80
CONF_THRESH_YELLOW = 0.50
FRAMES_TO_SKIP = 3
MAX_FRAME_WIDTH = 1280 # Mettre à None pour désactiver redimensionnement
# <<< TIMEOUT DE DEADLOCK MODIFIÉ À 10 SECONDES >>>
DEADLOCK_TIMEOUT = 10 # Secondes d'inactivité keypoints + capture morte avant de forcer

# --- Paramètres d'Extraction ---
NUM_POSE_KEYPOINTS = 4
NUM_HAND_KEYPOINTS = 21
NUM_COORDS = 3
FEATURES_PER_FRAME = (NUM_POSE_KEYPOINTS * NUM_COORDS) + \
                     (NUM_HAND_KEYPOINTS * NUM_COORDS) * 2

# --- Logging des configurations ---
logging.info("--- Configuration ---")
logging.info(f"MODEL_PATH: {MODEL_PATH}")
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
logging.info(f"DEADLOCK_TIMEOUT: {DEADLOCK_TIMEOUT}s") # Log de la valeur mise à jour
logging.info(f"FEATURES_PER_FRAME: {FEATURES_PER_FRAME}")
logging.info("--------------------")

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
    pose = np.array([[res.x, res.y, res.z] for res in results.pose_landmarks.landmark[:NUM_POSE_KEYPOINTS]]).flatten() \
        if results.pose_landmarks else np.zeros(NUM_POSE_KEYPOINTS * NUM_COORDS)
    lh = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]).flatten() \
        if results.left_hand_landmarks else np.zeros(NUM_HAND_KEYPOINTS * NUM_COORDS)
    rh = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]).flatten() \
        if results.right_hand_landmarks else np.zeros(NUM_HAND_KEYPOINTS * NUM_COORDS)

    extracted = np.concatenate([pose, lh, rh])
    if extracted.shape[0] != FEATURES_PER_FRAME:
        logging.warning(f"Extraction a produit {extracted.shape[0]} features, attendu {FEATURES_PER_FRAME}. Retour zéros.")
        return np.zeros(FEATURES_PER_FRAME)
    return extracted

# --- Keypoint Extractor Class ---
class KeypointExtractor:
    def __init__(self):
        self.mp_holistic = mp.solutions.holistic
        self.holistic = None
        self.frame_queue = queue.Queue(maxsize=50)
        self.keypoint_queue = queue.Queue(maxsize=50)
        self.running = threading.Event()
        self.video_capture = None
        self.capture_thread = None
        self.extraction_thread = None
        self.video_path = None

    # <<< capture_frames AVEC TRY ENVELOPPANT TOUT >>>
    def capture_frames(self):
        """Thread function to capture frames from video."""
        frame_count_read = 0
        frame_count_queued = 0
        capture_start_time = time.time()
        self.video_capture = None

        try:
            logging.info(f"{Colors.GREEN}>>> DÉMARRAGE THREAD CAPTURE pour {os.path.basename(self.video_path)}{Colors.RESET}")

            self.video_capture = cv2.VideoCapture(self.video_path)
            if not self.video_capture.isOpened():
                logging.error(f"Impossible d'ouvrir vidéo : {self.video_path}")
                raise ValueError(f"Impossible d'ouvrir vidéo : {self.video_path}")

            frame_width = int(self.video_capture.get(cv2.CAP_PROP_FRAME_WIDTH))
            frame_height = int(self.video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = self.video_capture.get(cv2.CAP_PROP_FPS)
            logging.info(f"Vidéo ouverte: {os.path.basename(self.video_path)} ({frame_width}x{frame_height} @ {fps:.2f} FPS)")

            while self.running.is_set():
                logging.debug(f"[Capture] Tentative lecture frame {frame_count_read}...")
                read_start = time.time()
                ret, frame = self.video_capture.read()
                read_time = time.time() - read_start

                if not ret:
                    logging.info(f"Fin vidéo ou erreur lecture après {frame_count_read} frames lues: {os.path.basename(self.video_path)}.")
                    break

                logging.debug(f"[Capture] Frame {frame_count_read} lue (ret={ret}) en {read_time:.4f}s.")
                frame_count_read += 1

                if (frame_count_read -1) % FRAMES_TO_SKIP == 0:
                    frame_to_queue = frame
                    if MAX_FRAME_WIDTH and frame is not None:
                        h, w = frame.shape[:2]
                        if w > MAX_FRAME_WIDTH:
                            scale = MAX_FRAME_WIDTH / w
                            new_h, new_w = int(h * scale), MAX_FRAME_WIDTH
                            try:
                                frame_to_queue = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)
                                logging.debug(f"[Capture] Frame {frame_count_read} redim. {w}x{h} -> {new_w}x{new_h}")
                            except Exception as resize_err:
                                logging.warning(f"[Capture] Erreur redim. frame {frame_count_read}: {resize_err}")
                                frame_to_queue = frame

                    if frame_to_queue is not None:
                        try:
                            put_start = time.time()
                            self.frame_queue.put(frame_to_queue, timeout=2.0)
                            put_time = time.time() - put_start
                            frame_count_queued += 1
                            if put_time > 0.1:
                                logging.debug(f"[Capture] Mise en queue frame {frame_count_queued} (lue #{frame_count_read}) lente: {put_time:.3f}s.")
                        except queue.Full:
                             logging.warning(f"[Capture] Queue frames pleine (frame {frame_count_queued}). Attente...")
                             try:
                                 self.frame_queue.put(frame_to_queue)
                                 frame_count_queued += 1
                                 logging.info("[Capture] Place trouvée queue frames.")
                             except Exception as e_block:
                                 logging.error(f"[Capture] Échec final mise en queue après attente: {e_block}")
                                 break
                        except Exception as e_put:
                             logging.exception(f"[Capture] Erreur inattendue frame_queue.put : {e_put}")
                             break
                    else:
                        logging.warning(f"[Capture] frame_to_queue est None (frame #{frame_count_read}), skip.")
                else:
                    logging.debug(f"[Capture] Frame #{frame_count_read} sautée (skip={FRAMES_TO_SKIP}).")
            logging.debug(f"[Capture] Sortie de la boucle while pour {os.path.basename(self.video_path)}.")

        except Exception as e_globale_capture:
            logging.exception(f"{Colors.RED}!!! ERREUR GLOBALE CAPTURÉE dans capture_frames pour {os.path.basename(self.video_path)} !!! : {e_globale_capture}{Colors.RESET}")
        finally:
            logging.debug(f"{Colors.RED}>>> Entrée dans le FINALLY de capture_frames pour {os.path.basename(self.video_path)}{Colors.RESET}")
            if self.video_capture:
                 if self.video_capture.isOpened():
                     logging.debug(f"[Capture - Finally] Libération VideoCapture...")
                     self.video_capture.release()
                     logging.info(f"Capture vidéo relâchée pour {os.path.basename(self.video_path)}")
                 else: logging.debug(f"[Capture - Finally] VideoCapture existe mais n'était pas/plus ouvert.")
            else: logging.debug(f"[Capture - Finally] VideoCapture était None.")

            if self.running.is_set():
                 logging.info(f"[Capture - Finally] Flag 'running' est True. Tentative d'envoi signal fin (None) à frame_queue...")
                 try:
                     self.frame_queue.put(None, timeout=5.0)
                     logging.info("[Capture - Finally] Signal fin (None) envoyé avec succès à frame_queue.")
                 except queue.Full:
                     logging.error("[Capture - Finally] Échec envoi signal fin (None) - Queue frames pleine.")
                 except Exception as e_final:
                     logging.error(f"[Capture - Finally] Erreur lors de l'envoi signal fin (None) : {e_final}")
            else:
                 logging.warning(f"[Capture - Finally] Flag 'running' est False. Pas d'envoi de signal fin (None).")

            total_time = time.time() - capture_start_time
            logging.info(f"Thread capture terminé pour {os.path.basename(self.video_path)}. {frame_count_queued}/{frame_count_read} frames en queue en {total_time:.2f}s.")
            logging.info(f"{Colors.RED}### FIN THREAD CAPTURE pour {os.path.basename(self.video_path)} ###{Colors.RESET}")
            logging.debug(f"{Colors.RED}<<< Sortie du FINALLY de capture_frames pour {os.path.basename(self.video_path)}{Colors.RESET}")

    # <<< extract_keypoints_loop reste le même >>>
    def extract_keypoints_loop(self):
        """Thread function to extract keypoints."""
        logging.info(f"Démarrage boucle extraction pour {os.path.basename(self.video_path)}...")
        frames_processed = 0
        extraction_start_time = time.time()
        holistic_instance = None
        try:
            holistic_instance = self.mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5)
            logging.info("[Extraction] Instance MediaPipe Holistic créée.")

            while True:
                if not self.running.is_set() and self.frame_queue.empty():
                    logging.info(f"[Extraction] Arrêt demandé et queue frames vide. Arrêt boucle.")
                    break

                frame = None
                try:
                    get_start = time.time()
                    frame = self.frame_queue.get(timeout=0.5)
                    get_time = time.time() - get_start
                    logging.debug(f"[Extraction] Frame récupérée queue en {get_time:.4f}s.")
                    if frame is None:
                        logging.info(f"[Extraction] Signal fin (None) reçu de frame_queue. Fin normale.")
                        break
                except queue.Empty:
                    logging.debug("[Extraction] Queue frames vide (timeout get). Vérification état...")
                    continue

                try:
                    logging.debug(f"[Extraction] Début MediaPipe frame {frames_processed + 1}...")
                    process_start = time.time()
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    frame_rgb.flags.writeable = False
                    results = holistic_instance.process(frame_rgb)
                    process_time = time.time() - process_start
                    logging.debug(f"[Extraction] Fin MediaPipe en {process_time:.4f}s.")

                    keypoints = extract_keypoints(results)
                    frames_processed += 1

                    put_start = time.time()
                    self.keypoint_queue.put(keypoints, timeout=2.0)
                    put_time = time.time() - put_start
                    if put_time > 0.1:
                         logging.debug(f"[Extraction] Mise en queue keypoints {frames_processed} lente: {put_time:.3f}s.")

                except queue.Full:
                    logging.warning(f"[Extraction] Queue keypoints pleine (kp {frames_processed}). Attente...")
                    try:
                        self.keypoint_queue.put(keypoints)
                        logging.info("[Extraction] Place trouvée queue keypoints.")
                    except Exception as e_block_kp:
                        logging.error(f"[Extraction] Échec final mise en queue keypoints après attente : {e_block_kp}")
                        break
                except Exception as e_process:
                    logging.exception(f"[Extraction] Erreur pendant traitement MediaPipe/extraction: {e_process}")
                    pass

        except Exception as e_init_loop:
             logging.exception(f"[Extraction] Erreur majeure initialisation ou boucle extraction: {e_init_loop}")
        finally:
             logging.debug(f"{Colors.RED}>>> Entrée dans le FINALLY de extract_keypoints_loop pour {os.path.basename(self.video_path)}{Colors.RESET}")
             if holistic_instance:
                 logging.debug("[Extraction - Finally] Fermeture instance Holistic...")
                 try:
                     holistic_instance.close()
                     logging.debug("[Extraction - Finally] Instance Holistic fermée.")
                 except Exception as e_close_holistic:
                     logging.warning(f"[Extraction - Finally] Erreur fermeture Holistic: {e_close_holistic}")

             if self.running.is_set():
                 logging.info(f"[Extraction - Finally] Flag 'running' est True. Tentative envoi signal fin (None) à keypoint_queue...")
                 try:
                     self.keypoint_queue.put(None, timeout=5.0)
                     logging.info("[Extraction - Finally] Signal fin (None) envoyé avec succès à keypoint_queue.")
                 except queue.Full:
                     logging.error("[Extraction - Finally] Échec envoi signal fin (None) - Queue keypoints pleine.")
                 except Exception as e_final_kp:
                     logging.error(f"[Extraction - Finally] Erreur envoi signal fin (None) à keypoint_queue : {e_final_kp}")
             else:
                 logging.warning(f"[Extraction - Finally] Flag 'running' est False. Pas d'envoi de signal fin (None).")

             total_time = time.time() - extraction_start_time
             logging.info(f"Fin boucle extraction pour {os.path.basename(self.video_path)}. Traité {frames_processed} frames en {total_time:.2f}s.")
             logging.info(f"{Colors.RED}### FIN THREAD EXTRACTION pour {os.path.basename(self.video_path)} ###{Colors.RESET}")
             logging.debug(f"{Colors.RED}<<< Sortie du FINALLY de extract_keypoints_loop pour {os.path.basename(self.video_path)}{Colors.RESET}")

    # <<< start reste le même >>>
    def start(self, video_path):
        """Starts the capture and extraction threads."""
        if self.capture_thread is not None and self.capture_thread.is_alive() or \
           self.extraction_thread is not None and self.extraction_thread.is_alive():
            logging.warning("Tentative de démarrer alors que threads actifs. Appel stop() d'abord...")
            self.stop()

        self.video_path = video_path
        self.running.set()

        logging.debug("Vidage queues avant démarrage...")
        while not self.frame_queue.empty():
             try: self.frame_queue.get_nowait()
             except queue.Empty: break
             except Exception as e_clear: logging.warning(f"Erreur vidage frame_queue: {e_clear}")
        while not self.keypoint_queue.empty():
             try: self.keypoint_queue.get_nowait()
             except queue.Empty: break
             except Exception as e_clear: logging.warning(f"Erreur vidage keypoint_queue: {e_clear}")
        logging.debug("Queues vidées.")

        self.capture_thread = threading.Thread(target=self.capture_frames, name=f"Capture-{os.path.basename(video_path)}")
        self.extraction_thread = threading.Thread(target=self.extract_keypoints_loop, name=f"Extract-{os.path.basename(video_path)}")

        self.extraction_thread.start()
        self.capture_thread.start()
        logging.info(f"Threads démarrés pour {os.path.basename(video_path)}")

    # <<< stop reste le même >>>
    def stop(self):
        """Signals threads to stop and waits for them to finish with detailed logging."""
        logging.info(f"{Colors.BRIGHT_YELLOW}>>> Entrée dans stop() pour {os.path.basename(self.video_path)}{Colors.RESET}")
        self.running.clear()
        logging.info(f"Flag 'running' mis à False.")

        join_timeout_capture = 10
        join_timeout_extract = 20

        if self.capture_thread is not None:
            thread_name = self.capture_thread.name
            if self.capture_thread.is_alive():
                logging.info(f"Attente fin thread capture '{thread_name}' (max {join_timeout_capture}s)...")
                self.capture_thread.join(timeout=join_timeout_capture)
                if self.capture_thread.is_alive():
                    logging.warning(f"{Colors.RED}TIMEOUT!{Colors.RESET} Thread capture '{thread_name}' n'a pas terminé.")
                else:
                    logging.info(f"Thread capture '{thread_name}' terminé.")
            else:
                logging.debug(f"Thread capture '{thread_name}' déjà terminé avant join().")
        else:
             logging.debug("stop(): Thread capture non trouvé (None).")

        if self.extraction_thread is not None:
            thread_name = self.extraction_thread.name
            if self.extraction_thread.is_alive():
                logging.info(f"Attente fin thread extraction '{thread_name}' (max {join_timeout_extract}s)...")
                self.extraction_thread.join(timeout=join_timeout_extract)
                if self.extraction_thread.is_alive():
                     logging.warning(f"{Colors.RED}TIMEOUT!{Colors.RESET} Thread extraction '{thread_name}' n'a pas terminé.")
                else:
                      logging.info(f"Thread extraction '{thread_name}' terminé.")
            else:
                 logging.debug(f"Thread extraction '{thread_name}' déjà terminé avant join().")
        else:
             logging.debug("stop(): Thread extraction non trouvé (None).")

        logging.info(f"Vérification arrêt threads terminée pour {os.path.basename(self.video_path)}.")
        self.capture_thread = None
        self.extraction_thread = None
        logging.info(f"{Colors.BRIGHT_YELLOW}<<< Sortie de stop() pour {os.path.basename(self.video_path)}{Colors.RESET}")


# --- Main Function (CORRIGÉE AVEC FLAG TIMEOUT) ---
def main():
    # --- Initialisation TF et Modèle ---
    model = None
    try:
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            try:
                for gpu in gpus: tf.config.experimental.set_memory_growth(gpu, True)
                logging.info(f"GPU détecté et configuré : {gpus}")
            except RuntimeError as e: logging.error(f"Erreur config GPU: {e}")
        else: logging.warning("Aucun GPU détecté. Utilisation CPU.")

        if not os.path.exists(MODEL_PATH):
             logging.error(f"Fichier modèle non trouvé : {MODEL_PATH}"); return
        model = tf.keras.models.load_model(MODEL_PATH)
        logging.info(f"Modèle chargé depuis {MODEL_PATH}")
        try:
            expected_shape = model.input_shape
            logging.info(f"Forme entrée attendue modèle : {expected_shape}")
            if len(expected_shape) != 3 or expected_shape[1] != FIXED_LENGTH or expected_shape[2] != FEATURES_PER_FRAME:
                 logging.warning(f"Incohérence Shape! Modèle={expected_shape}, Script=(None, {FIXED_LENGTH}, {FEATURES_PER_FRAME})")
        except Exception as e: logging.warning(f"Impossible vérifier forme entrée modèle: {e}")
    except Exception as e:
        logging.exception(f"Erreur majeure init TF/Modèle : {e}"); return

    # --- Chargement Vocabulaire ---
    vocabulaire = load_vocabulary(VOCABULARY_FILE)
    if not vocabulaire: logging.error("Erreur critique : Vocabulaire vide. Arrêt."); return
    index_to_word = {i: word for word, i in vocabulaire.items()}

    # --- Vérification Dossier Vidéos et Listing ---
    if not os.path.isdir(VIDEOS_DIR): logging.error(f"Chemin vidéo non valide : {VIDEOS_DIR}"); return
    try:
        video_files_to_process = [f for f in os.listdir(VIDEOS_DIR) if f.lower().endswith(('.mp4', '.avi', '.mov', '.mkv'))]
        video_files_to_process.sort()
        logging.debug(f"Fichiers vidéo trouvés et triés: {video_files_to_process}")
    except Exception as e: logging.exception(f"Erreur listage fichiers dans {VIDEOS_DIR}: {e}"); return
    logging.info(f"Trouvé {len(video_files_to_process)} vidéos à traiter dans {VIDEOS_DIR}")
    if not video_files_to_process: logging.info("Aucune vidéo trouvée. Fin."); return

    # --- Préparation Sauvegarde Keypoints et CSV ---
    if SAVE_KEYPOINTS:
        try:
            os.makedirs(KEYPOINTS_SAVE_DIR, exist_ok=True)
            logging.info(f"Sauvegarde keypoints activée -> '{KEYPOINTS_SAVE_DIR}'")
        except OSError as e: logging.error(f"Impossible créer dossier keypoints '{KEYPOINTS_SAVE_DIR}': {e}")
    try:
        file_exists = os.path.isfile(PREDICTION_CSV_FILE)
        write_header = not file_exists or os.path.getsize(PREDICTION_CSV_FILE) == 0
        with open(PREDICTION_CSV_FILE, 'a', newline='', encoding='utf-8') as csvfile:
            writer = csv.writer(csvfile)
            if write_header:
                writer.writerow(CSV_HEADER)
                logging.info(f"Fichier CSV '{PREDICTION_CSV_FILE}' prêt. En-tête écrit si nécessaire.")
    except IOError as e:
        logging.error(f"Impossible ouvrir/écrire en-tête CSV {PREDICTION_CSV_FILE}: {e}")
        # return # Optionnel

    main_start_time = time.time()
    extractor = None
    try:
        # --- Boucle Principale sur les Vidéos avec tqdm ---
        for video_index, video_file in enumerate(tqdm(video_files_to_process, desc="Traitement Vidéos", unit="video")):
            video_path = os.path.join(VIDEOS_DIR, video_file)
            window_name = f"Video - {os.path.basename(video_path)}"
            logging.info(f"--- [{video_index+1}/{len(video_files_to_process)}] Début: {os.path.basename(video_path)} ---")
            video_start_time = time.time()

            extractor = KeypointExtractor()
            extractor.start(video_path)

            sequence_window = deque(maxlen=FIXED_LENGTH)
            all_keypoints_for_video = []
            all_predictions_details = []
            prediction_display_buffer = deque(maxlen=SMOOTHING_WINDOW_SIZE)
            frame_display_buffer = None
            processing_active = True
            frames_processed_main = 0
            predictions_made = 0
            max_confidence_seen_video = 0.0
            last_top_n_text = ["Initialisation..."]
            last_keypoint_time = time.time()
            # <<< AJOUT: Flag pour mémoriser le timeout >>>
            deadlock_timeout_occurred = False

            try:
                while processing_active:
                    keypoints = None
                    try:
                        keypoints = extractor.keypoint_queue.get(timeout=0.5)
                        if keypoints is None:
                            logging.info("[Main] Signal fin (None) reçu keypoint_queue. Fin traitement vidéo.")
                            processing_active = False
                        else:
                            frames_processed_main += 1
                            logging.debug(f"[Main] Keypoints {frames_processed_main} reçus.")
                            if SAVE_KEYPOINTS: all_keypoints_for_video.append(keypoints)
                            sequence_window.append(keypoints)
                            last_keypoint_time = time.time()

                    except queue.Empty:
                        # --- Logique de Timeout / Deadlock ---
                        logging.debug("[Main] Queue keypoints vide (timeout get). Vérification état threads...")
                        # Vérifier l'état des threads ICI pour avoir l'info la plus à jour
                        capture_alive = extractor.capture_thread and extractor.capture_thread.is_alive()
                        extract_alive = extractor.extraction_thread and extractor.extraction_thread.is_alive()

                        # Condition 1: Fin normale
                        if (not extractor.running.is_set() or (not capture_alive and not extract_alive)) and extractor.keypoint_queue.empty():
                            logging.info(f"[Main] Arrêt demandé ou threads morts (C:{capture_alive}, E:{extract_alive}) et queue vide. Fin vidéo.")
                            processing_active = False
                        # Condition 2: Détection Deadlock/Timeout
                        elif not capture_alive and extract_alive:
                            time_since_last_keypoint = time.time() - last_keypoint_time
                            logging.warning(f"[Main] Deadlock potentiel détecté (Capture mort, Extract vivant). Inactivité keypoints: {time_since_last_keypoint:.1f}s / {DEADLOCK_TIMEOUT}s")
                            if time_since_last_keypoint > DEADLOCK_TIMEOUT:
                                logging.error(f"{Colors.RED}[Main] TIMEOUT DE DEADLOCK ({DEADLOCK_TIMEOUT}s) atteint pour {os.path.basename(video_path)}! Forçage arrêt.{Colors.RESET}")
                                # <<< Mémoriser l'occurrence du timeout >>>
                                deadlock_timeout_occurred = True
                                processing_active = False # Forcer sortie boucle while
                            # else: continuer d'attendre
                        # Condition 3: Attente normale
                        else:
                            logging.debug(f"[Main] Attente données/fin threads (running={extractor.running.is_set()}, C_alive={capture_alive}, E_alive={extract_alive})")
                            pass

                    # --- Prédiction si keypoints reçus ---
                    if keypoints is not None:
                        current_sequence_len = len(sequence_window)
                        padded_sequence = None
                        if current_sequence_len <= FIXED_LENGTH:
                            if current_sequence_len < FIXED_LENGTH:
                                padding = np.zeros((FIXED_LENGTH - current_sequence_len, FEATURES_PER_FRAME))
                                padded_sequence = np.concatenate((padding, np.array(sequence_window)), axis=0)
                            else:
                                padded_sequence = np.array(sequence_window)

                            if padded_sequence is not None and padded_sequence.shape == (FIXED_LENGTH, FEATURES_PER_FRAME):
                                reshaped_sequence = np.expand_dims(padded_sequence, axis=0)
                                try:
                                    predict_start = time.time()
                                    res = model.predict(reshaped_sequence, verbose=0)[0]
                                    predict_time = time.time() - predict_start
                                    predictions_made += 1
                                    top_n_indices = np.argsort(res)[-TOP_N:][::-1]
                                    top_n_confidences = res[top_n_indices]
                                    top_n_words = [index_to_word.get(idx, f"Idx_{idx}?") for idx in top_n_indices]
                                    top_pred_idx = top_n_indices[0]; top_pred_conf = top_n_confidences[0]
                                    all_predictions_details.append((top_pred_idx, top_pred_conf))
                                    prediction_display_buffer.append(top_pred_idx)
                                    max_confidence_seen_video = max(max_confidence_seen_video, top_pred_conf)
                                    last_top_n_text = [f"{w} ({c:.2f})" for w, c in zip(top_n_words, top_n_confidences)]
                                    logging.debug(f"[Main] Pred {predictions_made}: Top1={last_top_n_text[0]} (tps: {predict_time:.4f}s)")
                                except Exception as e_pred:
                                    logging.exception(f"[Main] Erreur model.predict: {e_pred}")
                                    last_top_n_text = ["Erreur Prediction"]
                            else:
                                logging.warning(f"[Main] Shape incorrecte ({padded_sequence.shape if padded_sequence is not None else 'None'}) avant préd.")

                    # --- Gestion Affichage ---
                    try:
                        new_frame = extractor.frame_queue.get_nowait()
                        if new_frame is not None: frame_display_buffer = new_frame
                    except queue.Empty: pass

                    if frame_display_buffer is not None and frame_display_buffer.size > 0:
                        display_frame = frame_display_buffer.copy()
                        try:
                            top_conf = 0.0
                            if all_predictions_details: top_conf = all_predictions_details[-1][1]
                            text_color = Colors.CV_RED
                            if top_conf >= CONF_THRESH_GREEN: text_color = Colors.CV_GREEN
                            elif top_conf >= CONF_THRESH_YELLOW: text_color = Colors.CV_YELLOW

                            y_offset = 30
                            for line_idx, line in enumerate(last_top_n_text):
                                current_color = text_color if line_idx == 0 else Colors.CV_WHITE
                                cv2.putText(display_frame, line, (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.7, current_color, 2, cv2.LINE_AA)
                                y_offset += 25

                            if prediction_display_buffer:
                                smoothed_index = Counter(prediction_display_buffer).most_common(1)[0][0]
                                smoothed_word = index_to_word.get(smoothed_index, "?")
                                cv2.putText(display_frame, f"Lisse: {smoothed_word}", (10, display_frame.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, Colors.CV_WHITE, 2, cv2.LINE_AA)

                            cv2.imshow(window_name, display_frame)
                            key = cv2.waitKey(1) & 0xFF
                            if key == ord('q'):
                                logging.info("Touche 'q' pressée, arrêt global demandé.")
                                processing_active = False
                                raise KeyboardInterrupt("Arrêt utilisateur via 'q'")
                        except cv2.error as e_cv:
                            if "NULL window" in str(e_cv) or "Invalid window handle" in str(e_cv):
                                logging.warning(f"[Main] Fenêtre '{window_name}' fermée manuellement?")
                                processing_active = False
                            else: logging.warning(f"[Main] Erreur cv2.imshow/waitKey: {e_cv}")
                        except Exception as e_show:
                             logging.exception(f"[Main] Erreur affichage/texte: {e_show}")
                             processing_active = False

                    if not processing_active:
                         logging.debug("[Main] processing_active est False, sortie boucle vidéo interne.")
                         break
                # --- Fin boucle while processing_active ---
                logging.info(f"Fin boucle traitement principale pour {video_file}.")

            finally:
                # --- Nettoyage APRES chaque vidéo ---
                logging.info(f"--- Nettoyage pour vidéo {os.path.basename(video_path)}... ---")
                if extractor: extractor.stop()
                try:
                    logging.info("Appel tf.keras.backend.clear_session()...")
                    tf.keras.backend.clear_session()
                    logging.info("clear_session() terminé.")
                except Exception as e_clear: logging.warning(f"Erreur lors de clear_session(): {e_clear}")
                try:
                     if cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) >= 0:
                         cv2.destroyWindow(window_name)
                         logging.info(f"Fenêtre '{window_name}' fermée.")
                     else: logging.debug(f"Fenêtre '{window_name}' déjà fermée/non trouvée.")
                except Exception as e_close: logging.warning(f"Erreur fermeture fenêtre '{window_name}': {e_close}")

                video_end_time = time.time()
                processing_time_sec = video_end_time - video_start_time
                logging.info(f"Vidéo {os.path.basename(video_path)}: {frames_processed_main} kp traités, {predictions_made} préds.")
                logging.info(f"Temps traitement vidéo: {processing_time_sec:.2f} sec.")

                # --- Sauvegarde Keypoints (.npy) ---
                if SAVE_KEYPOINTS and all_keypoints_for_video:
                    npy_filename = os.path.splitext(video_file)[0] + ".npy"
                    npy_filepath = os.path.join(KEYPOINTS_SAVE_DIR, npy_filename)
                    try:
                        np.save(npy_filepath, np.array(all_keypoints_for_video))
                        logging.info(f"Keypoints sauvegardés: {npy_filepath}")
                    except Exception as e_save: logging.error(f"Erreur sauvegarde keypoints {npy_filepath}: {e_save}")

                # --- Analyse Finale Prédictions & Log CSV (CORRIGÉ) ---
                final_word = "N/A"; final_word_freq = 0; avg_conf_final_word = 0.0
                # <<< Utiliser le flag deadlock_timeout_occurred >>>
                if deadlock_timeout_occurred:
                     final_word = "TIMEOUT_DEADLOCK"
                     logging.warning(f"Enregistrement CSV : {final_word} pour {video_file}")

                elif all_predictions_details:
                    try:
                        prediction_indices = [idx for idx, conf in all_predictions_details]
                        if prediction_indices:
                            index_counts = Counter(prediction_indices)
                            most_common_index, final_word_freq = index_counts.most_common(1)[0]
                            final_word = index_to_word.get(most_common_index, f"Idx_{most_common_index}?")
                            confidences_for_final_word = [conf for idx, conf in all_predictions_details if idx == most_common_index]
                            if confidences_for_final_word: avg_conf_final_word = sum(confidences_for_final_word) / len(confidences_for_final_word)
                            logging.info(f"-> Mot final: {Colors.BRIGHT_GREEN}{final_word}{Colors.RESET} ({final_word_freq}/{predictions_made} fois, conf avg: {avg_conf_final_word:.2f}, max vue: {max_confidence_seen_video:.2f})")
                        else: logging.warning(f"{Colors.RED}-> Aucune préd valide pour analyse {video_file}.{Colors.RESET}")
                    except Exception as e_analyze:
                        logging.exception(f"Erreur analyse préds finales {video_file}: {e_analyze}")
                        final_word = "Erreur Analyse"
                else:
                    logging.warning(f"{Colors.RED}-> Aucune préd générée pour {video_file}.{Colors.RESET}")

                # Écriture CSV
                try:
                    current_timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
                    with open(PREDICTION_CSV_FILE, 'a', newline='', encoding='utf-8') as csvfile:
                        writer = csv.writer(csvfile)
                        writer.writerow([ current_timestamp, video_file, final_word, final_word_freq, predictions_made, f"{avg_conf_final_word:.4f}", f"{max_confidence_seen_video:.4f}", f"{processing_time_sec:.2f}" ])
                    logging.info(f"Résultat ajouté à {PREDICTION_CSV_FILE}")
                except IOError as e_io_csv: logging.error(f"Impossible écrire CSV {PREDICTION_CSV_FILE}: {e_io_csv}")
                except Exception as e_csv: logging.exception(f"Erreur inattendue écriture CSV: {e_csv}")

            logging.info(f"--- Fin traitement complet pour {os.path.basename(video_path)} ---")
            # time.sleep(0.5)

        # --- Fin boucle principale sur toutes les vidéos ---
        total_main_time = time.time() - main_start_time
        logging.info(f"Traitement des {len(video_files_to_process)} vidéos terminé en {total_main_time:.2f} secondes.")

    except KeyboardInterrupt:
         logging.info("Arrêt programme demandé par utilisateur (KeyboardInterrupt).")
         if extractor is not None and extractor.running.is_set():
             logging.info("Tentative arrêt threads suite à KeyboardInterrupt...")
             extractor.stop()
    except Exception as e_main_loop:
         logging.exception(f"Erreur inattendue dans boucle principale vidéos: {e_main_loop}")
         if extractor is not None and extractor.running.is_set():
             logging.info("Tentative arrêt threads suite à une erreur...")
             extractor.stop()
    finally:
         # Nettoyage final global
         logging.info("Nettoyage final : Fermeture fenêtres OpenCV restantes...")
         cv2.destroyAllWindows()
         if model is not None:
             logging.debug("Libération session Keras globale...")
             tf.keras.backend.clear_session()
             del model
             logging.debug("Session Keras globale libérée.")
         time.sleep(0.5)
         logging.info("Programme terminé.")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logging.exception(f"Erreur non gérée au niveau __main__: {e}")
    finally:
        cv2.destroyAllWindows()
        logging.info("Sortie du script.")