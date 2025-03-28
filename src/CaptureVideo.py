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
import matplotlib.pyplot as plt # <<< Pour le graphique

# --- Configuration du logging ---
# Remettre à INFO pour moins de verbosité une fois le problème de deadlock résolu
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
MODEL_PATH = "models/model.h5"
VOCABULARY_FILE = "vocabulaire.txt"
FIXED_LENGTH = 46
VIDEOS_DIR = "D:/bonneaup.SNIRW/Test2/video" # CHEMIN ABSOLU (à adapter)

# --- CONFIGURATION ---
PREDICTION_CSV_FILE = "prediction_log.csv"
CSV_HEADER = ["Timestamp", "VideoFile", "MostFrequentWord", "Frequency", "TotalPredictions", "AvgConfidenceMostFrequent", "MaxConfidenceSeen", "ProcessingTimeSec"]
SAVE_KEYPOINTS = True # <<< Global constant definition
KEYPOINTS_SAVE_DIR = "extracted_keypoints"
TOP_N = 3
SMOOTHING_WINDOW_SIZE = 15
CONF_THRESH_GREEN = 0.80
CONF_THRESH_YELLOW = 0.50
FRAMES_TO_SKIP = 3 # Process 1 frame every 'FRAMES_TO_SKIP' frames (1 = process all, 2 = process every other, etc.)
MAX_FRAME_WIDTH = 1280
DEADLOCK_TIMEOUT = 10 # Secondes (Timeout increased slightly for robustness)

# --- Paramètres d'Extraction ---
NUM_POSE_KEYPOINTS = 4
NUM_HAND_KEYPOINTS = 21
NUM_COORDS = 3 # x, y, z
FEATURES_PER_FRAME = (NUM_POSE_KEYPOINTS * NUM_COORDS) + \
                     (NUM_HAND_KEYPOINTS * NUM_COORDS) * 2 # Left + Right hand

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
logging.info(f"FRAMES_TO_SKIP: {FRAMES_TO_SKIP} (process 1 frame every {FRAMES_TO_SKIP})")
if MAX_FRAME_WIDTH: logging.info(f"MAX_FRAME_WIDTH: {MAX_FRAME_WIDTH}")
else: logging.info("MAX_FRAME_WIDTH: Désactivé")
logging.info(f"DEADLOCK_TIMEOUT: {DEADLOCK_TIMEOUT}s")
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
    # Use only first NUM_POSE_KEYPOINTS for pose
    pose = np.array([[res.x, res.y, res.z] for res in results.pose_landmarks.landmark[:NUM_POSE_KEYPOINTS]]).flatten() \
        if results.pose_landmarks and len(results.pose_landmarks.landmark) >= NUM_POSE_KEYPOINTS else np.zeros(NUM_POSE_KEYPOINTS * NUM_COORDS)
    # Ensure full hand keypoints or zeros
    lh = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]).flatten() \
        if results.left_hand_landmarks and len(results.left_hand_landmarks.landmark) == NUM_HAND_KEYPOINTS else np.zeros(NUM_HAND_KEYPOINTS * NUM_COORDS)
    rh = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]).flatten() \
        if results.right_hand_landmarks and len(results.right_hand_landmarks.landmark) == NUM_HAND_KEYPOINTS else np.zeros(NUM_HAND_KEYPOINTS * NUM_COORDS)

    extracted = np.concatenate([pose, lh, rh])
    # Final check for expected feature count
    if extracted.shape[0] != FEATURES_PER_FRAME:
        logging.warning(f"Extraction a produit {extracted.shape[0]} features, attendu {FEATURES_PER_FRAME}. Retour zéros.")
        return np.zeros(FEATURES_PER_FRAME)
    return extracted

# <<< MODIFICATION ICI: Ajout de .lower() pour normaliser la casse >>>
def get_expected_word_from_filename(filename):
    """Extrait le mot attendu du nom de fichier et le met en minuscules."""
    name_without_ext = os.path.splitext(filename)[0]
    # Prend la partie avant le premier '_' ou le nom complet si pas de '_'
    parts = name_without_ext.split('_', 1)
    expected_word = parts[0]
    # Convertir en minuscules pour ignorer la casse et enlever les espaces
    return expected_word.strip().lower()
# <<< FIN MODIFICATION >>>

# --- Keypoint Extractor Class ---
class KeypointExtractor:
    def __init__(self):
        self.mp_holistic = mp.solutions.holistic
        self.frame_queue = queue.Queue(maxsize=50) # Queue for frames going TO extraction
        self.keypoint_queue = queue.Queue(maxsize=100) # Queue for keypoints coming FROM extraction (can be larger)
        self.display_queue = queue.Queue(maxsize=10) # Small queue for frames to display
        self.running = threading.Event()
        self.video_capture = None
        self.capture_thread = None
        self.extraction_thread = None
        self.video_path = None

    def capture_frames(self):
        """Thread function to capture frames from video."""
        frame_count_read = 0
        frame_count_queued_extract = 0
        frame_count_queued_display = 0
        capture_start_time = time.time()
        self.video_capture = None

        try:
            logging.info(f"{Colors.GREEN}>>> DÉMARRAGE THREAD CAPTURE pour {os.path.basename(self.video_path)}{Colors.RESET}")

            self.video_capture = cv2.VideoCapture(self.video_path)
            if not self.video_capture.isOpened():
                logging.error(f"Impossible d'ouvrir vidéo : {self.video_path}")
                # Signal error by putting None immediately if needed by other threads?
                # self.frame_queue.put(None) # Or handle this in the main loop/extractor thread
                raise ValueError(f"Impossible d'ouvrir vidéo : {self.video_path}")

            frame_width = int(self.video_capture.get(cv2.CAP_PROP_FRAME_WIDTH))
            frame_height = int(self.video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = self.video_capture.get(cv2.CAP_PROP_FPS)
            total_frames = int(self.video_capture.get(cv2.CAP_PROP_FRAME_COUNT))
            logging.info(f"Vidéo ouverte: {os.path.basename(self.video_path)} ({frame_width}x{frame_height} @ {fps:.2f} FPS, Total Frames: {total_frames})")

            target_width = frame_width
            target_height = frame_height
            resize_needed = False
            if MAX_FRAME_WIDTH and frame_width > MAX_FRAME_WIDTH:
                scale = MAX_FRAME_WIDTH / frame_width
                target_width = MAX_FRAME_WIDTH
                target_height = int(frame_height * scale)
                resize_needed = True
                logging.info(f"Redimensionnement activé: {frame_width}x{frame_height} -> {target_width}x{target_height}")

            while self.running.is_set():
                read_start = time.time()
                ret, frame = self.video_capture.read()
                read_time = time.time() - read_start

                if not ret:
                    logging.info(f"Fin vidéo ou erreur lecture après {frame_count_read} frames lues: {os.path.basename(self.video_path)}.")
                    break

                frame_count_read += 1

                # Skip frames logic
                if (frame_count_read - 1) % FRAMES_TO_SKIP == 0:
                    frame_to_process = frame
                    # Resize if necessary
                    if resize_needed:
                        try:
                            frame_to_process = cv2.resize(frame, (target_width, target_height), interpolation=cv2.INTER_AREA)
                            # logging.debug(f"[Capture] Frame {frame_count_read} redim.") # Less verbose
                        except Exception as resize_err:
                            logging.warning(f"[Capture] Erreur redim. frame {frame_count_read}: {resize_err}")
                            frame_to_process = frame # Use original on error

                    if frame_to_process is not None:
                        # Put frame in extraction queue
                        try:
                            self.frame_queue.put(frame_to_process, timeout=2.0)
                            frame_count_queued_extract += 1
                        except queue.Full:
                             logging.warning(f"[Capture] Queue frames (extract) pleine (frame {frame_count_queued_extract}). Attente...")
                             try:
                                 self.frame_queue.put(frame_to_process) # Blocking put
                                 frame_count_queued_extract += 1
                                 logging.info("[Capture] Place trouvée queue frames (extract).")
                             except Exception as e_block:
                                 logging.error(f"[Capture] Échec final mise en queue (extract) après attente: {e_block}")
                                 self.running.clear() # Signal other threads to stop
                                 break
                        except Exception as e_put_extract:
                             logging.exception(f"[Capture] Erreur inattendue frame_queue.put : {e_put_extract}")
                             self.running.clear()
                             break

                        # Put frame in display queue (try non-blocking first)
                        try:
                            self.display_queue.put_nowait(frame_to_process)
                            frame_count_queued_display += 1
                        except queue.Full:
                            # If display queue is full, drop the oldest frame and add the new one
                            try:
                                self.display_queue.get_nowait() # Remove oldest
                                self.display_queue.put_nowait(frame_to_process) # Add newest
                                # logging.debug("[Capture] Display queue pleine, frame remplacée.") # Less verbose
                            except queue.Empty: pass # Should not happen if Full was raised
                            except Exception as e_display_replace:
                                logging.warning(f"[Capture] Erreur remplacement frame display queue: {e_display_replace}")
                        except Exception as e_put_display:
                             logging.warning(f"[Capture] Erreur inattendue display_queue.put : {e_put_display}")

                    else:
                        logging.warning(f"[Capture] frame_to_process est None (frame #{frame_count_read}), skip.")

            logging.debug(f"[Capture] Sortie de la boucle while pour {os.path.basename(self.video_path)}.")

        except ValueError: # Catch the VideoCapture open error specifically
             pass # Already logged
        except Exception as e_globale_capture:
            logging.exception(f"{Colors.RED}!!! ERREUR GLOBALE CAPTURÉE dans capture_frames pour {os.path.basename(self.video_path)} !!! : {e_globale_capture}{Colors.RESET}")
            self.running.clear() # Signal stop on major error
        finally:
            logging.debug(f"{Colors.RED}>>> Entrée dans le FINALLY de capture_frames pour {os.path.basename(self.video_path)}{Colors.RESET}")
            if self.video_capture:
                 if self.video_capture.isOpened():
                     logging.debug(f"[Capture - Finally] Libération VideoCapture...")
                     self.video_capture.release()
                     logging.info(f"Capture vidéo relâchée pour {os.path.basename(self.video_path)}")
                 else: logging.debug(f"[Capture - Finally] VideoCapture existe mais n'était pas/plus ouvert.")
            else: logging.debug(f"[Capture - Finally] VideoCapture était None.")

            # Always try to send termination signal (None) to frame_queue for the extractor
            logging.info(f"[Capture - Finally] Tentative d'envoi signal fin (None) à frame_queue...")
            try:
                # Use put with timeout, but don't block indefinitely if extractor died
                self.frame_queue.put(None, timeout=5.0)
                logging.info("[Capture - Finally] Signal fin (None) envoyé avec succès à frame_queue.")
            except queue.Full:
                logging.error("[Capture - Finally] Échec envoi signal fin (None) - Queue frames pleine. L'extracteur pourrait attendre indéfiniment.")
            except Exception as e_final:
                logging.error(f"[Capture - Finally] Erreur lors de l'envoi signal fin (None) : {e_final}")

            # Display queue doesn't need a None signal as it's read non-blockingly

            total_time = time.time() - capture_start_time
            logging.info(f"Thread capture terminé pour {os.path.basename(self.video_path)}. {frame_count_queued_extract}/{frame_count_read} frames vers extraction, {frame_count_queued_display} vers affichage en {total_time:.2f}s.")
            logging.info(f"{Colors.RED}### FIN THREAD CAPTURE pour {os.path.basename(self.video_path)} ###{Colors.RESET}")
            logging.debug(f"{Colors.RED}<<< Sortie du FINALLY de capture_frames pour {os.path.basename(self.video_path)}{Colors.RESET}")

    def extract_keypoints_loop(self):
        """Thread function to extract keypoints using MediaPipe."""
        logging.info(f"{Colors.GREEN}>>> DÉMARRAGE THREAD EXTRACTION pour {os.path.basename(self.video_path)}{Colors.RESET}")
        frames_processed = 0
        extraction_start_time = time.time()
        holistic_instance = None
        try:
            # Initialize MediaPipe Holistic within the thread
            holistic_instance = self.mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5)
            logging.info("[Extraction] Instance MediaPipe Holistic créée.")

            while True:
                # Check running flag first
                if not self.running.is_set():
                    logging.info("[Extraction] Arrêt demandé (running=False). Vidage queue frames...")
                    # Try to process remaining frames quickly if stop was requested
                    while True:
                        try:
                            frame = self.frame_queue.get_nowait()
                            if frame is None: break # End signal found
                            # Process the frame (optional, could just drain)
                            # ... (process frame logic) ...
                        except queue.Empty:
                            break # No more frames
                    logging.info("[Extraction] Queue frames vidée après demande d'arrêt.")
                    break # Exit main extraction loop

                frame = None
                try:
                    # Wait for a frame from the capture thread
                    frame = self.frame_queue.get(timeout=1.0) # Increased timeout slightly
                    if frame is None:
                        logging.info(f"[Extraction] Signal fin (None) reçu de frame_queue. Fin normale.")
                        break # Exit loop on termination signal
                except queue.Empty:
                    # Timeout occurred, check if capture thread is still alive
                    # logging.debug("[Extraction] Queue frames vide (timeout get). Vérification état capture...") # Less verbose
                    if self.capture_thread and not self.capture_thread.is_alive():
                        logging.warning("[Extraction] Queue frames vide et thread capture est mort. Arrêt extraction.")
                        break # Capture thread died, no more frames will come
                    else:
                        continue # Capture thread alive, just waiting for frames

                # Process the frame with MediaPipe
                try:
                    process_start = time.time()
                    # Convert the BGR image to RGB.
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    # To improve performance, optionally mark the image as not writeable to pass by reference.
                    frame_rgb.flags.writeable = False
                    # Process the image and find holistic landmarks.
                    results = holistic_instance.process(frame_rgb)
                    process_time = time.time() - process_start

                    # Extract keypoints
                    keypoints = extract_keypoints(results)
                    frames_processed += 1

                    # Put keypoints into the keypoint queue
                    try:
                        self.keypoint_queue.put(keypoints, timeout=2.0)
                        # logging.debug(f"[Extraction] Keypoints {frames_processed} mis en queue.") # Less verbose
                    except queue.Full:
                        logging.warning(f"[Extraction] Queue keypoints pleine (kp {frames_processed}). Attente...")
                        try:
                            self.keypoint_queue.put(keypoints) # Blocking put
                            logging.info("[Extraction] Place trouvée queue keypoints.")
                        except Exception as e_block_kp:
                            logging.error(f"[Extraction] Échec final mise en queue keypoints après attente : {e_block_kp}")
                            self.running.clear() # Signal stop
                            break # Exit loop
                    except Exception as e_put_kp:
                        logging.exception(f"[Extraction] Erreur inattendue keypoint_queue.put : {e_put_kp}")
                        self.running.clear() # Signal stop
                        break # Exit loop

                except Exception as e_process:
                    logging.exception(f"[Extraction] Erreur pendant traitement MediaPipe/extraction frame {frames_processed + 1}: {e_process}")
                    # Decide whether to continue or stop on processing error
                    # For now, we continue to the next frame
                    pass

            logging.debug(f"[Extraction] Sortie de la boucle while pour {os.path.basename(self.video_path)}.")

        except Exception as e_init_loop:
             logging.exception(f"[Extraction] Erreur majeure initialisation ou boucle extraction: {e_init_loop}")
             self.running.clear() # Signal stop on major error
        finally:
             logging.debug(f"{Colors.RED}>>> Entrée dans le FINALLY de extract_keypoints_loop pour {os.path.basename(self.video_path)}{Colors.RESET}")
             if holistic_instance:
                 logging.debug("[Extraction - Finally] Fermeture instance Holistic...")
                 try:
                     holistic_instance.close()
                     logging.debug("[Extraction - Finally] Instance Holistic fermée.")
                 except Exception as e_close_holistic:
                     logging.warning(f"[Extraction - Finally] Erreur fermeture Holistic: {e_close_holistic}")

             # Always try to send termination signal (None) to keypoint_queue for the main thread
             logging.info(f"[Extraction - Finally] Tentative envoi signal fin (None) à keypoint_queue...")
             try:
                 # Use put with timeout, but don't block indefinitely if main thread died
                 self.keypoint_queue.put(None, timeout=5.0)
                 logging.info("[Extraction - Finally] Signal fin (None) envoyé avec succès à keypoint_queue.")
             except queue.Full:
                 logging.error("[Extraction - Finally] Échec envoi signal fin (None) - Queue keypoints pleine. Le thread principal pourrait attendre indéfiniment.")
             except Exception as e_final_kp:
                 logging.error(f"[Extraction - Finally] Erreur envoi signal fin (None) à keypoint_queue : {e_final_kp}")

             total_time = time.time() - extraction_start_time
             logging.info(f"Fin boucle extraction pour {os.path.basename(self.video_path)}. Traité {frames_processed} frames en {total_time:.2f}s.")
             logging.info(f"{Colors.RED}### FIN THREAD EXTRACTION pour {os.path.basename(self.video_path)} ###{Colors.RESET}")
             logging.debug(f"{Colors.RED}<<< Sortie du FINALLY de extract_keypoints_loop pour {os.path.basename(self.video_path)}{Colors.RESET}")

    def start(self, video_path):
        """Starts the capture and extraction threads."""
        if self.capture_thread is not None and self.capture_thread.is_alive() or \
           self.extraction_thread is not None and self.extraction_thread.is_alive():
            logging.warning(f"{Colors.BRIGHT_YELLOW}Tentative de démarrer alors que threads actifs pour {os.path.basename(self.video_path)}. Appel stop() d'abord...{Colors.RESET}")
            self.stop() # Ensure clean state before starting

        self.video_path = video_path
        self.running.set() # Set running flag BEFORE starting threads

        # Clear queues before starting new video processing
        logging.debug("Vidage queues avant démarrage...")
        queues_to_clear = [self.frame_queue, self.keypoint_queue, self.display_queue]
        for q in queues_to_clear:
            while not q.empty():
                try: q.get_nowait()
                except queue.Empty: break
                except Exception as e_clear: logging.warning(f"Erreur vidage queue {q}: {e_clear}")
        logging.debug("Queues vidées.")

        # Create and start threads
        self.capture_thread = threading.Thread(target=self.capture_frames, name=f"Capture-{os.path.basename(video_path)}")
        self.extraction_thread = threading.Thread(target=self.extract_keypoints_loop, name=f"Extract-{os.path.basename(video_path)}")

        self.extraction_thread.start()
        self.capture_thread.start()
        logging.info(f"Threads démarrés pour {os.path.basename(video_path)}")

    def stop(self):
        """Signals threads to stop and waits for them to finish with detailed logging."""
        video_name = os.path.basename(self.video_path) if self.video_path else "Unknown Video"
        logging.info(f"{Colors.BRIGHT_YELLOW}>>> Entrée dans stop() pour {video_name}{Colors.RESET}")
        self.running.clear() # Signal threads to stop their loops
        logging.info(f"Flag 'running' mis à False pour {video_name}.")

        # Define timeouts for join
        join_timeout_capture = 10 # seconds
        join_timeout_extract = 20 # seconds (extraction might need more time to process remaining queue)

        # Wait for Capture Thread
        if self.capture_thread is not None:
            thread_name = self.capture_thread.name
            if self.capture_thread.is_alive():
                logging.info(f"Attente fin thread capture '{thread_name}' (max {join_timeout_capture}s)...")
                self.capture_thread.join(timeout=join_timeout_capture)
                if self.capture_thread.is_alive():
                    logging.warning(f"{Colors.RED}TIMEOUT!{Colors.RESET} Thread capture '{thread_name}' n'a pas terminé dans le délai imparti.")
                    # Consider more drastic measures if needed, but often just logging is sufficient
                else:
                    logging.info(f"Thread capture '{thread_name}' terminé.")
            else:
                logging.debug(f"Thread capture '{thread_name}' déjà terminé avant appel join().")
        else:
             logging.debug("stop(): Thread capture non trouvé (None).")

        # Wait for Extraction Thread
        if self.extraction_thread is not None:
            thread_name = self.extraction_thread.name
            if self.extraction_thread.is_alive():
                logging.info(f"Attente fin thread extraction '{thread_name}' (max {join_timeout_extract}s)...")
                self.extraction_thread.join(timeout=join_timeout_extract)
                if self.extraction_thread.is_alive():
                     logging.warning(f"{Colors.RED}TIMEOUT!{Colors.RESET} Thread extraction '{thread_name}' n'a pas terminé dans le délai imparti.")
                else:
                      logging.info(f"Thread extraction '{thread_name}' terminé.")
            else:
                 logging.debug(f"Thread extraction '{thread_name}' déjà terminé avant appel join().")
        else:
             logging.debug("stop(): Thread extraction non trouvé (None).")

        # Clean up thread references
        self.capture_thread = None
        self.extraction_thread = None
        logging.info(f"Vérification arrêt threads terminée pour {video_name}.")
        logging.info(f"{Colors.BRIGHT_YELLOW}<<< Sortie de stop() pour {video_name}{Colors.RESET}")


# --- Main Function ---
def main():
    global SAVE_KEYPOINTS # <<< Declare intention to modify global variable if needed

    # --- Initialisation TF et Modèle ---
    model = None
    try:
        # Configure GPU memory growth if GPU is available
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            try:
                for gpu in gpus:
                    tf.config.experimental.set_memory_growth(gpu, True)
                logging.info(f"GPU(s) détecté(s) et configuré(s) pour Memory Growth : {gpus}")
            except RuntimeError as e:
                logging.error(f"Erreur configuration Memory Growth GPU: {e}")
        else:
            logging.warning("Aucun GPU détecté par TensorFlow. Utilisation CPU.")

        # Load the Keras model
        if not os.path.exists(MODEL_PATH):
             logging.error(f"Fichier modèle non trouvé : {MODEL_PATH}"); return
        model = tf.keras.models.load_model(MODEL_PATH)
        logging.info(f"Modèle chargé depuis {MODEL_PATH}")
        try:
            # Log expected input shape
            expected_shape = model.input_shape
            logging.info(f"Forme entrée attendue par le modèle : {expected_shape}")
            # Validate shape against constants
            if len(expected_shape) != 3 or expected_shape[1] != FIXED_LENGTH or expected_shape[2] != FEATURES_PER_FRAME:
                 logging.warning(f"{Colors.RED}Incohérence Shape! Modèle={expected_shape}, Script Config=(None, {FIXED_LENGTH}, {FEATURES_PER_FRAME}){Colors.RESET}")
        except Exception as e:
             logging.warning(f"Impossible de vérifier la forme d'entrée du modèle: {e}")
    except Exception as e:
        logging.exception(f"Erreur majeure lors de l'initialisation TensorFlow/Modèle : {e}"); return

    # --- Chargement Vocabulaire ---
    vocabulaire = load_vocabulary(VOCABULARY_FILE)
    if not vocabulaire:
        logging.error("Erreur critique : Vocabulaire vide ou non chargé. Arrêt."); return
    # Create reverse mapping for prediction output
    index_to_word = {i: word for word, i in vocabulaire.items()}

    # --- Vérification Dossier Vidéos et Listing ---
    if not os.path.isdir(VIDEOS_DIR):
        logging.error(f"Chemin vidéo non valide ou dossier non trouvé : {VIDEOS_DIR}"); return
    try:
        # List video files, sort them for consistent processing order
        video_files_to_process = [f for f in os.listdir(VIDEOS_DIR) if f.lower().endswith(('.mp4', '.avi', '.mov', '.mkv'))]
        video_files_to_process.sort()
        logging.debug(f"Fichiers vidéo trouvés et triés: {video_files_to_process}")
    except Exception as e:
        logging.exception(f"Erreur lors du listage des fichiers dans {VIDEOS_DIR}: {e}"); return

    if not video_files_to_process:
        logging.info(f"Aucune vidéo trouvée dans {VIDEOS_DIR}. Fin du programme."); return
    logging.info(f"Trouvé {len(video_files_to_process)} vidéos à traiter dans {VIDEOS_DIR}")

    # --- Préparation Sauvegarde Keypoints et CSV ---
    if SAVE_KEYPOINTS: # Reading the global variable
        try:
            os.makedirs(KEYPOINTS_SAVE_DIR, exist_ok=True)
            logging.info(f"Sauvegarde des keypoints activée -> Dossier: '{KEYPOINTS_SAVE_DIR}'")
        except OSError as e:
            logging.error(f"Impossible de créer le dossier pour les keypoints '{KEYPOINTS_SAVE_DIR}': {e}")
            SAVE_KEYPOINTS = False # This now modifies the GLOBAL variable
            logging.warning("Sauvegarde des keypoints désactivée en raison de l'erreur de création de dossier.")

    # Prepare CSV file for logging predictions
    try:
        # Check if file exists and is empty to decide whether to write header
        file_exists = os.path.isfile(PREDICTION_CSV_FILE)
        write_header = not file_exists or os.path.getsize(PREDICTION_CSV_FILE) == 0
        with open(PREDICTION_CSV_FILE, 'a', newline='', encoding='utf-8') as csvfile:
            writer = csv.writer(csvfile)
            if write_header:
                writer.writerow(CSV_HEADER)
                logging.info(f"Fichier CSV '{PREDICTION_CSV_FILE}' prêt. En-tête écrit.")
            else:
                 logging.info(f"Fichier CSV '{PREDICTION_CSV_FILE}' prêt. Ajout des données.")
    except IOError as e:
        logging.error(f"Impossible d'ouvrir/écrire l'en-tête CSV dans {PREDICTION_CSV_FILE}: {e}")
        # Consider exiting if CSV logging is critical, or just continue without it
        # return

    main_start_time = time.time()
    extractor = None # Initialize extractor variable outside the loop
    # <<< Dictionnaires pour le suivi de la précision >>>
    word_counts = {} # Compte total par mot attendu (normalisé en minuscules)
    correct_predictions = {} # Compte correct par mot attendu (normalisé en minuscules)

    try:
        # --- Boucle Principale sur les Vidéos avec tqdm ---
        for video_index, video_file in enumerate(tqdm(video_files_to_process, desc="Traitement Vidéos", unit="video")):
            video_path = os.path.join(VIDEOS_DIR, video_file)
            base_video_name = os.path.basename(video_path)
            window_name = f"Video - {base_video_name}" # Unique window name per video
            logging.info(f"{Colors.BRIGHT_YELLOW}--- [{video_index+1}/{len(video_files_to_process)}] Début Traitement: {base_video_name} ---{Colors.RESET}")
            video_start_time = time.time()

            # Initialize KeypointExtractor for the current video
            extractor = KeypointExtractor()
            extractor.start(video_path)

            # Data structures for the current video
            sequence_window = deque(maxlen=FIXED_LENGTH) # Holds the sequence for model input
            all_keypoints_for_video = [] # Stores all extracted keypoints if SAVE_KEYPOINTS is True
            all_predictions_details = [] # Stores tuples of (predicted_index, confidence)
            prediction_display_buffer = deque(maxlen=SMOOTHING_WINDOW_SIZE) # For smoothing displayed prediction
            frame_display_buffer = None # Holds the latest frame for display
            processing_active = True # Flag to control the inner loop
            frames_processed_main = 0 # Count keypoints received in main thread
            predictions_made = 0 # Count predictions made by the model
            max_confidence_seen_video = 0.0 # Track max confidence for this video
            last_top_n_text = ["Initialisation..."] # Text to display on frame
            last_keypoint_time = time.time() # For deadlock detection
            deadlock_timeout_occurred = False # Reset flag for each video

            try:
                # --- Inner loop: Process keypoints and display ---
                while processing_active:
                    keypoints = None
                    try:
                        # Get keypoints from the extraction thread
                        keypoints = extractor.keypoint_queue.get(timeout=0.5) # Wait briefly
                        if keypoints is None:
                            # Termination signal received
                            logging.info(f"[Main] Signal fin (None) reçu de keypoint_queue. Fin traitement pour {base_video_name}.")
                            processing_active = False
                            # No break here, let the loop finish its current iteration naturally
                        else:
                            # Valid keypoints received
                            frames_processed_main += 1
                            last_keypoint_time = time.time() # Update time for deadlock check
                            if SAVE_KEYPOINTS: # Reading the (potentially modified) global variable
                                all_keypoints_for_video.append(keypoints)
                            # Add keypoints to the sequence window for prediction
                            sequence_window.append(keypoints)
                            # logging.debug(f"[Main] Keypoints {frames_processed_main} reçus.") # Less verbose

                    except queue.Empty:
                        # --- Timeout logic / Deadlock detection ---
                        # logging.debug("[Main] Queue keypoints vide (timeout get). Vérification état threads...") # Less verbose
                        capture_alive = extractor.capture_thread and extractor.capture_thread.is_alive()
                        extract_alive = extractor.extraction_thread and extractor.extraction_thread.is_alive()

                        # Condition 1: Normal termination (threads finished AND queue empty)
                        if not extractor.running.is_set() and extractor.keypoint_queue.empty() and not capture_alive and not extract_alive:
                            logging.info(f"[Main] Arrêt demandé ou threads terminés (C:{capture_alive}, E:{extract_alive}) et queue vide. Fin vidéo {base_video_name}.")
                            processing_active = False
                        # Condition 2: Deadlock detection (Capture died, Extract alive, no keypoints for a while)
                        elif not capture_alive and extract_alive:
                            time_since_last_keypoint = time.time() - last_keypoint_time
                            if time_since_last_keypoint > DEADLOCK_TIMEOUT:
                                logging.error(f"{Colors.RED}[Main] TIMEOUT DE DEADLOCK ({DEADLOCK_TIMEOUT}s) détecté pour {base_video_name}! (Capture mort, Extract vivant, pas de keypoints). Forçage arrêt.{Colors.RESET}")
                                deadlock_timeout_occurred = True # Set the flag!
                                processing_active = False # Force stop for this video
                            else:
                                logging.warning(f"[Main] Deadlock potentiel? Capture mort, Extract vivant. Inactivité keypoints: {time_since_last_keypoint:.1f}s / {DEADLOCK_TIMEOUT}s")
                        # Condition 3: Still running normally or waiting for threads to finish
                        elif extractor.running.is_set() or capture_alive or extract_alive:
                             # logging.debug(f"[Main] Attente données/fin threads (running={extractor.running.is_set()}, C_alive={capture_alive}, E_alive={extract_alive}, Q_empty={extractor.keypoint_queue.empty()})") # Less verbose
                             pass # Continue waiting
                        # Condition 4: Both threads seemingly dead but queue wasn't empty on last check? Should eventually hit condition 1.
                        else: # not running, not capture_alive, not extract_alive
                             if not extractor.keypoint_queue.empty():
                                 logging.info("[Main] Threads morts mais queue keypoints pas vide? Tentative de vidage...")
                                 # Loop will continue to try getting from queue until empty or processing_active is false
                             else:
                                 logging.info("[Main] Threads morts et queue vide. Devrait s'arrêter.")
                                 processing_active = False


                    # --- Prediction logic (only if keypoints were received in this iteration) ---
                    if keypoints is not None and not deadlock_timeout_occurred: # Don't predict if deadlock occurred
                        current_sequence_len = len(sequence_window)
                        padded_sequence = None

                        # Only predict if we have enough frames (or pad if fewer)
                        if current_sequence_len > 0: # Need at least one frame
                            if current_sequence_len < FIXED_LENGTH:
                                # Pad sequence at the beginning with zeros
                                padding = np.zeros((FIXED_LENGTH - current_sequence_len, FEATURES_PER_FRAME))
                                padded_sequence = np.concatenate((padding, np.array(sequence_window)), axis=0)
                            else: # Exactly FIXED_LENGTH frames
                                padded_sequence = np.array(sequence_window)

                            # Ensure the sequence has the correct shape before prediction
                            if padded_sequence is not None and padded_sequence.shape == (FIXED_LENGTH, FEATURES_PER_FRAME):
                                # Reshape for model input (add batch dimension)
                                reshaped_sequence = np.expand_dims(padded_sequence, axis=0)
                                try:
                                    predict_start = time.time()
                                    # Perform prediction
                                    res = model.predict(reshaped_sequence, verbose=0)[0] # verbose=0 for less console output
                                    predict_time = time.time() - predict_start
                                    predictions_made += 1

                                    # Get top N predictions
                                    top_n_indices = np.argsort(res)[-TOP_N:][::-1] # Indices of top N predictions
                                    top_n_confidences = res[top_n_indices] # Confidences of top N
                                    top_n_words = [index_to_word.get(idx, f"Idx_{idx}?") for idx in top_n_indices] # Map indices to words

                                    # Store details of the top prediction
                                    top_pred_idx = top_n_indices[0]
                                    top_pred_conf = top_n_confidences[0]
                                    all_predictions_details.append((top_pred_idx, top_pred_conf))

                                    # Update buffer for smoothed display
                                    prediction_display_buffer.append(top_pred_idx)

                                    # Track max confidence seen in this video
                                    max_confidence_seen_video = max(max_confidence_seen_video, top_pred_conf)

                                    # Prepare text for display
                                    last_top_n_text = [f"{w} ({c:.2f})" for w, c in zip(top_n_words, top_n_confidences)]
                                    # logging.debug(f"[Main] Pred {predictions_made}: Top1={last_top_n_text[0]} (tps: {predict_time:.4f}s)") # Less verbose

                                except tf.errors.InvalidArgumentError as e_tf_shape:
                                     logging.error(f"[Main] Erreur TensorFlow (shape?) pendant predict: {e_tf_shape}. Shape fournie: {reshaped_sequence.shape}")
                                     last_top_n_text = ["Erreur Shape TF"]
                                except Exception as e_pred:
                                    logging.exception(f"[Main] Erreur inconnue lors de model.predict: {e_pred}")
                                    last_top_n_text = ["Erreur Prediction"]
                            else:
                                # This should ideally not happen if padding/logic is correct
                                logging.warning(f"[Main] Shape incorrecte ({padded_sequence.shape if padded_sequence is not None else 'None'}) juste avant prédiction. Attendu: ({FIXED_LENGTH}, {FEATURES_PER_FRAME})")
                                last_top_n_text = ["Erreur Seq Shape"]

                    # --- Display logic ---
                    # Try to get the latest frame from the display queue non-blockingly
                    try:
                        new_frame = extractor.display_queue.get_nowait()
                        if new_frame is not None:
                            frame_display_buffer = new_frame # Update the frame to be displayed
                    except queue.Empty:
                        pass # No new frame available, keep showing the last one

                    # Display the frame if we have one
                    if frame_display_buffer is not None and frame_display_buffer.size > 0:
                        display_frame = frame_display_buffer.copy() # Work on a copy
                        try:
                            # Determine text color based on confidence of the latest top prediction
                            top_conf = 0.0
                            if all_predictions_details:
                                top_conf = all_predictions_details[-1][1] # Confidence of the last prediction made

                            text_color = Colors.CV_RED
                            if top_conf >= CONF_THRESH_GREEN: text_color = Colors.CV_GREEN
                            elif top_conf >= CONF_THRESH_YELLOW: text_color = Colors.CV_YELLOW

                            # Display Top N predictions
                            y_offset = 30
                            for line_idx, line in enumerate(last_top_n_text):
                                current_color = text_color if line_idx == 0 else Colors.CV_WHITE # Highlight top prediction
                                cv2.putText(display_frame, line, (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.7, current_color, 2, cv2.LINE_AA)
                                y_offset += 25

                            # Display Smoothed prediction (most frequent in buffer)
                            if prediction_display_buffer:
                                try:
                                    smoothed_index = Counter(prediction_display_buffer).most_common(1)[0][0]
                                    smoothed_word = index_to_word.get(smoothed_index, "?")
                                    cv2.putText(display_frame, f"Lisse ({SMOOTHING_WINDOW_SIZE}f): {smoothed_word}", (10, display_frame.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, Colors.CV_WHITE, 2, cv2.LINE_AA)
                                except IndexError: # Buffer might be empty initially
                                    pass

                            # Show the frame
                            cv2.imshow(window_name, display_frame)
                            key = cv2.waitKey(1) & 0xFF # Very important waitKey(1) for display

                            # Check for 'q' key press to quit
                            if key == ord('q'):
                                logging.info("Touche 'q' pressée, arrêt global demandé.")
                                processing_active = False
                                extractor.running.clear() # Signal threads immediately
                                raise KeyboardInterrupt("Arrêt utilisateur via 'q'") # Propagate to stop main loop

                        except cv2.error as e_cv:
                            # Handle potential errors if window was closed manually
                            if "NULL window" in str(e_cv) or "Invalid window handle" in str(e_cv):
                                logging.warning(f"[Main] Fenêtre OpenCV '{window_name}' fermée manuellement ou invalide? Arrêt traitement vidéo.")
                                processing_active = False # Stop processing this video
                            else:
                                logging.warning(f"[Main] Erreur cv2 (imshow/putText?): {e_cv}")
                        except Exception as e_show:
                             logging.exception(f"[Main] Erreur inattendue pendant l'affichage/texte: {e_show}")
                             # Consider stopping processing for this video on display errors
                             # processing_active = False

                    # Exit point from the inner while loop if processing_active becomes False
                    if not processing_active:
                         logging.debug(f"[Main] processing_active est False, sortie boucle vidéo interne pour {base_video_name}.")
                         break
                # --- End of inner while processing_active loop ---
                logging.info(f"Fin boucle traitement principale (while processing_active) pour {base_video_name}.")

            except KeyboardInterrupt:
                 logging.info(f"KeyboardInterrupt reçu pendant traitement {base_video_name}. Arrêt...")
                 # Ensure threads are signaled to stop if not already
                 if extractor: extractor.running.clear()
                 raise # Re-raise to be caught by the outer try-except
            except Exception as e_inner_loop:
                logging.exception(f"Erreur inattendue dans la boucle interne (while processing_active) pour {base_video_name}: {e_inner_loop}")
                if extractor: extractor.running.clear() # Signal stop on error
            finally:
                # --- Cleanup AFTER processing each video ---
                logging.info(f"{Colors.BRIGHT_YELLOW}--- Nettoyage pour vidéo {base_video_name}... ---{Colors.RESET}")
                if extractor:
                    extractor.stop() # Ensure threads are properly stopped and joined

                # Try to close the specific OpenCV window for this video
                try:
                     # Check if window still exists before trying to destroy
                     if cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) >= 1:
                         cv2.destroyWindow(window_name)
                         logging.info(f"Fenêtre '{window_name}' fermée.")
                     else:
                         logging.debug(f"Fenêtre '{window_name}' déjà fermée ou non trouvée lors du nettoyage.")
                         cv2.waitKey(1) # Add a small waitkey potentially needed after destroyWindow/if closed manually
                except Exception as e_close:
                    logging.warning(f"Erreur lors de la fermeture de la fenêtre '{window_name}': {e_close}")
                    cv2.waitKey(1) # Still add waitkey

                # Optional: Clear Keras session - Can help with memory leaks but adds overhead
                # Consider doing this less frequently if performance is an issue
                # try:
                #     logging.debug("Appel tf.keras.backend.clear_session()...")
                #     tf.keras.backend.clear_session()
                #     logging.debug("clear_session() terminé.")
                # except Exception as e_clear: logging.warning(f"Erreur lors de clear_session(): {e_clear}")


                video_end_time = time.time()
                processing_time_sec = video_end_time - video_start_time
                logging.info(f"Vidéo {base_video_name}: {frames_processed_main} keypoints traités, {predictions_made} prédictions faites.")
                logging.info(f"Temps total traitement vidéo: {processing_time_sec:.2f} sec.")

                # --- Sauvegarde Keypoints (.npy) ---
                if SAVE_KEYPOINTS and all_keypoints_for_video: # Reading global SAVE_KEYPOINTS
                    npy_filename = os.path.splitext(base_video_name)[0] + ".npy"
                    npy_filepath = os.path.join(KEYPOINTS_SAVE_DIR, npy_filename)
                    try:
                        np.save(npy_filepath, np.array(all_keypoints_for_video))
                        logging.info(f"Keypoints sauvegardés: {npy_filepath} ({len(all_keypoints_for_video)} frames)")
                    except Exception as e_save:
                        logging.error(f"Erreur lors de la sauvegarde des keypoints {npy_filepath}: {e_save}")

                # --- Analyse Finale des Prédictions & Log CSV & Suivi Précision ---
                final_word = "N/A"; final_word_freq = 0; avg_conf_final_word = 0.0
                is_correct = False # Track if the final prediction matches expected

                # Extraire le mot attendu du nom de fichier (normalisé en minuscules)
                expected_word = get_expected_word_from_filename(base_video_name)
                # logging.debug(f"Fichier: '{base_video_name}', Mot Attendu Normalisé: '[{expected_word}]'")

                if not expected_word:
                     logging.warning(f"Impossible d'extraire un mot attendu valide de '{base_video_name}'")
                else:
                     # Incrémenter le compteur total pour ce mot attendu (clé en minuscules)
                     # Do this even if deadlock occurs, as an attempt was made
                     word_counts[expected_word] = word_counts.get(expected_word, 0) + 1

                # --- Vérifier si le timeout de deadlock a eu lieu ---
                if deadlock_timeout_occurred:
                     final_word = "TIMEOUT_DEADLOCK"
                     final_word_freq = 0 # No prediction analysis possible
                     avg_conf_final_word = 0.0 # No confidence
                     # predictions_made and max_confidence_seen_video keep their values from before the timeout
                     logging.warning(f"Enregistrement CSV : {Colors.RED}{final_word}{Colors.RESET} pour {base_video_name} en raison du timeout de deadlock.")
                     # 'is_correct' remains False

                # --- Si pas de deadlock, analyser les prédictions accumulées ---
                elif all_predictions_details:
                    try:
                        # Extract just the predicted indices
                        prediction_indices = [idx for idx, conf in all_predictions_details]
                        if prediction_indices:
                            # Find the most frequent prediction index
                            index_counts = Counter(prediction_indices)
                            most_common_index, final_word_freq = index_counts.most_common(1)[0]
                            final_word = index_to_word.get(most_common_index, f"Idx_{most_common_index}?") # Convert index to word

                            # Calculate average confidence for the most frequent word
                            confidences_for_final_word = [conf for idx, conf in all_predictions_details if idx == most_common_index]
                            if confidences_for_final_word:
                                avg_conf_final_word = sum(confidences_for_final_word) / len(confidences_for_final_word)

                            # --- Comparaison pour la précision (insensible à la casse) ---
                            if expected_word and final_word.lower() == expected_word: # expected_word is already lowercase
                                is_correct = True
                                # Incrémenter le compteur correct SEULEMENT si la prédiction est bonne
                                correct_predictions[expected_word] = correct_predictions.get(expected_word, 0) + 1
                                logging.info(f"-> Mot final: {Colors.BRIGHT_GREEN}{final_word}{Colors.RESET} ({final_word_freq}/{predictions_made} fois, conf avg: {avg_conf_final_word:.2f}) - CORRECT")
                            else:
                                # Log as incorrect if expected word exists but doesn't match, OR if no expected word was found

                                # --- CORRECTED CODE ---
                                # Select the appropriate logging function
                                logger_func = logging.info if expected_word else logging.warning
                                # Call the selected function with the message
                                logger_func(f"-> Mot final: {Colors.RED}{final_word}{Colors.RESET} ({final_word_freq}/{predictions_made} fois, conf avg: {avg_conf_final_word:.2f}) - INCORRECT (Attendu: {expected_word if expected_word else 'N/A'})")
                                # --- END CORRECTION ---

                        else:
                            # all_predictions_details existed but was empty? Should not happen if filled correctly.
                            logging.warning(f"{Colors.RED}-> 'all_predictions_details' non vide mais sans indices valides pour {base_video_name}?{Colors.RESET}")
                            final_word = "Erreur_Analyse_Indices"

                    except Exception as e_analyze:
                        logging.exception(f"Erreur lors de l'analyse finale des prédictions pour {base_video_name}: {e_analyze}")
                        final_word = "Erreur_Analyse_Exception"
                # --- Si pas de deadlock ET aucune prédiction n'a été faite du tout ---
                else:
                    logging.warning(f"{Colors.RED}-> Aucune prédiction générée pour {base_video_name} (pas de deadlock). 'final_word' reste '{final_word}'.{Colors.RESET}")
                    # final_word remains "N/A" as initialized

                # --- Écriture dans le fichier CSV ---
                try:
                    current_timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
                    with open(PREDICTION_CSV_FILE, 'a', newline='', encoding='utf-8') as csvfile:
                        writer = csv.writer(csvfile)
                        # Write the data row
                        writer.writerow([
                            current_timestamp,
                            base_video_name, # Use base name for CSV
                            final_word,
                            final_word_freq,
                            predictions_made,
                            f"{avg_conf_final_word:.4f}", # Format float
                            f"{max_confidence_seen_video:.4f}", # Format float
                            f"{processing_time_sec:.2f}" # Format float
                        ])
                    logging.info(f"Résultat '{final_word}' ajouté à {PREDICTION_CSV_FILE}")
                except IOError as e_io_csv:
                    logging.error(f"Impossible d'écrire dans le fichier CSV {PREDICTION_CSV_FILE}: {e_io_csv}")
                except Exception as e_csv:
                    logging.exception(f"Erreur inattendue lors de l'écriture CSV: {e_csv}")

                logging.info(f"{Colors.BRIGHT_YELLOW}--- Fin traitement complet pour {base_video_name} ---{Colors.RESET}")
                # Optional short pause between videos
                # time.sleep(0.5)

        # --- Fin de la boucle principale sur toutes les vidéos ---
        total_main_time = time.time() - main_start_time
        logging.info(f"{Colors.BRIGHT_GREEN}=== Traitement des {len(video_files_to_process)} vidéos terminé en {total_main_time:.2f} secondes. ===")

        # --- Calcul et Affichage de la Précision Globale et par Mot ---
        total_videos_processed_for_accuracy = sum(word_counts.values()) # Total attempts where expected word was identified
        total_correct_overall = sum(correct_predictions.values()) # Total correct predictions

        if total_videos_processed_for_accuracy > 0:
            overall_accuracy = (total_correct_overall / total_videos_processed_for_accuracy) * 100
            logging.info(f"=== Précision Globale: {total_correct_overall}/{total_videos_processed_for_accuracy} ({overall_accuracy:.2f}%) ===")
        else:
            logging.info("=== Aucune vidéo avec mot attendu identifiable n'a été traitée pour calculer la précision globale. ===")

        word_accuracies = {}
        logging.info("--- Précision par Mot ---")
        sorted_expected_words = sorted(word_counts.keys()) # Sort words alphabetically for report
        if not sorted_expected_words:
             logging.info("Aucun mot attendu n'a été extrait des noms de fichiers.")
        else:
            for word in sorted_expected_words:
                total = word_counts[word]
                correct = correct_predictions.get(word, 0) # Get correct count, default to 0 if word was never predicted correctly
                accuracy = (correct / total) * 100 if total > 0 else 0
                word_accuracies[word] = accuracy
                logging.info(f"- {word}: {correct}/{total} ({accuracy:.2f}%)")
        logging.info("------------------------")

        # --- Génération du Graphique ---
        if word_accuracies:
            try:
                words = list(word_accuracies.keys())
                accuracies = list(word_accuracies.values())

                plt.figure(figsize=(max(10, len(words) * 0.8), 6)) # Adjust width based on number of words
                bars = plt.bar(words, accuracies, color='skyblue')
                plt.xlabel("Mot Attendu (normalisé)")
                plt.ylabel("Précision (%)")
                plt.title("Précision de la Prédiction par Mot")
                plt.ylim(0, 105) # Y-axis from 0 to 105%
                plt.xticks(rotation=45, ha='right') # Rotate labels for better readability

                # Add percentage labels on top of bars
                for bar in bars:
                    yval = bar.get_height()
                    plt.text(bar.get_x() + bar.get_width()/2.0, yval + 1, f'{yval:.1f}%', va='bottom', ha='center', fontsize=9)

                plt.tight_layout() # Adjust layout to prevent labels overlapping
                graph_filename = "prediction_accuracy_per_word.png"
                plt.savefig(graph_filename) # Sauvegarde le graphique
                logging.info(f"Graphique de précision sauvegardé dans '{graph_filename}'")
                plt.show() # <<<<<<< DÉCOMMENTEZ CETTE LIGNE POUR AFFICHER LE GRAPHIQUE
            except Exception as e_plot:
                logging.error(f"Erreur lors de la génération ou sauvegarde du graphique: {e_plot}")
        else:
            logging.info("Aucune donnée de précision par mot disponible pour générer le graphique.")

    except KeyboardInterrupt:
         logging.info(f"{Colors.RED}Arrêt programme demandé par utilisateur (KeyboardInterrupt).{Colors.RESET}")
         if extractor is not None: # Check if extractor exists
             logging.info("Tentative d'arrêt propre des threads suite à KeyboardInterrupt...")
             extractor.stop()
    except Exception as e_main_loop:
         logging.exception(f"{Colors.RED}Erreur inattendue et non gérée dans la boucle principale des vidéos: {e_main_loop}{Colors.RESET}")
         if extractor is not None: # Check if extractor exists
             logging.info("Tentative d'arrêt propre des threads suite à une erreur majeure...")
             extractor.stop()
    finally:
         # --- Nettoyage final global ---
         logging.info("Nettoyage final : Fermeture des fenêtres OpenCV restantes...")
         cv2.destroyAllWindows()
         # Add multiple waitKeys just in case
         for _ in range(5): cv2.waitKey(1)

         # Clear Keras session finally if model was loaded
         if 'model' in locals() and model is not None:
             try:
                 logging.debug("Libération finale de la session Keras globale...")
                 tf.keras.backend.clear_session()
                 del model # Explicitly delete model reference
                 logging.debug("Session Keras globale libérée.")
             except Exception as e_final_clear:
                  logging.warning(f"Erreur lors du nettoyage final de la session Keras: {e_final_clear}")

         time.sleep(0.5) # Small pause before exiting
         logging.info(f"{Colors.BRIGHT_GREEN}Programme terminé.{Colors.RESET}")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logging.exception(f"{Colors.RED}Erreur non gérée au niveau __main__: {e}{Colors.RESET}")
    finally:
        # Ensure all windows are closed again, just in case
        cv2.destroyAllWindows()
        for _ in range(5): cv2.waitKey(1)
        logging.info("Sortie finale du script.")