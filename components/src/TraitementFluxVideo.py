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
from collections import deque, Counter

# --- Configuration du logging ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - [%(threadName)s] - %(message)s')

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

# --- Constants & Configuration ---
# !!! IMPORTANT: Assurez-vous que MODEL_PATH pointe vers le modèle
#     entraîné avec les features incluant la bouche !!!
MODEL_PATH = "models/model.h5"
VOCABULARY_FILE = "vocabulaire.txt"
FIXED_LENGTH = 46 # Doit correspondre à l'entraînement
CAMERA_INDEX = 0    # Index de la caméra (0 pour la webcam par défaut)

# Configuration de l'affichage et de la prédiction
TOP_N = 3                   # Afficher les N meilleures prédictions
SMOOTHING_WINDOW_SIZE = 15  # Nombre de prédictions pour le lissage
CONF_THRESH_GREEN = 0.80    # Seuil de confiance pour afficher en vert
CONF_THRESH_YELLOW = 0.50   # Seuil de confiance pour afficher en jaune (sinon rouge)
FRAMES_TO_SKIP = 2          # Traiter 1 frame sur N (1 = tout traiter, 2 = 1/2, 3 = 1/3...)
MAX_FRAME_WIDTH = 1024      # Optionnel: Redimensionner la largeur max de la frame pour performance

# --- Paramètres d'Extraction (DOIVENT CORRESPONDRE à l'entraînement) ---
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

# ---> FEATURES_PER_FRAME (DOIT CORRESPONDRE à l'entraînement) <---
FEATURES_PER_FRAME = (NUM_POSE_KEYPOINTS * NUM_COORDS) + \
                     (NUM_HAND_KEYPOINTS * NUM_COORDS) * 2 + \
                     (NUM_MOUTH_KEYPOINTS * NUM_COORDS) # <-- AJOUT

# --- Logging des configurations ---
logging.info("--- Configuration Traitement Flux Vidéo (avec Bouche) ---")
logging.info(f"MODEL_PATH: {MODEL_PATH} (Doit être entraîné avec bouche!)")
logging.info(f"VOCABULARY_FILE: {VOCABULARY_FILE}")
logging.info(f"FIXED_LENGTH: {FIXED_LENGTH}")
logging.info(f"CAMERA_INDEX: {CAMERA_INDEX}")
logging.info(f"TOP_N Predictions: {TOP_N}")
logging.info(f"SMOOTHING_WINDOW_SIZE: {SMOOTHING_WINDOW_SIZE}")
logging.info(f"FRAMES_TO_SKIP: {FRAMES_TO_SKIP}")
if MAX_FRAME_WIDTH: logging.info(f"MAX_FRAME_WIDTH: {MAX_FRAME_WIDTH}")
else: logging.info("MAX_FRAME_WIDTH: Désactivé")
logging.info(f"Nombre points Pose: {NUM_POSE_KEYPOINTS}, Main: {NUM_HAND_KEYPOINTS}, Bouche: {NUM_MOUTH_KEYPOINTS}")
logging.info(f"FEATURES_PER_FRAME attendu: {FEATURES_PER_FRAME}")
logging.info("---------------------------------------------------------")

# --- Utility Functions ---
def load_vocabulary(filepath):
    """Charge le vocabulaire depuis un fichier texte (mot:index)."""
    vocabulaire = {}
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split(":")
                if len(parts) == 2 and parts[0] and parts[1].isdigit():
                    vocabulaire[parts[0]] = int(parts[1]) # Clé = mot, Valeur = index
                elif line.strip():
                    logging.warning(f"Format ligne incorrect vocabulaire: '{line.strip()}'")
    except (FileNotFoundError, ValueError) as e:
        logging.error(f"Erreur chargement vocabulaire {filepath}: {e}")
        return {}
    if not vocabulaire:
         logging.warning(f"Vocabulaire chargé depuis {filepath} est vide.")
    else:
         logging.info(f"Vocabulaire chargé ({len(vocabulaire)} mots) depuis {filepath}")
    return vocabulaire

def extract_keypoints(results):
    """
    Extracts POSE(4) + HANDS(2x21) + MOUTH(N) keypoints.
    Identique à la version dans traitementVideo.py et CaptureVideo.py.
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
    mouth = np.zeros(NUM_MOUTH_KEYPOINTS * NUM_COORDS)
    if results.face_landmarks:
        try:
            if all(idx < len(results.face_landmarks.landmark) for idx in MOUTH_LANDMARK_INDICES):
                 mouth_points = [results.face_landmarks.landmark[i] for i in MOUTH_LANDMARK_INDICES]
                 mouth = np.array([[res.x, res.y, res.z] for res in mouth_points]).flatten()
            else:
                 logging.debug(f"Indices bouche manquants (face_landmarks len={len(results.face_landmarks.landmark)}). Zéros pour bouche.") # Debug au lieu de Warning pour moins de spam
        except Exception as e:
            logging.error(f"Erreur extraction points bouche: {e}. Zéros pour bouche.")
            mouth = np.zeros(NUM_MOUTH_KEYPOINTS * NUM_COORDS)
    # Concaténation
    extracted = np.concatenate([pose, lh, rh, mouth])
    # Vérification finale
    if extracted.shape[0] != FEATURES_PER_FRAME:
        logging.warning(f"Extraction a produit {extracted.shape[0]} features, attendu {FEATURES_PER_FRAME}. Retour zéros.")
        return np.zeros(FEATURES_PER_FRAME)
    return extracted

# --- Keypoint Extractor Class (Adaptée pour flux continu) ---
class KeypointExtractor:
    def __init__(self):
        self.mp_holistic = mp.solutions.holistic
        self.frame_queue = queue.Queue(maxsize=30) # File pour frames vers l'extraction
        self.keypoint_queue = queue.Queue(maxsize=60) # File pour keypoints vers le thread principal
        self.display_queue = queue.Queue(maxsize=5) # File pour frames vers l'affichage
        self.running = threading.Event()           # Événement pour contrôler les threads
        self.capture_thread = None
        self.extraction_thread = None
        self.camera_index = 0                      # Index de la caméra
        self.holistic_instance = None              # Instance Holistic pour le thread d'extraction

    def capture_frames(self):
        """Thread fonction pour capturer les frames de la caméra."""
        frame_count_read = 0
        frame_count_queued_extract = 0
        frame_count_queued_display = 0
        capture_start_time = time.time()
        cap = None

        try:
            logging.info(f"{Colors.GREEN}>>> DÉMARRAGE THREAD CAPTURE (Caméra Index: {self.camera_index}){Colors.RESET}")
            cap = cv2.VideoCapture(self.camera_index)
            if not cap.isOpened():
                logging.error(f"Impossible d'ouvrir la caméra (Index: {self.camera_index})")
                self.running.clear() # Signaler l'arrêt aux autres threads
                return # Sortir du thread si la caméra ne s'ouvre pas

            # Obtenir les propriétés de la caméra (optionnel, pour info)
            frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = cap.get(cv2.CAP_PROP_FPS) # Peut être 0 ou incorrect pour certaines webcams
            logging.info(f"Caméra ouverte: {frame_width}x{frame_height} @ ~{fps:.2f} FPS")

            # Calcul du redimensionnement si nécessaire
            target_width = frame_width; target_height = frame_height; resize_needed = False
            if MAX_FRAME_WIDTH and frame_width > MAX_FRAME_WIDTH:
                scale = MAX_FRAME_WIDTH / frame_width
                target_width = MAX_FRAME_WIDTH
                target_height = int(frame_height * scale)
                resize_needed = True
                logging.info(f"Redimensionnement activé: {frame_width}x{frame_height} -> {target_width}x{target_height}")

            while self.running.is_set():
                ret, frame = cap.read()
                if not ret:
                    logging.warning("Impossible de lire la frame de la caméra. Nouvelle tentative...")
                    time.sleep(0.1) # Petite pause avant de réessayer
                    # Vérifier si la caméra est toujours ouverte
                    if not cap.isOpened():
                         logging.error("La caméra semble s'être déconnectée. Arrêt capture.")
                         self.running.clear()
                         break
                    continue # Revenir au début de la boucle while

                frame_count_read += 1

                # Logique pour sauter des frames
                if (frame_count_read - 1) % FRAMES_TO_SKIP == 0:
                    frame_to_process = frame
                    if resize_needed:
                        try:
                             frame_to_process = cv2.resize(frame, (target_width, target_height), interpolation=cv2.INTER_AREA)
                        except Exception as resize_err:
                             logging.warning(f"[Capture] Erreur redim. frame {frame_count_read}: {resize_err}")
                             frame_to_process = frame # Utiliser l'original si erreur

                    if frame_to_process is not None:
                        # Mettre frame dans la queue d'extraction (avec timeout)
                        try:
                            self.frame_queue.put(frame_to_process, timeout=0.5) # Timeout court
                            frame_count_queued_extract += 1
                        except queue.Full:
                             # Si plein, on pourrait dropper la frame ou attendre un peu
                             logging.debug("[Capture] Queue frames (extract) pleine. Frame droppée.")
                             pass # On perd une frame, évite le blocage
                        except Exception as e_put_extract:
                             logging.exception(f"[Capture] Erreur frame_queue.put : {e_put_extract}")
                             self.running.clear(); break

                        # Mettre frame dans la queue d'affichage (remplacer si plein)
                        try:
                            self.display_queue.put_nowait(frame_to_process)
                            frame_count_queued_display += 1
                        except queue.Full:
                            try:
                                self.display_queue.get_nowait() # Enlever la plus vieille
                                self.display_queue.put_nowait(frame_to_process) # Ajouter la nouvelle
                            except queue.Empty: pass
                            except Exception as e_replace: logging.warning(f"[Capture] Erreur remplacement display queue: {e_replace}")
                        except Exception as e_put_display:
                             logging.warning(f"[Capture] Erreur display_queue.put : {e_put_display}")
                    # else: # frame_to_process est None après redim? Ignorer
                    #    logging.warning(f"[Capture] frame_to_process est None après redim.?")

                # Petite pause pour éviter de saturer le CPU si la caméra est très rapide
                # et que le traitement ne suit pas. Ajuster si nécessaire.
                # time.sleep(0.001)

            logging.info(f"[Capture] Boucle terminée (running={self.running.is_set()}).")

        except Exception as e_globale_capture:
            logging.exception(f"{Colors.RED}!!! ERREUR GLOBALE dans capture_frames !!! : {e_globale_capture}{Colors.RESET}")
            self.running.clear() # Signaler l'arrêt
        finally:
            logging.debug(f"{Colors.RED}>>> Entrée FINALLY capture_frames{Colors.RESET}")
            if cap and cap.isOpened():
                cap.release()
                logging.info("Capture vidéo (caméra) relâchée.")
            # Toujours envoyer le signal de fin à l'extracteur
            try:
                self.frame_queue.put(None, timeout=1.0) # Signal de fin pour l'extracteur
                logging.info("[Capture - Finally] Signal fin (None) envoyé à frame_queue.")
            except queue.Full:
                logging.error("[Capture - Finally] Échec envoi signal fin (None) - Queue frames pleine. L'extracteur pourrait attendre.")
            except Exception as e_final:
                logging.error(f"[Capture - Finally] Erreur envoi signal fin (None) : {e_final}")

            total_time = time.time() - capture_start_time
            logging.info(f"Thread capture terminé. {frame_count_queued_extract}/{frame_count_read} frames vers extraction en {total_time:.2f}s.")
            logging.info(f"{Colors.RED}### FIN THREAD CAPTURE ###{Colors.RESET}")

    def extract_keypoints_loop(self):
        """Thread fonction pour extraire les keypoints en continu."""
        logging.info(f"{Colors.GREEN}>>> DÉMARRAGE THREAD EXTRACTION{Colors.RESET}")
        frames_processed = 0
        extraction_start_time = time.time()
        # Créer l'instance Holistic DANS ce thread
        self.holistic_instance = self.mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5)
        if not self.holistic_instance:
             logging.error("Impossible d'initialiser MediaPipe Holistic. Arrêt thread extraction.")
             self.running.clear()
             # Envoyer signal fin à keypoint_queue même si erreur init
             try: self.keypoint_queue.put(None, timeout=1.0)
             except: pass
             return

        logging.info("[Extraction] Instance MediaPipe Holistic créée.")
        try:
            while self.running.is_set(): # Vérifier le flag avant chaque itération
                frame = None
                try:
                    # Attendre une frame de la queue (avec timeout)
                    frame = self.frame_queue.get(timeout=1.0)
                    if frame is None:
                        logging.info("[Extraction] Signal fin (None) reçu de frame_queue. Arrêt normal.")
                        break # Sortir de la boucle while
                except queue.Empty:
                    # Timeout - vérifier si on doit continuer ou si le thread capture est mort
                    if not self.running.is_set(): # Vérifier à nouveau le flag au cas où il aurait changé pendant le timeout
                         logging.info("[Extraction] Arrêt demandé pendant attente frame.")
                         break
                    # Si le thread capture n'est plus vivant, on arrête aussi
                    if self.capture_thread and not self.capture_thread.is_alive():
                        logging.warning("[Extraction] Queue frames vide et thread capture semble mort. Arrêt extraction.")
                        self.running.clear() # Assurer que tout s'arrête
                        break
                    continue # Sinon, on continue d'attendre

                # Traiter la frame avec MediaPipe
                try:
                    # Convertir BGR en RGB
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    # Marquer comme non modifiable pour optimisation
                    frame_rgb.flags.writeable = False
                    # Processus MediaPipe
                    results = self.holistic_instance.process(frame_rgb)
                    # Marquer comme modifiable à nouveau (bonne pratique)
                    # frame_rgb.flags.writeable = True # Pas nécessaire ici

                    # Extraire les keypoints (utilise la fonction partagée avec bouche)
                    keypoints = extract_keypoints(results)
                    frames_processed += 1

                    # Mettre les keypoints dans la queue pour le thread principal
                    try:
                        self.keypoint_queue.put(keypoints, timeout=0.5) # Timeout court
                    except queue.Full:
                         logging.debug("[Extraction] Queue keypoints pleine. Keypoints droppés.")
                         pass # Perdre les keypoints si la queue est pleine
                    except Exception as e_put_kp:
                        logging.exception(f"[Extraction] Erreur keypoint_queue.put : {e_put_kp}")
                        self.running.clear(); break # Arrêter si erreur critique

                except Exception as e_process:
                    logging.exception(f"[Extraction] Erreur pendant traitement MediaPipe/extraction: {e_process}")
                    # On continue avec la frame suivante si possible
                    pass

            logging.info(f"[Extraction] Boucle terminée (running={self.running.is_set()}).")

        except Exception as e_init_loop:
             logging.exception(f"[Extraction] Erreur majeure dans la boucle d'extraction: {e_init_loop}")
             self.running.clear()
        finally:
             logging.debug(f"{Colors.RED}>>> Entrée FINALLY extract_keypoints_loop{Colors.RESET}")
             # Fermer l'instance Holistic
             if self.holistic_instance:
                 try: self.holistic_instance.close(); logging.debug("[Extraction - Finally] Instance Holistic fermée.")
                 except Exception as e_close: logging.warning(f"[Extraction - Finally] Erreur fermeture Holistic: {e_close}")
             # Toujours envoyer le signal de fin au thread principal
             try:
                 self.keypoint_queue.put(None, timeout=1.0)
                 logging.info("[Extraction - Finally] Signal fin (None) envoyé à keypoint_queue.")
             except queue.Full:
                 logging.error("[Extraction - Finally] Échec envoi signal fin (None) keypoints - Queue pleine.")
             except Exception as e_final_kp:
                 logging.error(f"[Extraction - Finally] Erreur envoi signal fin (None) keypoints : {e_final_kp}")

             total_time = time.time() - extraction_start_time
             logging.info(f"Thread extraction terminé. Traité {frames_processed} frames en {total_time:.2f}s.")
             logging.info(f"{Colors.RED}### FIN THREAD EXTRACTION ###{Colors.RESET}")

    def start(self, camera_index=0):
        """Démarre les threads de capture et d'extraction."""
        if self.capture_thread and self.capture_thread.is_alive() or \
           self.extraction_thread and self.extraction_thread.is_alive():
            logging.warning("Tentative de démarrer alors que des threads sont déjà actifs. Appel stop() d'abord.")
            self.stop()

        self.camera_index = camera_index
        self.running.set() # Indiquer que les threads doivent tourner

        # Vider les queues avant de démarrer
        for q in [self.frame_queue, self.keypoint_queue, self.display_queue]:
            while not q.empty():
                try: q.get_nowait()
                except queue.Empty: break
        logging.debug("Queues vidées avant démarrage.")

        # Créer et démarrer les threads
        self.extraction_thread = threading.Thread(target=self.extract_keypoints_loop, name="ExtractionThread")
        self.capture_thread = threading.Thread(target=self.capture_frames, name="CaptureThread")

        self.extraction_thread.start()
        self.capture_thread.start()
        logging.info(f"Threads Capture et Extraction démarrés (Caméra: {self.camera_index}).")

    def stop(self):
        """Signale aux threads de s'arrêter et attend leur terminaison."""
        logging.info(f"{Colors.BRIGHT_YELLOW}>>> Demande d'arrêt des threads...{Colors.RESET}")
        self.running.clear() # Signaler aux boucles while de s'arrêter

        join_timeout = 5 # Timeout pour attendre chaque thread (secondes)

        # Attendre la fin du thread d'extraction
        if self.extraction_thread and self.extraction_thread.is_alive():
            logging.debug(f"Attente fin thread extraction (max {join_timeout}s)...")
            self.extraction_thread.join(timeout=join_timeout)
            if self.extraction_thread.is_alive():
                 logging.warning(f"{Colors.RED}TIMEOUT!{Colors.RESET} Thread extraction non terminé.")
            else: logging.info("Thread extraction terminé.")
        # else: logging.debug("Thread extraction non trouvé ou déjà arrêté.")

        # Attendre la fin du thread de capture
        if self.capture_thread and self.capture_thread.is_alive():
            logging.debug(f"Attente fin thread capture (max {join_timeout}s)...")
            self.capture_thread.join(timeout=join_timeout)
            if self.capture_thread.is_alive():
                 logging.warning(f"{Colors.RED}TIMEOUT!{Colors.RESET} Thread capture non terminé.")
            else: logging.info("Thread capture terminé.")
        # else: logging.debug("Thread capture non trouvé ou déjà arrêté.")

        self.capture_thread = None
        self.extraction_thread = None
        logging.info(f"{Colors.BRIGHT_YELLOW}<<< Vérification arrêt threads terminée.{Colors.RESET}")

# --- Fonction Principale ---
def main():
    # --- Initialisation TF et Modèle ---
    model = None
    try:
        # Configuration GPU (optionnel mais recommandé si GPU disponible)
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            try:
                for gpu in gpus: tf.config.experimental.set_memory_growth(gpu, True)
                logging.info(f"GPU(s) configuré(s) pour Memory Growth : {gpus}")
            except RuntimeError as e: logging.error(f"Erreur config GPU Memory Growth: {e}")
        else: logging.warning("Aucun GPU détecté par TensorFlow.")

        # Charger le modèle Keras (doit être entraîné avec les bonnes features)
        if not os.path.exists(MODEL_PATH):
             logging.error(f"Fichier modèle non trouvé : {MODEL_PATH}"); return
        model = tf.keras.models.load_model(MODEL_PATH)
        logging.info(f"Modèle chargé depuis {MODEL_PATH}")
        # Vérifier la compatibilité de la shape d'entrée
        try:
            expected_shape_model = model.input_shape # (None, FIXED_LENGTH, FEATURES_PER_FRAME)
            script_shape = (FIXED_LENGTH, FEATURES_PER_FRAME)
            logging.info(f"Shape entrée attendue par modèle: {expected_shape_model}")
            logging.info(f"Shape entrée configurée script: (Batch, {script_shape[0]}, {script_shape[1]})")
            if len(expected_shape_model) != 3 or \
               expected_shape_model[1] is not None and expected_shape_model[1] != script_shape[0] or \
               expected_shape_model[2] is not None and expected_shape_model[2] != script_shape[1]:
                 logging.error(f"{Colors.RED}INCOMPATIBILITÉ DE SHAPE! Le modèle attend {expected_shape_model[2]} features, "
                               f"mais le script est configuré pour {script_shape[1]} features.{Colors.RESET}")
                 logging.error("Assurez-vous que le modèle chargé a été entraîné avec les MÊMES paramètres (points bouche inclus?). Arrêt.")
                 return # Arrêter si le modèle est incompatible
            else:
                 logging.info("Shape modèle compatible avec la configuration du script.")
        except Exception as e:
             logging.warning(f"Impossible vérifier/valider la shape d'entrée modèle: {e}")

    except Exception as e:
        logging.exception(f"Erreur majeure init TensorFlow/Modèle : {e}"); return

    # --- Chargement Vocabulaire ---
    vocabulaire = load_vocabulary(VOCABULARY_FILE)
    if not vocabulaire: logging.error("Vocabulaire vide/non chargé. Arrêt."); return
    # Créer mapping inverse (index -> mot) pour afficher les prédictions
    index_to_word = {i: word for word, i in vocabulaire.items()}
    logging.info(f"Prêt à reconnaître {len(vocabulaire)} mots: {list(vocabulaire.keys())}")

    # --- Initialisation de l'extracteur et démarrage des threads ---
    extractor = KeypointExtractor()
    extractor.start(camera_index=CAMERA_INDEX)

    # --- Structures pour la prédiction et l'affichage ---
    sequence_window = deque(maxlen=FIXED_LENGTH)         # Fenêtre glissante de keypoints
    prediction_display_buffer = deque(maxlen=SMOOTHING_WINDOW_SIZE) # Buffer pour lisser la prédiction affichée
    last_top_n_text = ["Initialisation..."]              # Texte à afficher pour Top-N
    smoothed_word = ""                                   # Mot lissé à afficher
    current_frame = None                                 # Dernière frame reçue pour affichage

    processing_active = True                             # Flag pour contrôler la boucle principale
    window_name = "Traitement Flux Video Temps Reel - Appuyez sur 'q' pour quitter"
    cv2.namedWindow(window_name)

    logging.info("Démarrage de la boucle principale de traitement...")
    try:
        # --- Boucle Principale (Traitement en temps réel) ---
        while processing_active:
            frame_to_display = None
            keypoints = None

            # 1. Récupérer les keypoints extraits (non bloquant)
            try:
                keypoints = extractor.keypoint_queue.get_nowait()
                if keypoints is None: # Signal de fin reçu de l'extracteur
                    logging.info("[Main] Signal fin (None) reçu de keypoint_queue. Arrêt.")
                    processing_active = False
                    break # Sortir de la boucle principale
                else:
                    # Ajouter les keypoints valides à la fenêtre glissante
                    sequence_window.append(keypoints)
            except queue.Empty:
                # Pas de nouveaux keypoints, on continue (la prédiction utilisera la fenêtre actuelle)
                pass
            except Exception as e_get_kp:
                 logging.exception(f"Erreur récupération keypoints: {e_get_kp}")
                 # Peut-être arrêter si l'erreur persiste ?
                 # processing_active = False; break

            # 2. Logique de Prédiction (si on a assez de données)
            current_sequence_len = len(sequence_window)
            top_pred_conf = 0.0 # Confiance de la meilleure prédiction de cette itération

            if current_sequence_len > 0: # Besoin d'au moins une frame pour commencer le padding
                padded_sequence = None
                if current_sequence_len < FIXED_LENGTH:
                    # Padder au début avec des zéros si pas assez de frames
                    padding = np.zeros((FIXED_LENGTH - current_sequence_len, FEATURES_PER_FRAME))
                    padded_sequence = np.concatenate((padding, np.array(sequence_window)), axis=0)
                else: # Assez de frames, prendre la fenêtre actuelle
                    padded_sequence = np.array(sequence_window)

                # Vérifier la shape avant de prédire
                if padded_sequence is not None and padded_sequence.shape == (FIXED_LENGTH, FEATURES_PER_FRAME):
                    reshaped_sequence = np.expand_dims(padded_sequence, axis=0) # Ajouter dimension Batch
                    try:
                        # Prédiction avec le modèle
                        res = model.predict(reshaped_sequence, verbose=0)[0] # verbose=0 pour moins de logs TF

                        # Obtenir Top N prédictions
                        top_n_indices = np.argsort(res)[-TOP_N:][::-1]
                        top_n_confidences = res[top_n_indices]
                        top_n_words = [index_to_word.get(idx, f"Idx_{idx}?") for idx in top_n_indices]

                        # Mettre à jour le buffer de lissage avec l'index de la meilleure prédiction
                        prediction_display_buffer.append(top_n_indices[0])

                        # Préparer le texte pour l'affichage Top-N
                        last_top_n_text = [f"{w} ({c*100:.1f}%)" for w, c in zip(top_n_words, top_n_confidences)]

                        # Garder la confiance de la meilleure prédiction (pour la couleur)
                        top_pred_conf = top_n_confidences[0]

                    except tf.errors.InvalidArgumentError as e_tf_shape:
                         logging.error(f"[Main] {Colors.RED}Erreur Shape TF pendant predict: {e_tf_shape}. Modèle incompatible?{Colors.RESET}")
                         last_top_n_text = ["Erreur Shape Modèle"]
                         # Optionnel: arrêter si l'erreur persiste
                         # processing_active = False; break
                    except Exception as e_pred:
                        logging.exception(f"[Main] Erreur inconnue model.predict: {e_pred}")
                        last_top_n_text = ["Erreur Prediction"]
                # else: # Ne devrait pas arriver si logique correcte
                #    logging.warning(f"[Main] Shape incorrecte ({padded_sequence.shape}) avant prédiction.")
                #    last_top_n_text = ["Erreur Seq Shape"]

            # 3. Calculer la prédiction lissée
            if prediction_display_buffer:
                try:
                    # Trouver l'index le plus fréquent dans le buffer
                    smoothed_index = Counter(prediction_display_buffer).most_common(1)[0][0]
                    smoothed_word = index_to_word.get(smoothed_index, "?")
                except IndexError: # Buffer vide au début
                    smoothed_word = ""

            # 4. Récupérer la dernière frame à afficher (non bloquant)
            try:
                new_frame = extractor.display_queue.get_nowait()
                if new_frame is not None:
                    current_frame = new_frame # Mettre à jour la frame à afficher
            except queue.Empty:
                # Pas de nouvelle frame, on garde la précédente (ou None si début)
                pass
            except Exception as e_get_disp:
                 logging.warning(f"Erreur get display_queue: {e_get_disp}")


            # 5. Affichage sur la frame (si une frame est disponible)
            if current_frame is not None and current_frame.size > 0:
                frame_to_display = current_frame.copy() # Travailler sur une copie

                # Déterminer couleur basée sur la confiance de la prédiction actuelle (non lissée)
                text_color = Colors.CV_RED
                if top_pred_conf >= CONF_THRESH_GREEN: text_color = Colors.CV_GREEN
                elif top_pred_conf >= CONF_THRESH_YELLOW: text_color = Colors.CV_YELLOW

                # Afficher Top N prédictions
                y_offset = 30
                cv2.putText(frame_to_display, "Predictions:", (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, Colors.CV_WHITE, 2, cv2.LINE_AA)
                y_offset += 25
                for line_idx, line in enumerate(last_top_n_text):
                    current_color = text_color if line_idx == 0 else Colors.CV_WHITE # Couleur spéciale pour Top 1
                    cv2.putText(frame_to_display, line, (15, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.7, current_color, 2, cv2.LINE_AA)
                    y_offset += 30

                # Afficher la prédiction lissée en bas
                cv2.putText(frame_to_display, f"Lisse ({SMOOTHING_WINDOW_SIZE}f): {smoothed_word}",
                            (10, frame_to_display.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.8, Colors.CV_WHITE, 2, cv2.LINE_AA)

                # Afficher la frame
                cv2.imshow(window_name, frame_to_display)

            # 6. Gérer les événements (touche 'q' pour quitter)
            key = cv2.waitKey(1) & 0xFF # ESSENTIEL pour l'affichage et la capture d'event
            if key == ord('q'):
                logging.info("Touche 'q' pressée, arrêt demandé.")
                processing_active = False # Sortir de la boucle principale
                break

            # Vérifier si les threads sont toujours vivants (sécurité)
            if not extractor.capture_thread.is_alive() or not extractor.extraction_thread.is_alive():
                 logging.error("Un des threads worker s'est arrêté de manière inattendue. Arrêt.")
                 processing_active = False
                 break

    except KeyboardInterrupt:
         logging.info(f"{Colors.RED}Arrêt programme demandé par utilisateur (Ctrl+C).{Colors.RESET}")
         processing_active = False # Assurer la sortie de boucle
    except Exception as e_main_loop:
         logging.exception(f"{Colors.RED}Erreur non gérée dans la boucle principale: {e_main_loop}{Colors.RESET}")
         processing_active = False # Assurer la sortie de boucle
    finally:
         # --- Nettoyage final ---
         logging.info("Nettoyage et fermeture...")
         # Signaler l'arrêt aux threads et attendre leur terminaison
         if 'extractor' in locals() and extractor:
             extractor.stop()

         # Fermer toutes les fenêtres OpenCV
         cv2.destroyAllWindows()
         # Essayer de s'assurer que la fenêtre se ferme vraiment
         for _ in range(5): cv2.waitKey(1)

         # Libérer la session Keras (optionnel, peut aider avec la mémoire)
         if 'model' in locals() and model is not None:
             try:
                 logging.debug("Libération session Keras...")
                 tf.keras.backend.clear_session()
                 del model
                 logging.debug("Session Keras libérée.")
             except Exception as e_final_clear:
                  logging.warning(f"Erreur nettoyage final Keras: {e_final_clear}")

         logging.info(f"{Colors.BRIGHT_GREEN}Programme terminé.{Colors.RESET}")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logging.exception(f"{Colors.RED}Erreur non gérée au niveau __main__: {e}{Colors.RESET}")
    finally:
        # Double sécurité pour fermer les fenêtres
        cv2.destroyAllWindows()
        for _ in range(5): cv2.waitKey(1)
        logging.info("Sortie finale du script.")