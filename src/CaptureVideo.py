import cv2
import mediapipe as mp
import numpy as np
import tensorflow as tf
import threading
import queue
import os
import logging
import time # Importer time

# --- Configuration du logging ---
# Changez logging.INFO en logging.DEBUG pour voir les messages de débogage détaillés
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - [%(threadName)s] - %(message)s')

# --- ANSI escape codes for colors ---
class Colors:
    RESET = '\x1b[0m'
    BRIGHT_YELLOW = '\x1b[93m'
    BRIGHT_GREEN = '\x1b[92m'
    RED = '\x1b[31m'

# --- Constants ---
MODEL_PATH = "models/model.h5"
VOCABULARY_FILE = "vocabulaire.txt"
FIXED_LENGTH = 46
VIDEOS_DIR = "D:/bonneaup.SNIRW/Test2/video" # CHEMIN ABSOLU (à adapter)
PREDICTION_LOG_FILE = "prediction_log.txt" # <<< AJOUT: Nom du fichier log des prédictions

# --- CONFIGURATION DU SAUT DE FRAMES ---
# Traite 1 frame toutes les N frames lues.
# Augmentez cette valeur pour accélérer (mais analyser moins de données).
# Mettre à 1 pour traiter toutes les frames (plus lent si goulot d'étranglement).
FRAMES_TO_SKIP = 3
logging.info(f"Configuration du saut de frames : Traitement de 1 frame toutes les {FRAMES_TO_SKIP} frames.")
# --------------------------------------

# --- CONFIGURATION REDIMENSIONNEMENT (Optionnel) ---
# Si les vidéos haute résolution causent des problèmes/lenteurs,
# activez le redimensionnement en définissant une largeur maximale.
# Mettre à None pour désactiver.
MAX_FRAME_WIDTH = 1280 # Par exemple: 1920, 1280, ou None
if MAX_FRAME_WIDTH:
    logging.info(f"Redimensionnement activé : Les frames plus larges que {MAX_FRAME_WIDTH}px seront réduites.")
else:
    logging.info("Redimensionnement des frames désactivé.")
# --------------------------------------------------

# --- Paramètres d'Extraction ---
NUM_POSE_KEYPOINTS = 4
NUM_HAND_KEYPOINTS = 21
NUM_COORDS = 3
FEATURES_PER_FRAME = (NUM_POSE_KEYPOINTS * NUM_COORDS) + \
                     (NUM_HAND_KEYPOINTS * NUM_COORDS) * 2
logging.info(f"Nombre de features par frame attendu : {FEATURES_PER_FRAME}")

# --- Utility Functions ---
def load_vocabulary(filepath):
    """Charge le vocabulaire à partir d'un fichier texte."""
    vocabulaire = {}
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split(":")
                if len(parts) == 2 and parts[0] and parts[1].isdigit():
                    vocabulaire[parts[0]] = int(parts[1])
                elif line.strip():
                    logging.warning(f"Format de ligne incorrect dans {filepath}: '{line.strip()}'")
    except (FileNotFoundError, ValueError) as e:
        logging.error(f"Erreur lors du chargement du vocabulaire depuis {filepath}: {e}")
        return {}
    logging.info(f"Vocabulaire chargé depuis {filepath} avec {len(vocabulaire)} mots.")
    return vocabulaire

def extract_keypoints(results):
    """Extracts keypoints (POSE + LH + RH), handling missing detections."""
    pose = np.array([[res.x, res.y, res.z] for res in results.pose_landmarks.landmark[:NUM_POSE_KEYPOINTS]]).flatten() \
        if results.pose_landmarks else np.zeros(NUM_POSE_KEYPOINTS * NUM_COORDS)
    lh = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]).flatten() \
        if results.left_hand_landmarks else np.zeros(NUM_HAND_KEYPOINTS * NUM_COORDS)
    rh = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]).flatten() \
        if results.right_hand_landmarks else np.zeros(NUM_HAND_KEYPOINTS * NUM_COORDS)

    extracted = np.concatenate([pose, lh, rh])
    if extracted.shape[0] != FEATURES_PER_FRAME:
        logging.warning(f"Extraction a produit {extracted.shape[0]} features, attendu {FEATURES_PER_FRAME}. Retour de zéros.")
        return np.zeros(FEATURES_PER_FRAME)
    return extracted

# --- Keypoint Extractor Class ---
class KeypointExtractor:
    def __init__(self):
        self.mp_holistic = mp.solutions.holistic
        self.holistic = None # Sera initialisé dans le thread d'extraction
        self.frame_queue = queue.Queue(maxsize=50)
        self.keypoint_queue = queue.Queue(maxsize=50)
        self.running = threading.Event()
        self.video_capture = None
        self.capture_thread = None
        self.extraction_thread = None
        self.video_path = None

    def capture_frames(self):
        """Thread function to capture frames from video. Implements frame skipping and optional resizing."""
        logging.info(f"Démarrage capture pour {self.video_path}")
        frame_count_read = 0 # Compteur total des frames lues
        frame_count_queued = 0 # Compteur des frames mises dans la queue
        capture_start_time = time.time()
        try:
            self.video_capture = cv2.VideoCapture(self.video_path)
            if not self.video_capture.isOpened():
                # Utiliser logging.error ici car c'est une erreur critique pour ce thread
                logging.error(f"Impossible d'ouvrir la vidéo : {self.video_path}")
                raise ValueError(f"Impossible d'ouvrir la vidéo : {self.video_path}")

            frame_width = int(self.video_capture.get(cv2.CAP_PROP_FRAME_WIDTH))
            frame_height = int(self.video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = self.video_capture.get(cv2.CAP_PROP_FPS)
            logging.info(f"Vidéo ouverte: {Colors.BRIGHT_YELLOW}{self.video_path}{Colors.RESET} ({frame_width}x{frame_height} @ {fps:.2f} FPS)")

            while self.running.is_set():
                logging.debug(f"[Capture] Tentative de lecture de la frame (compteur actuel: {frame_count_read})...")
                read_start = time.time()
                ret, frame = self.video_capture.read()
                read_time = time.time() - read_start

                if not ret:
                    # Log la fin normale ou une erreur de lecture
                    logging.info(f"Fin de la vidéo ou erreur de lecture après {frame_count_read} frames lues: {Colors.BRIGHT_YELLOW}{self.video_path}{Colors.RESET}.")
                    break # Sortir de la boucle while

                # Log le succès et le temps de lecture (si en mode DEBUG)
                logging.debug(f"[Capture] Frame {frame_count_read} lue avec succès (ret={ret}) en {read_time:.4f}s.")
                frame_count_read += 1 # Incrémenter pour chaque frame lue avec succès

                # --- LOGIQUE DE SAUT DE FRAMES ---
                if (frame_count_read -1) % FRAMES_TO_SKIP == 0: # Utiliser -1 car on a déjà incrémenté
                    frame_to_queue = frame

                    # --- LOGIQUE DE REDIMENSIONNEMENT (Optionnel) ---
                    if MAX_FRAME_WIDTH and frame is not None:
                        h, w = frame.shape[:2]
                        if w > MAX_FRAME_WIDTH:
                            scale = MAX_FRAME_WIDTH / w
                            new_h, new_w = int(h * scale), MAX_FRAME_WIDTH
                            try:
                                frame_to_queue = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)
                                logging.debug(f"[Capture] Frame {frame_count_read} redimensionnée de {w}x{h} à {new_w}x{new_h}")
                            except Exception as resize_err:
                                logging.warning(f"[Capture] Erreur lors du redimensionnement de la frame {frame_count_read}: {resize_err}")
                                frame_to_queue = frame # Tenter de continuer avec la frame originale

                    # Traiter cette frame (la mettre dans la queue)
                    if frame_to_queue is not None:
                        try:
                            put_start = time.time()
                            self.frame_queue.put(frame_to_queue, timeout=2.0) # Mettre un timeout pour détecter blocage
                            put_time = time.time() - put_start
                            frame_count_queued += 1
                            if put_time > 0.1:
                                logging.debug(f"[Capture] Mise en queue frame {frame_count_queued} (lue #{frame_count_read}) a pris {put_time:.3f}s (queue peut-être pleine).")
                        except queue.Full:
                             logging.warning(f"[Capture] Queue de frames pleine en essayant d'ajouter frame {frame_count_queued}. Le thread d'extraction est peut-être trop lent. Attente...")
                             # On pourrait ajouter une logique ici, comme sauter plus de frames temporairement, mais put avec timeout suffit pour le log
                             try:
                                 self.frame_queue.put(frame_to_queue) # Réessayer en mode bloquant (ou abandonner)
                                 frame_count_queued += 1
                                 logging.info("[Capture] Place trouvée dans la queue de frames après attente.")
                             except Exception as e_block:
                                 logging.error(f"[Capture] Échec final de mise en queue après queue pleine : {e_block}")
                                 break # Arrêter le thread si on ne peut plus mettre de frames
                        except Exception as e:
                             logging.exception(f"[Capture] Erreur inattendue lors du frame_queue.put : {e}")
                             break # Probablement plus sûr de s'arrêter
                    else:
                        logging.warning(f"[Capture] frame_to_queue est None pour frame lue #{frame_count_read}, skip.")
                else:
                    # Sauter cette frame (ne rien faire)
                    logging.debug(f"[Capture] Frame lue #{frame_count_read} sautée (skip={FRAMES_TO_SKIP}).")
                    pass
                # -----------------------------------

        except Exception as e:
            logging.exception(f"Erreur majeure dans capture_frames pour {Colors.BRIGHT_YELLOW}{self.video_path}{Colors.RESET} : {e}")
        finally:
            if self.video_capture and self.video_capture.isOpened():
                self.video_capture.release()
                logging.info(f"Capture vidéo relâchée pour {Colors.BRIGHT_YELLOW}{self.video_path}{Colors.RESET}")
            # Envoyer le signal de fin seulement si le thread n'a pas été arrêté prématurément
            if self.running.is_set(): # Vérifier si l'arrêt n'a pas déjà été demandé
                 logging.debug(f"[Capture] Envoi du signal de fin (None) à la frame_queue pour {Colors.BRIGHT_YELLOW}{os.path.basename(self.video_path)}{Colors.RESET}...")
                 try:
                     self.frame_queue.put(None, timeout=5.0) # Mettre un timeout ici aussi
                     logging.debug("[Capture] Signal de fin (None) envoyé à la frame_queue.")
                 except queue.Full:
                     logging.error("[Capture] Échec de l'envoi du signal de fin (None) - Queue de frames pleine même après 5s. Problème probable dans le thread d'extraction.")
                 except Exception as e_final:
                     logging.error(f"[Capture] Erreur lors de l'envoi du signal de fin (None) : {e_final}")

            total_time = time.time() - capture_start_time
            logging.info(f"Thread de capture terminé pour {Colors.BRIGHT_YELLOW}{self.video_path}{Colors.RESET}. Mis en queue {frame_count_queued}/{frame_count_read} frames en {total_time:.2f}s.")


    def extract_keypoints_loop(self):
        """Thread function to extract keypoints. Uses blocking put."""
        logging.info(f"Démarrage boucle d'extraction de keypoints pour {Colors.BRIGHT_YELLOW}{os.path.basename(self.video_path)}{Colors.RESET}...")
        frames_processed = 0
        extraction_start_time = time.time()
        try:
            # Initialiser Holistic ICI, dans le thread qui va l'utiliser
            with self.mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
                logging.info("[Extraction] Instance MediaPipe Holistic créée.")
                while True:
                    # Vérifier l'état avant de bloquer sur get()
                    if not self.running.is_set() and self.frame_queue.empty():
                        logging.info(f"[Extraction] Arrêt demandé et queue de frames vide. Arrêt extraction pour {Colors.BRIGHT_YELLOW}{os.path.basename(self.video_path)}{Colors.RESET}.")
                        break

                    frame = None
                    try:
                        get_start = time.time()
                        # Utiliser un timeout permet de vérifier self.running périodiquement
                        frame = self.frame_queue.get(timeout=0.5)
                        get_time = time.time() - get_start
                        logging.debug(f"[Extraction] Frame récupérée de la queue en {get_time:.4f}s.")

                        if frame is None:
                            logging.info(f"[Extraction] Signal de fin (None) reçu dans extract_keypoints_loop pour {Colors.BRIGHT_YELLOW}{os.path.basename(self.video_path)}{Colors.RESET}.")
                            break # Sortir de la boucle while

                    except queue.Empty:
                        # Timeout atteint, la queue est vide pour le moment
                        logging.debug("[Extraction] Queue de frames vide (timeout get). Vérification de l'état...")
                        # La condition d'arrêt est déjà vérifiée au début de la boucle,
                        # donc on continue simplement pour réessayer get()
                        if not self.running.is_set():
                             logging.info(f"[Extraction] Queue de frames vide et arrêt demandé pendant le timeout. Arrêt extraction.")
                             break
                        else:
                             continue # Retourner au début de la boucle pour réessayer get()

                    # Si on a reçu une frame (pas None)
                    try:
                        logging.debug(f"[Extraction] Début traitement MediaPipe pour frame {frames_processed + 1}...")
                        process_start = time.time()
                        # Convertir l'image BGR en RGB
                        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        # Améliorer les performances en marquant l'image comme non modifiable
                        frame_rgb.flags.writeable = False
                        # Détection avec MediaPipe
                        results = holistic.process(frame_rgb)
                        # Rendre l'image à nouveau modifiable si besoin (pas nécessaire ici)
                        # frame_rgb.flags.writeable = True
                        process_time = time.time() - process_start
                        logging.debug(f"[Extraction] Fin traitement MediaPipe en {process_time:.4f}s.")

                        # Extraction des keypoints
                        keypoints = extract_keypoints(results)
                        frames_processed += 1

                        # Mettre les keypoints dans la queue de sortie
                        put_start = time.time()
                        self.keypoint_queue.put(keypoints, timeout=2.0) # Mettre un timeout
                        put_time = time.time() - put_start

                        if put_time > 0.1:
                             logging.debug(f"[Extraction] Mise en queue keypoints {frames_processed} a pris {put_time:.3f}s (queue peut-être pleine).")

                    except queue.Full:
                        logging.warning(f"[Extraction] Queue de keypoints pleine en essayant d'ajouter keypoints {frames_processed}. Le thread principal est peut-être trop lent. Attente...")
                        try:
                            self.keypoint_queue.put(keypoints) # Réessayer en mode bloquant
                            logging.info("[Extraction] Place trouvée dans la queue de keypoints après attente.")
                        except Exception as e_block_kp:
                            logging.error(f"[Extraction] Échec final de mise en queue des keypoints après queue pleine : {e_block_kp}")
                            break # Arrêter si on ne peut plus envoyer les résultats
                    except Exception as e:
                        logging.exception(f"[Extraction] Erreur pendant le traitement MediaPipe ou l'extraction de keypoints sur une frame: {e}")
                        # On peut choisir de continuer avec la frame suivante ou d'arrêter
                        # ici on continue : pass

        except Exception as e:
             logging.exception(f"[Extraction] Erreur majeure dans l'initialisation ou la boucle d'extraction: {e}")
        finally:
             # Envoyer le signal de fin à la queue de keypoints
             logging.debug(f"[Extraction] Envoi du signal de fin (None) à la keypoint_queue pour {Colors.BRIGHT_YELLOW}{os.path.basename(self.video_path)}{Colors.RESET}...")
             try:
                 # S'assurer que le thread principal aura le temps de le recevoir
                 self.keypoint_queue.put(None, timeout=5.0)
                 logging.debug("[Extraction] Signal de fin (None) envoyé à la keypoint_queue.")
             except queue.Full:
                 logging.error("[Extraction] Échec de l'envoi du signal de fin (None) à la keypoint_queue - Queue pleine même après 5s.")
             except Exception as e_final_kp:
                 logging.error(f"[Extraction] Erreur lors de l'envoi du signal de fin (None) à keypoint_queue : {e_final_kp}")

             total_time = time.time() - extraction_start_time
             logging.info(f"Fin de la boucle d'extraction de keypoints pour {Colors.BRIGHT_YELLOW}{os.path.basename(self.video_path)}{Colors.RESET}. Traité {frames_processed} frames en {total_time:.2f}s.")

    def start(self, video_path):
        """Starts the capture and extraction threads."""
        if self.capture_thread is not None and self.capture_thread.is_alive() or \
           self.extraction_thread is not None and self.extraction_thread.is_alive():
            logging.warning("Tentative de démarrer alors que les threads sont déjà actifs. Appel de stop() d'abord...")
            self.stop() # Tenter d'arrêter proprement avant de relancer

        self.video_path = video_path
        self.running.set() # Mettre le flag de fonctionnement AVANT de vider les queues

        # Vider les queues de potentiels restes d'une exécution précédente
        logging.debug("Vidage des queues avant démarrage...")
        while not self.frame_queue.empty():
             try: self.frame_queue.get_nowait()
             except queue.Empty: break
        while not self.keypoint_queue.empty():
             try: self.keypoint_queue.get_nowait()
             except queue.Empty: break
        logging.debug("Queues vidées.")

        # Créer et démarrer les threads
        self.capture_thread = threading.Thread(target=self.capture_frames, name=f"Capture-{os.path.basename(video_path)}")
        self.extraction_thread = threading.Thread(target=self.extract_keypoints_loop, name=f"Extract-{os.path.basename(video_path)}")

        self.capture_thread.start()
        self.extraction_thread.start()
        logging.info(f"Threads démarrés pour {Colors.BRIGHT_YELLOW}{video_path}{Colors.RESET}")

    def stop(self):
        """Signals threads to stop and waits for them to finish."""
        logging.info("Demande d'arrêt des threads...")
        self.running.clear() # Indiquer aux boucles des threads de s'arrêter

        # Attendre la fin des threads avec des timeouts raisonnables
        join_timeout_capture = 10 # secondes (doit être > timeout de read/put)
        join_timeout_extract = 15 # secondes (doit être > timeout de get/process/put)

        if self.capture_thread is not None and self.capture_thread.is_alive():
             logging.info(f"Attente de la fin du thread de capture (max {join_timeout_capture}s)...")
             self.capture_thread.join(timeout=join_timeout_capture)
             if self.capture_thread.is_alive():
                 logging.warning("Le thread de capture n'a pas terminé dans le délai imparti.")
             else:
                 logging.info("Thread de capture terminé.")
        # else:
        #      logging.debug("Thread de capture déjà terminé ou non démarré.") # Moins verbeux

        if self.extraction_thread is not None and self.extraction_thread.is_alive():
             logging.info(f"Attente de la fin du thread d'extraction (max {join_timeout_extract}s)...")
             self.extraction_thread.join(timeout=join_timeout_extract)
             if self.extraction_thread.is_alive():
                 logging.warning("Le thread d'extraction n'a pas terminé dans le délai imparti.")
             else:
                  logging.info("Thread d'extraction terminé.")
        # else:
        #     logging.debug("Thread d'extraction déjà terminé ou non démarré.") # Moins verbeux

        logging.info("Vérification de l'arrêt des threads terminée.")
        self.capture_thread = None
        self.extraction_thread = None


# --- Main Function ---
def main():
    try:
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            try:
                # Configuration typique pour éviter les erreurs OOM sur certains GPU
                for gpu in gpus:
                    tf.config.experimental.set_memory_growth(gpu, True)
                logging.info(f"GPU détecté et configuré : {gpus}")
            except RuntimeError as e:
                # Log l'erreur mais continue sur CPU si possible
                logging.error(f"Erreur lors de la configuration de la croissance mémoire du GPU: {e}")
        else:
            logging.warning("Aucun GPU détecté par TensorFlow. L'inférence se fera sur CPU (peut être significativement plus lent).")

        # Charger le modèle Keras
        if not os.path.exists(MODEL_PATH):
             logging.error(f"Fichier modèle non trouvé : {MODEL_PATH}")
             return
        model = tf.keras.models.load_model(MODEL_PATH)
        logging.info(f"Modèle chargé depuis {MODEL_PATH}")
        try:
            # Vérifier la forme d'entrée attendue par le modèle
            expected_shape = model.input_shape # Doit être (None, FIXED_LENGTH, FEATURES_PER_FRAME)
            logging.info(f"Forme d'entrée attendue par le modèle : {expected_shape}")
            # Vérifier la cohérence avec les constantes définies
            if len(expected_shape) == 3 and expected_shape[1] is not None and expected_shape[2] is not None:
                if expected_shape[1] != FIXED_LENGTH or expected_shape[2] != FEATURES_PER_FRAME:
                     logging.warning(f"Incohérence détectée ! Modèle attend (None, {expected_shape[1]}, {expected_shape[2]}), mais script configuré pour (None, {FIXED_LENGTH}, {FEATURES_PER_FRAME})")
            else:
                logging.warning(f"Impossible de vérifier précisément la forme d'entrée attendue (obtenu: {expected_shape}). Assurez-vous qu'elle correspond à ({FIXED_LENGTH}, {FEATURES_PER_FRAME}).")
        except Exception as e:
            logging.warning(f"Impossible de vérifier automatiquement la forme d'entrée du modèle: {e}")

    except Exception as e:
        logging.exception(f"Erreur majeure lors de l'initialisation TensorFlow ou du chargement du modèle : {e}")
        return # Arrêter si le modèle ne peut pas être chargé

    # Charger le vocabulaire
    vocabulaire = load_vocabulary(VOCABULARY_FILE)
    if not vocabulaire:
        logging.error("Erreur critique : Le vocabulaire est vide ou n'a pas pu être chargé. Arrêt.")
        return
    # Créer le mapping inverse pour obtenir le mot à partir de l'index prédit
    index_to_word = {i: word for word, i in vocabulaire.items()}

    # Vérifier le dossier des vidéos
    if not os.path.isdir(VIDEOS_DIR):
        logging.error(f"Le chemin vidéo spécifié n'est pas un dossier valide : {VIDEOS_DIR}")
        return

    # Lister les fichiers vidéo à traiter
    try:
        video_files_to_process = [f for f in os.listdir(VIDEOS_DIR) if f.lower().endswith(('.mp4', '.avi', '.mov', '.mkv'))]
    except Exception as e:
        logging.exception(f"Erreur lors du listage des fichiers dans {VIDEOS_DIR}: {e}")
        return

    logging.info(f"Trouvé {len(video_files_to_process)} vidéos à traiter dans {Colors.BRIGHT_YELLOW}{VIDEOS_DIR}{Colors.RESET}")
    if not video_files_to_process:
        logging.info("Aucune vidéo trouvée. Fin du programme.")
        return

    main_start_time = time.time()
    try:
        # Boucle principale sur chaque fichier vidéo
        for i, video_file in enumerate(video_files_to_process):
            video_path = os.path.join(VIDEOS_DIR, video_file)
            window_name = f"Video - {os.path.basename(video_path)}" # Nom de fenêtre unique par vidéo
            logging.info(f"--- Début traitement vidéo {i+1}/{len(video_files_to_process)}: {Colors.BRIGHT_YELLOW}{video_path}{Colors.RESET} ---")
            video_start_time = time.time()

            extractor = KeypointExtractor()
            extractor.start(video_path) # Démarre les threads de capture et d'extraction

            sequence = [] # Stocke la séquence actuelle de keypoints
            all_predictions_indices = [] # Stocke tous les index prédits pour cette vidéo
            frame_display_buffer = None # Dernière frame reçue pour l'affichage
            processing_active = True # Flag pour contrôler la boucle principale de traitement
            frames_processed_main = 0 # Compteur de keypoints traités dans main
            predictions_made = 0
            last_pred_text = "Initialisation..." # Texte à afficher sur la vidéo

            try:
                while processing_active:
                    keypoints = None
                    try:
                        # Récupérer les keypoints extraits (avec timeout pour rester réactif)
                        keypoints = extractor.keypoint_queue.get(timeout=0.5) # Timeout en secondes

                        if keypoints is None:
                            logging.info("[Main] Signal de fin (None) reçu de la queue de keypoints. Fin du traitement pour cette vidéo.")
                            processing_active = False # Terminer la boucle while
                        else:
                            frames_processed_main += 1 # Incrémenter si on a reçu des keypoints valides
                            logging.debug(f"[Main] Keypoints {frames_processed_main} reçus.")

                    except queue.Empty:
                        # La queue est vide, vérifier si les threads sont toujours actifs
                        logging.debug("[Main] Queue de keypoints vide (timeout get).")
                        # Si les threads ne tournent plus ET que la queue est vide, on a fini
                        if (not extractor.running.is_set() and
                            (extractor.capture_thread is None or not extractor.capture_thread.is_alive()) and
                            (extractor.extraction_thread is None or not extractor.extraction_thread.is_alive()) and
                            extractor.keypoint_queue.empty()):
                            logging.info("[Main] Threads arrêtés et queue de keypoints vide. Fin définitive du traitement pour cette vidéo.")
                            processing_active = False
                        else:
                            # Les threads tournent encore ou la queue n'est pas garantie vide, on continue d'attendre
                            pass # Continue la boucle pour réessayer get() ou gérer l'affichage/input

                    # Traitement si des keypoints ont été reçus
                    if keypoints is not None:
                        sequence.append(keypoints)
                        # Garder seulement les N dernières frames (N = FIXED_LENGTH)
                        sequence = sequence[-FIXED_LENGTH:]
                        current_sequence_len = len(sequence)

                        # Préparer la séquence pour le modèle (padding si nécessaire)
                        padded_sequence = None
                        if current_sequence_len <= FIXED_LENGTH:
                            # Si on a moins de frames que nécessaire, ajouter du padding (zéros) au début
                            if current_sequence_len < FIXED_LENGTH:
                                logging.debug(f"[Main] Padding séquence de {current_sequence_len} à {FIXED_LENGTH} frames.")
                                padding = np.zeros((FIXED_LENGTH - current_sequence_len, FEATURES_PER_FRAME))
                                padded_sequence = np.concatenate((padding, np.array(sequence)), axis=0)
                            else: # Exactement le bon nombre de frames
                                padded_sequence = np.array(sequence)

                            # Vérifier la forme avant de prédire
                            if padded_sequence is not None and padded_sequence.shape == (FIXED_LENGTH, FEATURES_PER_FRAME):
                                # Ajouter une dimension pour le batch (taille 1)
                                reshaped_sequence = np.expand_dims(padded_sequence, axis=0)
                                try:
                                    predict_start = time.time()
                                    # Faire la prédiction
                                    res = model.predict(reshaped_sequence, verbose=0)[0]
                                    predict_time = time.time() - predict_start
                                    predictions_made += 1

                                    # Obtenir l'index du mot prédit (celui avec la plus haute probabilité)
                                    predicted_word_index = np.argmax(res)
                                    prediction_confidence = res[predicted_word_index]
                                    all_predictions_indices.append(predicted_word_index) # Stocker l'index

                                    # Mettre à jour le texte à afficher
                                    word = index_to_word.get(predicted_word_index, f"Index_{predicted_word_index}?")
                                    last_pred_text = f"{word} ({prediction_confidence:.2f})"
                                    logging.debug(f"[Main] Prédiction {predictions_made}: {last_pred_text} (temps: {predict_time:.4f}s)")

                                except Exception as e:
                                    logging.exception(f"[Main] Erreur lors de model.predict: {e}")
                                    last_pred_text = "Erreur Prediction"
                            else:
                                 logging.warning(f"[Main] Shape incorrecte ({padded_sequence.shape if padded_sequence is not None else 'None'}) ou padded_sequence est None avant prédiction. Attendu: ({FIXED_LENGTH}, {FEATURES_PER_FRAME})")
                                 # Ne pas mettre à jour last_pred_text ou le mettre à "Shape Error" ?
                                 # last_pred_text = "Shape Error"

                    # --- Gestion de l'affichage ---
                    # Essayer de récupérer une nouvelle frame pour l'affichage (non bloquant)
                    try:
                        new_frame = extractor.frame_queue.get_nowait()
                        if new_frame is not None:
                            frame_display_buffer = new_frame # Mettre à jour le buffer d'affichage
                    except queue.Empty:
                        pass # Pas de nouvelle frame disponible, on garde l'ancienne

                    # Afficher la dernière frame reçue avec le texte de prédiction
                    display_frame_with_text = None
                    if frame_display_buffer is not None and frame_display_buffer.size > 0:
                        try:
                            # Copier pour éviter de modifier le buffer original si besoin plus tard
                            display_frame_with_text = frame_display_buffer.copy()
                            # Ajouter le texte sur la frame copiée
                            cv2.putText(display_frame_with_text, last_pred_text, (10, 30),
                                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
                        except Exception as e_draw:
                             logging.warning(f"[Main] Erreur lors de la préparation de l'image pour affichage: {e_draw}")
                             display_frame_with_text = frame_display_buffer # Afficher sans texte si erreur

                    # Afficher si on a une image
                    if display_frame_with_text is not None:
                        try:
                            cv2.imshow(window_name, display_frame_with_text)
                            # Attendre très brièvement une touche (1ms)
                            key = cv2.waitKey(1) & 0xFF
                            # Si 'q' est pressée, arrêter tout
                            if key == ord('q'):
                                logging.info("Touche 'q' pressée, arrêt global demandé.")
                                processing_active = False # Sortir de la boucle while interne
                                raise KeyboardInterrupt("Arrêt demandé par l'utilisateur via 'q'") # Pour arrêter la boucle externe
                        except cv2.error as e:
                            # Gérer le cas où la fenêtre a été fermée manuellement
                            if "NULL window" in str(e) or "Invalid window handle" in str(e):
                                logging.warning(f"[Main] Fenêtre '{window_name}' fermée manuellement ou invalide. Arrêt de l'affichage pour cette vidéo.")
                                processing_active = False # Arrêter le traitement de cette vidéo
                            else:
                                logging.warning(f"[Main] Erreur cv2.imshow/waitKey non gérée: {e}")
                                # Optionnel: arrêter aussi ? processing_active = False
                        except Exception as e_show:
                             logging.exception(f"[Main] Erreur inattendue pendant l'affichage: {e_show}")
                             processing_active = False # Sécurité: arrêter en cas d'erreur inconnue
                    # Si processing_active est devenu False (par signal None, 'q', ou erreur), sortir
                    if not processing_active:
                         break

                # Fin de la boucle while processing_active pour cette vidéo
                logging.info(f"Fin de la boucle de traitement principale pour {video_file}.")

            finally:
                # Nettoyage spécifique à la vidéo, même en cas d'erreur dans la boucle
                logging.info(f"Nettoyage pour la vidéo {Colors.BRIGHT_YELLOW}{video_file}{Colors.RESET}...")
                if extractor.running.is_set(): # Si l'arrêt n'a pas déjà été demandé (ex: par 'q')
                     extractor.stop() # Demande l'arrêt et attend la fin des threads
                else:
                    logging.info("Threads déjà signalés pour arrêt.")
                    # On peut quand même appeler stop pour s'assurer du join, ou juste attendre ici si stop a déjà été appelé
                    extractor.stop() # Appeler stop garantit la tentative de join même si running est False

                # Fermer la fenêtre OpenCV spécifique à cette vidéo
                try:
                     # Vérifier si la fenêtre existe et est visible avant de la détruire
                     if cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) >= 1:
                         cv2.destroyWindow(window_name)
                         logging.info(f"Fenêtre '{window_name}' fermée.")
                     # else: # Moins verbeux
                     #    logging.debug(f"Fenêtre '{window_name}' non trouvée ou déjà fermée.")
                except cv2.error:
                    # Ignorer les erreurs si la fenêtre n'existe plus
                    logging.debug(f"Erreur lors de la tentative de fermeture de la fenêtre '{window_name}' (probablement déjà fermée).")
                except Exception as e_close:
                    logging.warning(f"Erreur inattendue lors de la fermeture de la fenêtre '{window_name}': {e_close}")

                video_end_time = time.time()
                # Log résumé du traitement de la vidéo
                logging.info(f"Vidéo {Colors.BRIGHT_YELLOW}{video_file}{Colors.RESET}: Traitement principal a reçu {frames_processed_main} keypoints, {predictions_made} prédictions effectuées.")
                logging.info(f"Temps de traitement total pour {Colors.BRIGHT_YELLOW}{video_file}{Colors.RESET}: {video_end_time - video_start_time:.2f} secondes.")

            # --- Analyse des prédictions pour la vidéo terminée ---
            final_prediction_text = "Aucune prédiction" # Valeur par défaut
            if all_predictions_indices:
                try:
                    # Trouver l'index prédit le plus fréquemment
                    most_common_index = max(set(all_predictions_indices), key=all_predictions_indices.count)
                    # Obtenir le mot correspondant
                    predicted_word = index_to_word.get(most_common_index, f"Index_{most_common_index}?")
                    count = all_predictions_indices.count(most_common_index)
                    final_prediction_text = predicted_word # Mettre à jour la prédiction finale
                    logging.info(f"-> Le mot prédit le plus fréquent pour {Colors.BRIGHT_YELLOW}{video_file}{Colors.RESET} est : {Colors.BRIGHT_GREEN}{predicted_word}{Colors.RESET} ({count}/{predictions_made} prédictions)")
                except Exception as e:
                    logging.exception(f"Erreur lors de l'analyse des prédictions finales pour {video_file}: {e}")
                    final_prediction_text = "Erreur analyse prédiction" # Indiquer une erreur
            else:
                # Aucune prédiction n'a été faite (vidéo trop courte, erreur, etc.)
                logging.warning(f"{Colors.RED}-> Aucune prédiction générée pour la vidéo {Colors.BRIGHT_YELLOW}{video_file}{Colors.RESET}.{Colors.RESET}")
                # final_prediction_text reste "Aucune prédiction"

            # <<< AJOUT: Écriture dans le fichier log des prédictions >>>
            try:
                with open(PREDICTION_LOG_FILE, "a", encoding="utf-8") as log_f:
                    log_f.write(f"{video_file}: {final_prediction_text}\n")
                logging.info(f"Résultat enregistré dans {PREDICTION_LOG_FILE}")
            except IOError as e:
                logging.error(f"Impossible d'écrire dans le fichier log {PREDICTION_LOG_FILE}: {e}")
            # <<< FIN AJOUT >>>

            logging.info(f"--- Fin traitement vidéo : {Colors.BRIGHT_YELLOW}{video_path}{Colors.RESET} ---")
            # Optionnel: petite pause entre les vidéos
            time.sleep(0.5)

        # Fin de la boucle sur toutes les vidéos
        total_main_time = time.time() - main_start_time
        logging.info(f"Traitement de toutes les {len(video_files_to_process)} vidéos terminé en {total_main_time:.2f} secondes.")

    except KeyboardInterrupt:
         # Gérer l'interruption clavier (Ctrl+C ou 'q') proprement
         logging.info("Arrêt du programme demandé par l'utilisateur (KeyboardInterrupt).")
         # Assurez-vous que le dernier extractor est stoppé s'il existe et tournait encore
         if 'extractor' in locals() and extractor is not None:
             logging.info("Tentative d'arrêt des derniers threads actifs...")
             extractor.stop()
    except Exception as e:
         # Capturer toute autre exception non prévue
         logging.exception(f"Une erreur inattendue a mis fin au programme principal: {e}")
    finally:
         # Nettoyage final, s'assurer que toutes les fenêtres OpenCV sont fermées
         logging.info("Nettoyage final : Fermeture de toutes les fenêtres OpenCV restantes...")
         cv2.destroyAllWindows()
         # Ajouter une petite attente pour laisser le temps à l'OS de fermer les fenêtres
         time.sleep(0.5)
         logging.info("Programme terminé.")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        # Ce bloc ne devrait pas être atteint si bien géré dans main(), mais sécurité
        logging.info("Interruption clavier détectée au niveau __main__ (devrait être gérée dans main).")
    except Exception as e:
        logging.exception(f"Erreur non gérée au niveau __main__: {e}")
    finally:
        # Double sécurité pour fermer les fenêtres
        cv2.destroyAllWindows()
        logging.info("Sortie du script.")