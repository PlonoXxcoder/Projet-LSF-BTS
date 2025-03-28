import cv2
import mediapipe as mp
import numpy as np
import os
import time
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
import logging

# --- Configuration du logging ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Constants ---
VIDEOS_SOURCE_DIR = "video"  # Dossier contenant vos vidéos
DATA_SAVE_DIR = "data"      # Dossier où sauvegarder les fichiers .npy

# --- Paramètres d'Extraction ---
NUM_POSE_KEYPOINTS = 4
NUM_HAND_KEYPOINTS = 21
NUM_COORDS = 3
FEATURES_PER_FRAME = (NUM_POSE_KEYPOINTS * NUM_COORDS) + \
                     (NUM_HAND_KEYPOINTS * NUM_COORDS) * 2

# --- Fonctions d'Extraction ---
def extract_keypoints(results):
    """Extracts POSE (4) + LEFT HAND (21) + RIGHT HAND (21) keypoints."""
    pose = np.array([[res.x, res.y, res.z] for res in results.pose_landmarks.landmark[:NUM_POSE_KEYPOINTS]]).flatten() \
        if results.pose_landmarks else np.zeros(NUM_POSE_KEYPOINTS * NUM_COORDS)
    lh = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]).flatten() \
        if results.left_hand_landmarks else np.zeros(NUM_HAND_KEYPOINTS * NUM_COORDS)
    rh = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]).flatten() \
        if results.right_hand_landmarks else np.zeros(NUM_HAND_KEYPOINTS * NUM_COORDS)
    return np.concatenate([pose, lh, rh])

def process_frame(frame, holistic_instance):
    """Processes a single frame to extract keypoints."""
    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    image_rgb.flags.writeable = False
    results = holistic_instance.process(image_rgb)
    image_rgb.flags.writeable = True
    keypoints = extract_keypoints(results)
    return keypoints

def process_video(video_file, holistic_instance, print_lock):
    """Traite une seule vidéo pour extraire et sauvegarder les keypoints."""
    video_path = os.path.join(VIDEOS_SOURCE_DIR, video_file)
    base_name = os.path.splitext(video_file)[0]
    save_path = os.path.join(DATA_SAVE_DIR, f"{base_name}_keypoints.npy")

    with print_lock:
        if os.path.exists(save_path):
            logging.info(f"  [Thread {threading.current_thread().name}] Fichier '{os.path.basename(save_path)}' existe déjà. Ignoré.")
            return f"Ignoré (existe déjà): {video_file}"

        logging.info(f"  [Thread {threading.current_thread().name}] Traitement de '{video_file}'...")

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        with print_lock:
            logging.error(f"    [Thread {threading.current_thread().name}] Erreur: Impossible d'ouvrir '{video_file}'.")
        return f"Erreur (ouverture): {video_file}"

    video_keypoints = []
    frame_count = 0
    success = True

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame_count += 1

            try:
                keypoints = process_frame(frame, holistic_instance)
                video_keypoints.append(keypoints)
            except Exception as e:
                with print_lock:
                    logging.error(f"    [Thread {threading.current_thread().name}] Erreur lors du traitement de la frame {frame_count} de '{video_file}': {e}")
                success = False
                break #On sort de la boucle while interne si il y a une erreur.

    except Exception as e:
        with print_lock:
            logging.exception(f"    [Thread {threading.current_thread().name}] Erreur pendant le traitement de '{video_file}': {e}")
        success = False
    finally:
        cap.release()

    if success and video_keypoints:
        video_keypoints_np = np.array(video_keypoints)
        try:
            np.save(save_path, video_keypoints_np)
            with print_lock:
                logging.info(f"    [Thread {threading.current_thread().name}] -> {frame_count} frames. Sauvegarde: '{os.path.basename(save_path)}' (Shape: {video_keypoints_np.shape})")
            return f"Succès: {video_file}"
        except Exception as e:
            with print_lock:
                logging.exception(f"    [Thread {threading.current_thread().name}] Erreur sauvegarde '{os.path.basename(save_path)}': {e}")
            return f"Erreur (sauvegarde): {video_file}"
    elif success:
        with print_lock:
            logging.warning(f"    [Thread {threading.current_thread().name}] -> Aucune frame/keypoint extrait pour '{video_file}'.")
        return f"Échec (pas de données): {video_file}"
    else:
        return f"Erreur (traitement): {video_file}"

# --- Point d'Entrée Principal ---
def main():
    # Créer le dossier de sauvegarde
    os.makedirs(DATA_SAVE_DIR, exist_ok=True)

    # Lister les fichiers vidéo
    logging.info(f"Recherche de vidéos dans '{VIDEOS_SOURCE_DIR}'...")
    try:
        video_files = [f for f in os.listdir(VIDEOS_SOURCE_DIR) if f.endswith(('.mp4', '.avi', '.mov', '.mkv'))]
    except FileNotFoundError:
        logging.error(f"Erreur: Le dossier source '{VIDEOS_SOURCE_DIR}' n'existe pas.")
        exit()

    if not video_files:
        logging.info(f"Aucun fichier vidéo trouvé dans '{VIDEOS_SOURCE_DIR}'.")
        exit()
    else:
        logging.info(f"Trouvé {len(video_files)} fichiers vidéo à traiter.")

    # --- Gestion du Multithreading ---
    max_workers = os.cpu_count() or 4
    logging.info(f"Utilisation de {max_workers} threads pour le traitement.")

    print_lock = threading.Lock()

    results_summary = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(process_video, video_file, mp.solutions.holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5), print_lock): video_file for video_file in video_files}

        for future in as_completed(futures):
            video_file = futures[future]
            try:
                result_status = future.result()
                results_summary.append(result_status)
            except Exception as exc:
                with print_lock:
                    logging.exception(f'Exception pour {video_file}: {exc}')
                results_summary.append(f"Erreur (Exception globale): {video_file}")

    logging.info("\n--- Résumé du Traitement ---")
    success_count = sum(1 for r in results_summary if r.startswith("Succès"))
    ignored_count = sum(1 for r in results_summary if r.startswith("Ignoré"))
    error_count = len(results_summary) - success_count - ignored_count
    logging.info(f"Vidéos traitées avec succès : {success_count}")
    logging.info(f"Vidéos ignorées (existaient déjà) : {ignored_count}")
    logging.info(f"Erreurs rencontrées : {error_count}")
    if error_count > 0:
        logging.info("Détail des erreurs/échecs :")
        for r in results_summary:
            if not r.startswith("Succès") and not r.startswith("Ignoré"):
                logging.info(f"  - {r}")

    logging.info("\n--- Fin de l'extraction des keypoints ---")

if __name__ == "__main__":
    main()