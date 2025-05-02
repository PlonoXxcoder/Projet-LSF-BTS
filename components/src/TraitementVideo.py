# -*- coding: utf-8 -*-
import cv2
import mediapipe as mp
import numpy as np
import os
import time
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
import logging

# --- Configuration du logging ---
# Format amélioré pour inclure le nom du fichier vidéo dans les logs si possible
# (Note: L'ajout direct ici est complexe, mais le message de log dans process_video l'inclut)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - [Thread %(threadName)s] - %(message)s')

# --- Constants ---
VIDEOS_SOURCE_DIR = "video"  # Dossier contenant vos vidéos
DATA_SAVE_DIR = "data"      # Dossier où sauvegarder les fichiers .npy

# --- Paramètres d'Extraction ---
NUM_POSE_KEYPOINTS = 4
NUM_HAND_KEYPOINTS = 21
NUM_COORDS = 3 # x, y, z

# Indices des points des lèvres (exemple basé sur MediaPipe Face Mesh diagram)
MOUTH_LANDMARK_INDICES = sorted(list(set([
    # Lèvre extérieure supérieure
    61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291,
    # Lèvre intérieure supérieure
    78, 95, 88, 178, 87, 14, 317, 402, 318, 324, 308,
    # Lèvre extérieure inférieure (partagé avec sup en partie)
    185, 40, 39, 37, 0, 267, 269, 270, 409,
    # Lèvre intérieure inférieure (partagé avec sup en partie)
    191, 80, 81, 82, 13, 312, 311, 310, 415
])))
NUM_MOUTH_KEYPOINTS = len(MOUTH_LANDMARK_INDICES)

# ---> Calcul FEATURES_PER_FRAME <---
FEATURES_PER_FRAME = (NUM_POSE_KEYPOINTS * NUM_COORDS) + \
                     (NUM_HAND_KEYPOINTS * NUM_COORDS) * 2 + \
                     (NUM_MOUTH_KEYPOINTS * NUM_COORDS) # Ajout des features de la bouche

# Logging des paramètres d'extraction
logging.info("--- Configuration Extraction ---")
logging.info(f"Dossier source vidéos: {VIDEOS_SOURCE_DIR}")
logging.info(f"Dossier sauvegarde data: {DATA_SAVE_DIR}")
logging.info(f"Nombre de points Pose: {NUM_POSE_KEYPOINTS}")
logging.info(f"Nombre de points Main (par main): {NUM_HAND_KEYPOINTS}")
logging.info(f"Nombre de points Bouche sélectionnés: {NUM_MOUTH_KEYPOINTS}")
logging.info(f"Nombre total de features par frame attendu : {FEATURES_PER_FRAME}")
logging.info("-----------------------------")

# --- Fonctions d'Extraction ---
def extract_keypoints(results):
    """
    Extracts POSE (4) + LEFT HAND (21) + RIGHT HAND (21) + MOUTH (NUM_MOUTH_KEYPOINTS) keypoints.
    Retourne un array numpy de taille FEATURES_PER_FRAME ou un array de zéros si erreur ou points manquants.
    """
    # Initialisation avec des zéros pour garantir la taille correcte même si des parties manquent
    pose = np.zeros(NUM_POSE_KEYPOINTS * NUM_COORDS)
    lh = np.zeros(NUM_HAND_KEYPOINTS * NUM_COORDS)
    rh = np.zeros(NUM_HAND_KEYPOINTS * NUM_COORDS)
    mouth = np.zeros(NUM_MOUTH_KEYPOINTS * NUM_COORDS)

    try:
        # Pose (4 points)
        if results.pose_landmarks and len(results.pose_landmarks.landmark) >= NUM_POSE_KEYPOINTS:
            pose = np.array([[res.x, res.y, res.z] for res in results.pose_landmarks.landmark[:NUM_POSE_KEYPOINTS]]).flatten()

        # Mains (21 points chacune)
        if results.left_hand_landmarks and len(results.left_hand_landmarks.landmark) == NUM_HAND_KEYPOINTS:
            lh = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]).flatten()
        if results.right_hand_landmarks and len(results.right_hand_landmarks.landmark) == NUM_HAND_KEYPOINTS:
            rh = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]).flatten()

        # Bouche (NUM_MOUTH_KEYPOINTS points)
        if results.face_landmarks:
            # Vérifier si tous les indices demandés existent dans les landmarks détectés
            if all(idx < len(results.face_landmarks.landmark) for idx in MOUTH_LANDMARK_INDICES):
                 mouth_points = [results.face_landmarks.landmark[i] for i in MOUTH_LANDMARK_INDICES]
                 mouth = np.array([[res.x, res.y, res.z] for res in mouth_points]).flatten()
            else:
                 # Log seulement si des landmarks faciaux sont détectés mais pas assez/les bons
                 logging.warning(f"Indices de bouche manquants dans face_landmarks (détectés: {len(results.face_landmarks.landmark)}). Retour zéros pour la bouche.")
                 # mouth reste à zéro

    except Exception as e:
        # Capturer toute autre erreur potentielle pendant l'accès aux landmarks
        logging.error(f"Erreur inattendue lors de l'accès aux landmarks: {e}. Utilisation de zéros pour les parties affectées.")
        # Les valeurs par défaut (zéros) seront utilisées

    # Concaténation de toutes les features
    extracted = np.concatenate([pose, lh, rh, mouth])

    # Vérification finale de la taille (devrait toujours être correcte maintenant)
    if extracted.shape[0] != FEATURES_PER_FRAME:
        logging.error(f"ERREUR CRITIQUE: Taille d'extraction incohérente ({extracted.shape[0]}), attendu {FEATURES_PER_FRAME}. Retour de zéros.")
        return np.zeros(FEATURES_PER_FRAME)

    return extracted

def process_frame(frame, holistic_instance):
    """Processes a single frame to extract keypoints."""
    results = None # Initialiser results
    keypoints = np.zeros(FEATURES_PER_FRAME) # Initialiser avec des zéros

    try:
        # Convertir l'image BGR en RGB
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # Rendre l'image non modifiable pour potentiellement améliorer les performances
        image_rgb.flags.writeable = False

        # Traiter l'image avec MediaPipe Holistic
        results = holistic_instance.process(image_rgb)

        # Rendre l'image à nouveau modifiable (si nécessaire pour d'autres opérations)
        # image_rgb.flags.writeable = True # Pas nécessaire ici

        # Extraire les keypoints combinés (pose, mains, bouche)
        # La fonction extract_keypoints gère maintenant les cas où results ou des parties de results sont None
        keypoints = extract_keypoints(results)

    except cv2.error as cv_err:
        # Erreur spécifique à OpenCV (conversion de couleur, etc.)
        logging.error(f"Erreur OpenCV lors du pré-traitement de la frame: {cv_err}")
        # keypoints reste à zéro
    except Exception as e:
        # Capturer d'autres erreurs potentielles (ex: pdt holistic.process si non gérée par MediaPipe)
        logging.error(f"Erreur inattendue dans process_frame: {e}")
        # keypoints reste à zéro
    finally:
        # Optionnel : Libérer explicitement la mémoire de l'image RGB si la pression mémoire est extrême
        # del image_rgb
        pass

    return keypoints

# MODIFICATION: holistic_instance est maintenant créé DANS cette fonction
# pour s'assurer que chaque thread/tâche a sa propre instance isolée.
def process_video(video_file, print_lock):
    """Traite une seule vidéo pour extraire et sauvegarder les keypoints."""
    thread_name = threading.current_thread().name # Pour logging plus précis
    video_path = os.path.join(VIDEOS_SOURCE_DIR, video_file)
    base_name = os.path.splitext(video_file)[0]
    save_path = os.path.join(DATA_SAVE_DIR, f"{base_name}_keypoints.npy")

    # Utiliser le verrou pour l'affichage et la vérification de l'existence du fichier
    with print_lock:
        if os.path.exists(save_path):
            logging.info(f"[{thread_name}] Fichier '{os.path.basename(save_path)}' existe déjà. Ignoré pour '{video_file}'.")
            return f"Ignoré (existe déjà): {video_file}"
        logging.info(f"[{thread_name}] Début traitement de '{video_file}'...")

    cap = None # Initialiser à None
    video_keypoints = []
    frame_count = 0
    success = False # Indicateur de succès spécifique à cette vidéo
    error_message = "Inconnue" # Message d'erreur par défaut

    try:
        # --- Initialisation de MediaPipe Holistic DANS la fonction/thread ---
        # Ceci garantit une instance séparée pour chaque tâche concurrente,
        # évitant les problèmes potentiels de partage d'état et rendant
        # l'initialisation plus robuste même si l'ouverture de la vidéo échoue.
        with mp.solutions.holistic.Holistic(
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5) as holistic:

            # Ouvrir la capture vidéo APRÈS l'initialisation de Holistic (au cas où Holistic échouerait)
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                # Utiliser le verrou pour logger l'erreur d'ouverture
                error_message = f"Impossible d'ouvrir la vidéo '{video_file}'."
                with print_lock:
                    logging.error(f"[{thread_name}] {error_message}")
                # Pas besoin de 'finally' pour cap.release() car cap est None ou non ouvert
                return f"Erreur (ouverture): {video_file}"

            # Boucle de lecture des frames de la vidéo
            while True:
                ret, frame = False, None # Initialiser pour ce tour de boucle
                try:
                    ret, frame = cap.read()
                except cv2.error as cv_read_err:
                    # Gérer spécifiquement les erreurs de lecture OpenCV (souvent mémoire insuffisante)
                    error_message = f"Erreur OpenCV (probablement mémoire) lors de la lecture de la frame {frame_count + 1} de '{video_file}': {cv_read_err}"
                    with print_lock:
                        logging.error(f"[{thread_name}] {error_message}")
                    success = False # Échec pour cette vidéo
                    break # Sortir de la boucle while
                except Exception as read_err:
                    # Autres erreurs potentielles pendant la lecture
                    error_message = f"Erreur inattendue lors de la lecture de la frame {frame_count + 1} de '{video_file}': {read_err}"
                    with print_lock:
                        logging.error(f"[{thread_name}] {error_message}")
                    success = False
                    break # Sortir de la boucle while

                if not ret:
                    # Fin de la vidéo atteinte normalement OU erreur de lecture gérée ci-dessus
                    if error_message == "Inconnue": # Si aucune erreur n'a été définie avant
                       success = True # On considère que c'est une fin normale
                    break # Sortir de la boucle while

                frame_count += 1

                try:
                    # Traiter la frame actuelle pour extraire les keypoints
                    # Passe l'instance 'holistic' créée au début de la fonction
                    keypoints = process_frame(frame, holistic)
                    video_keypoints.append(keypoints)
                    # Optionnel: libérer la mémoire du frame si nécessaire
                    # del frame
                except RuntimeError as rt_err:
                     # Erreurs spécifiques de MediaPipe/TFLite (comme XNNPACK)
                    error_message = f"Erreur Runtime (MediaPipe/TFLite) lors du traitement de la frame {frame_count} de '{video_file}': {rt_err}"
                    with print_lock:
                        logging.error(f"[{thread_name}] {error_message}")
                    success = False
                    break # Sortir de la boucle while interne en cas d'erreur de frame
                except Exception as frame_proc_err:
                    # Log d'erreur si le traitement d'une frame échoue
                    error_message = f"Erreur lors du traitement de la frame {frame_count} de '{video_file}': {frame_proc_err}"
                    with print_lock:
                        logging.error(f"[{thread_name}] {error_message}")
                    success = False # Marquer l'échec pour cette vidéo
                    break # Sortir de la boucle while interne

            # Fin de la boucle while
            if error_message == "Inconnue" and frame_count > 0:
                 success = True # Succès si on est sorti normalement et qu'on a lu des frames

    except Exception as e:
        # Capturer les erreurs potentielles HORS de la boucle de lecture
        # (ex: échec initialisation Holistic, erreur non prévue)
        error_message = f"Erreur globale pendant le traitement de '{video_file}': {e}"
        with print_lock:
            # Utiliser logging.exception pour avoir la traceback complète
            logging.exception(f"[{thread_name}] {error_message}")
        success = False # Marquer l'échec
    finally:
        # Toujours libérer la ressource de capture vidéo si elle a été ouverte
        if cap is not None and cap.isOpened():
            cap.release()

    # --- Sauvegarde des keypoints ---
    if success and video_keypoints:
        # Convertir la liste de keypoints en un seul array numpy 2D
        video_keypoints_np = np.array(video_keypoints)
        try:
            np.save(save_path, video_keypoints_np)
            with print_lock:
                logging.info(f"[{thread_name}] -> {frame_count} frames traités. Sauvegarde: '{os.path.basename(save_path)}' (Shape: {video_keypoints_np.shape}) pour '{video_file}'.")
            return f"Succès: {video_file}"
        except Exception as save_err:
            with print_lock:
                logging.exception(f"[{thread_name}] Erreur sauvegarde '{os.path.basename(save_path)}' pour '{video_file}': {save_err}")
            return f"Erreur (sauvegarde): {video_file}"
    elif success: # Traitement "réussi" mais aucune donnée extraite
        with print_lock:
            logging.warning(f"[{thread_name}] -> Aucune frame/keypoint extrait pour '{video_file}' (frames lues: {frame_count}). Aucun fichier .npy généré.")
        return f"Échec (pas de données): {video_file}"
    else: # Si success est False (erreur pendant le traitement)
        # Le message d'erreur spécifique a déjà été loggué
        return f"Erreur (traitement - {error_message[:50]}...): {video_file}" # Retourne un statut plus descriptif

# --- Point d'Entrée Principal ---
def main():
    start_time = time.time()
    # Créer le dossier de sauvegarde s'il n'existe pas
    try:
        os.makedirs(DATA_SAVE_DIR, exist_ok=True)
    except OSError as e:
        logging.error(f"Impossible de créer le dossier de sauvegarde '{DATA_SAVE_DIR}': {e}")
        exit()

    # Lister les fichiers vidéo dans le dossier source
    logging.info(f"Recherche de vidéos dans '{VIDEOS_SOURCE_DIR}'...")
    try:
        all_files = os.listdir(VIDEOS_SOURCE_DIR)
        video_files = [f for f in all_files if f.lower().endswith(('.mp4', '.avi', '.mov', '.mkv', '.webm')) and os.path.isfile(os.path.join(VIDEOS_SOURCE_DIR, f))]
    except FileNotFoundError:
        logging.error(f"Erreur: Le dossier source '{VIDEOS_SOURCE_DIR}' n'existe pas.")
        exit()
    except Exception as e:
         logging.error(f"Erreur lors du listage des fichiers dans '{VIDEOS_SOURCE_DIR}': {e}")
         exit()

    if not video_files:
        logging.info(f"Aucun fichier vidéo trouvé dans '{VIDEOS_SOURCE_DIR}'.")
        exit()
    else:
        logging.info(f"Trouvé {len(video_files)} fichiers vidéo à traiter.")
        # Optionnel: Trier les fichiers pour un ordre de traitement prévisible
        video_files.sort()

    # --- Gestion du Multithreading ---
    # !! MODIFICATION IMPORTANTE !!
    # Réduire drastiquement le nombre de workers pour éviter l'épuisement de la RAM.
    # Commencez avec 1 ou 2, et augmentez prudemment SEULEMENT si votre système a beaucoup de RAM
    # et que vous ne rencontrez plus d'erreurs "Insufficient Memory".
    # Surveillez l'utilisation de la RAM de votre système pendant l'exécution.

    # max_workers = os.cpu_count() or 4 # <-- LIGNE ORIGINALE (trop agressive si RAM limitée)
    max_workers = 2  # <--- RÉDUIRE ICI ! Essayez 1, 2, peut-être 4 si vous avez >16GB RAM.
    logging.info(f"Utilisation d'un maximum de {max_workers} threads pour le traitement (réduit pour limiter l'utilisation de la RAM).")

    # Créer un verrou pour synchroniser les accès concurrents aux logs (print/logging)
    # et à la vérification de l'existence des fichiers pour éviter les race conditions.
    print_lock = threading.Lock()

    results_summary = [] # Liste pour stocker les statuts de chaque vidéo
    futures = {}         # Dictionnaire pour mapper les futures aux noms de fichiers

    try:
        # Utiliser ThreadPoolExecutor pour gérer les threads
        with ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix='Worker') as executor:
            # Soumettre chaque tâche de traitement vidéo au pool
            for video_file in video_files:
                # Chaque tâche appelle process_video, qui créera sa propre instance Holistic
                # On passe seulement le nom du fichier et le verrou partagé.
                future = executor.submit(process_video, video_file, print_lock)
                futures[future] = video_file # Associer le future au nom de fichier

            # Récupérer les résultats au fur et à mesure que les tâches se terminent
            logging.info(f"Traitement lancé pour {len(futures)} vidéos. Attente des résultats...")
            processed_count = 0
            for future in as_completed(futures):
                video_file = futures[future] # Récupérer le nom du fichier associé
                processed_count += 1
                try:
                    # Obtenir le résultat retourné par process_video (la chaîne de statut)
                    result_status = future.result()
                    results_summary.append(result_status)
                    # Log de progression optionnel
                    # with print_lock:
                    #    logging.info(f"[{threading.current_thread().name}] Terminé ({processed_count}/{len(futures)}): {result_status}")

                except Exception as exc:
                    # Capturer toute exception non gérée survenue PENDANT l'exécution de la tâche
                    # (normalement gérée dans process_video, mais sécurité supplémentaire)
                    with print_lock:
                        logging.exception(f"[{threading.current_thread().name}] Exception non gérée récupérée pour {video_file}: {exc}")
                    results_summary.append(f"Erreur (Exception Toplevel Thread): {video_file}")

    except Exception as pool_exc:
         # Erreur lors de la création ou gestion du Pool lui-même
         logging.exception(f"Erreur majeure avec ThreadPoolExecutor: {pool_exc}")


    # --- Afficher le résumé final du traitement ---
    end_time = time.time()
    total_time = end_time - start_time
    logging.info("\n--- Résumé Complet du Traitement ---")
    logging.info(f"Temps total d'exécution: {total_time:.2f} secondes")

    success_count = sum(1 for r in results_summary if r.startswith("Succès"))
    ignored_count = sum(1 for r in results_summary if r.startswith("Ignoré"))
    error_no_data_count = sum(1 for r in results_summary if r.startswith("Échec (pas de données)"))
    error_processing_count = sum(1 for r in results_summary if r.startswith("Erreur")) # Inclut sauvegarde, ouverture, traitement, exceptions...

    logging.info(f"Vidéos traitées avec succès et sauvegardées : {success_count}")
    logging.info(f"Vidéos ignorées (fichier .npy existait déjà) : {ignored_count}")
    logging.info(f"Échecs (aucune donnée extraite malgré traitement) : {error_no_data_count}")
    logging.info(f"Erreurs rencontrées pendant le traitement : {error_processing_count}")
    logging.info(f"Total traité/tenté : {len(results_summary)} / {len(video_files)}")


    # Afficher les détails des erreurs/échecs s'il y en a eu
    if error_no_data_count + error_processing_count > 0:
        logging.info("\n--- Détail des erreurs/échecs ---")
        for r in results_summary:
            if not r.startswith("Succès") and not r.startswith("Ignoré"):
                # Logger comme warning ou error pour meilleure visibilité
                if r.startswith("Échec"):
                    logging.warning(f"  - {r}")
                else:
                     logging.error(f"  - {r}")

    logging.info("\n--- Fin de l'extraction des keypoints ---")

if __name__ == "__main__":
    # Vérifications initiales simples
    if not os.path.exists(VIDEOS_SOURCE_DIR):
         print(f"ERREUR: Le dossier source '{VIDEOS_SOURCE_DIR}' n'existe pas. Veuillez le créer ou corriger le chemin.")
         exit(1)
    if not os.path.exists(DATA_SAVE_DIR):
        print(f"INFO: Le dossier de données '{DATA_SAVE_DIR}' n'existe pas. Il sera créé.")
        # La création se fait dans main()

    # Lancer l'exécution principale
    main()